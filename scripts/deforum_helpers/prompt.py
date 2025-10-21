import re
import numexpr
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .rendering.util.log_utils import RED, GREEN, PURPLE, RESET_COLOR

# ============================================================================
# PURE FUNCTIONS
# ============================================================================

def check_is_number(value: str) -> bool:
    """Check if string represents a valid float number."""
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value) is not None

def evaluate_weight_expression(expression: str, frame: int, max_frames: int) -> float:
    """Evaluate weight expression using numexpr (t and max_f variables available)."""
    t = frame
    max_f = max_frames  # Used by numexpr
    return float(numexpr.evaluate(expression))

def parse_weight(match: re.Match, frame: int = 0, max_frames: int = 0) -> float:
    """Parse weight from regex match - returns float weight value."""
    w_raw = match.group("weight")
    if w_raw is None:
        return 1.0
    if check_is_number(w_raw):
        return float(w_raw)
    if len(w_raw) < 3:
        # Invalid expression (too short to be valid: needs at least `x`)
        return 1.0
    # Strip backticks and evaluate
    return evaluate_weight_expression(w_raw[1:-1], frame, max_frames)

def split_prompt_into_pos_neg(text: str) -> Tuple[str, str]:
    """Split prompt text into positive and negative parts using --neg delimiter."""
    parts = text.split("--neg", 1)
    if len(parts) > 1:
        return parts[0], parts[1]
    return parts[0], ""

def substitute_weight_expressions(text: str, frame: int, max_frames: int) -> str:
    """Replace all weight expressions in text with evaluated values."""
    math_parser = re.compile(r"(?P<weight>(`[\S\s]*?`))", re.VERBOSE)
    return re.sub(math_parser, lambda m: str(parse_weight(m, frame, max_frames)), text)

def split_weighted_subprompts(text: str, frame: int = 0, max_frames: int = 0) -> Tuple[str, str]:
    """Split prompt into positive/negative after evaluating weight expressions."""
    parsed_prompt = substitute_weight_expressions(text, frame, max_frames)
    return split_prompt_into_pos_neg(parsed_prompt)

def parse_keyframe_number(key: str, max_frames: int) -> int:
    """Parse keyframe number - handles both numeric strings and math expressions."""
    if check_is_number(key):
        return int(float(key))
    # Evaluate expression (max_f variable available in numexpr context)
    max_f = max_frames  # Used by numexpr
    return int(numexpr.evaluate(key))

def parse_animation_prompts_dict(animation_prompts: Dict[str, str], max_frames: int) -> Dict[int, str]:
    """Convert string keys to integer frame numbers."""
    parsed = {}
    for key, value in animation_prompts.items():
        frame_num = parse_keyframe_number(str(key), max_frames)
        parsed[frame_num] = value
    return parsed

def interpolate_prompts(animation_prompts: Dict[str, str], max_frames: int) -> pd.Series:
    """Interpolate prompts between keyframes using composable diffusion."""
    # Parse and sort prompts by keyframe
    parsed_prompts = parse_animation_prompts_dict(animation_prompts, max_frames)
    sorted_prompts = sorted(parsed_prompts.items(), key=lambda item: int(item[0]))

    # Setup container for interpolated prompts
    prompt_series = pd.Series([np.nan for a in range(max_frames)])

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_frame = int(sorted_prompts[i][0])
        next_frame = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_frame >= next_frame:
            print(f"WARNING: Sequential prompt keyframes {i}:{current_frame} and {i + 1}:{next_frame} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]
        current_positive, current_negative, *_ = current_prompt.split("--neg") + [None]
        next_positive, next_negative, *_ = next_prompt.split("--neg") + [None]
        # Calculate how much to shift the weight from current to next prompt at each frame
        weight_step = 1 / (next_frame - current_frame)

        # Apply weighted prompt interpolation for each frame between current and next keyframe
        # using the syntax:  prompt1 :weight1 AND prompt1 :weight2 --neg nprompt1 :weight1 AND nprompt1 :weight2
        # (See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion )
        for f in range(current_frame, next_frame):
            next_weight = weight_step * (f - current_frame)
            current_weight = 1 - next_weight

            # We will build the prompt incrementally depending on which prompts are present
            prompt_series[f] = ''

            # Cater for the case where neither, either or both current & next have positive prompts:
            if current_positive:
                prompt_series[f] += f" ({current_positive}):{current_weight}"
            if current_positive and next_positive:
                prompt_series[f] += f" AND "
            if next_positive:
                prompt_series[f] += f" ({next_positive}):{next_weight}"

            # Cater for the case where neither, either or both current & next have negative prompts:
            if len(current_negative) > 1 or len(next_negative) > 1:
                prompt_series[f] += " --neg "
                if len(current_negative) > 1:
                    prompt_series[f] += f" ({current_negative}):{current_weight}"
                if len(current_negative) > 1 and len(next_negative) > 1:
                    prompt_series[f] += f" AND "
                if len(next_negative) > 1:
                    prompt_series[f] += f" ({next_negative}):{next_weight}"

    # Set explicitly declared keyframe prompts (overwriting interpolated values at the keyframe idx). This ensures:
    # - That final prompt is set, and
    # - Gives us a chance to emit warnings if any keyframe prompts are already using composable diffusion
    for i, prompt in parsed_animation_prompts.items():
        prompt_series[int(i)] = prompt
        if ' AND ' in prompt:
            print(f"WARNING: keyframe {i}'s prompt is using composable diffusion (aka the 'AND' keyword). This will cause unexpected behaviour with interpolation.")

    # Return the filled series, in case max_frames is greater than the last keyframe or any ranges were skipped.
    return prompt_series.ffill().bfill()


def prepare_prompt(prompt_series, max_frames, seed, frame_idx):
    max_f = max_frames - 1
    pattern = r'`.*?`'
    regex = re.compile(pattern)
    prompt_parsed = prompt_series
    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace('t', f'{frame_idx}').replace("max_f", f"{max_f}").replace('`', '')
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))

    prompt_to_print, *after_neg = prompt_parsed.strip().split("--neg")
    prompt_to_print = prompt_to_print.strip()
    after_neg = "".join(after_neg).strip()

    print(f"{GREEN}Seed: {RESET_COLOR}{seed}")
    print(f"{PURPLE}Prompt: {RESET_COLOR}{prompt_to_print}")
    if after_neg and after_neg.strip():
        print(f"{RED}Neg Prompt: {RESET_COLOR}{after_neg}")
        prompt_to_print += f"--neg {after_neg}"

    # set value back into the prompt
    return prompt_to_print
