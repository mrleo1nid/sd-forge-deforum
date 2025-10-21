import re
import numexpr
import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    from .rendering.util.log_utils import RED, GREEN, PURPLE, RESET_COLOR
except ImportError:
    try:
        from rendering.util.log_utils import RED, GREEN, PURPLE, RESET_COLOR
    except (ImportError, ModuleNotFoundError):
        # Fallback for unit tests
        RED = "\033[91m"
        GREEN = "\033[92m"
        PURPLE = "\033[95m"
        RESET_COLOR = "\033[0m"

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
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""

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

def calculate_interpolation_weights(frame: int, start_frame: int, end_frame: int) -> Tuple[float, float]:
    """Calculate current and next weights for linear interpolation between keyframes."""
    weight_step = 1.0 / (end_frame - start_frame)
    next_weight = weight_step * (frame - start_frame)
    current_weight = 1.0 - next_weight
    return current_weight, next_weight

def build_weighted_prompt_part(
    current: str | None,
    next_val: str | None,
    current_weight: float,
    next_weight: float
) -> str:
    """Build weighted prompt using composable diffusion syntax."""
    parts = []
    if current:
        parts.append(f"({current}):{current_weight}")
    if next_val:
        parts.append(f"({next_val}):{next_weight}")
    return " AND ".join(parts)

def build_interpolated_prompt(
    current_prompt: str,
    next_prompt: str,
    current_weight: float,
    next_weight: float
) -> str:
    """Build full interpolated prompt with positive and negative parts."""
    current_pos, current_neg = split_prompt_into_pos_neg(current_prompt)
    next_pos, next_neg = split_prompt_into_pos_neg(next_prompt)

    result_parts = []

    # Build positive part
    pos_part = build_weighted_prompt_part(current_pos, next_pos, current_weight, next_weight)
    if pos_part:
        result_parts.append(pos_part)

    # Build negative part
    has_neg = (current_neg and len(current_neg.strip()) > 0) or (next_neg and len(next_neg.strip()) > 0)
    if has_neg:
        neg_part = build_weighted_prompt_part(current_neg, next_neg, current_weight, next_weight)
        if neg_part:
            result_parts.append(f"--neg {neg_part}")

    return " ".join(result_parts)

def interpolate_prompts(animation_prompts: Dict[str, str], max_frames: int) -> pd.Series:
    """Interpolate prompts between keyframes using composable diffusion."""
    parsed_prompts = parse_animation_prompts_dict(animation_prompts, max_frames)
    sorted_prompts = sorted(parsed_prompts.items(), key=lambda item: int(item[0]))

    prompt_series = pd.Series([np.nan for _ in range(max_frames)])

    # Interpolate between consecutive keyframes
    for i in range(len(sorted_prompts) - 1):
        current_frame = int(sorted_prompts[i][0])
        next_frame = int(sorted_prompts[i + 1][0])

        if current_frame >= next_frame:
            continue  # Skip invalid ordering

        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        for f in range(current_frame, next_frame):
            current_weight, next_weight = calculate_interpolation_weights(f, current_frame, next_frame)
            prompt_series[f] = build_interpolated_prompt(
                current_prompt, next_prompt, current_weight, next_weight
            )

    # Set keyframe prompts (overwrite interpolated values)
    for frame_num, prompt in parsed_prompts.items():
        prompt_series[int(frame_num)] = prompt

    return prompt_series.ffill().bfill()

def evaluate_prompt_expression(expression: str, frame_idx: int, max_frames: int) -> str:
    """Evaluate math expression in prompt (replaces t and max_f variables)."""
    max_f = max_frames - 1
    expression_clean = expression.replace('t', f'{frame_idx}').replace("max_f", f"{max_f}")
    return str(numexpr.evaluate(expression_clean))

def substitute_prompt_expressions(text: str, frame_idx: int, max_frames: int) -> str:
    """Replace all backtick-wrapped expressions in prompt with evaluated values."""
    pattern = r'`.*?`'
    regex = re.compile(pattern)
    result = text
    for match in regex.finditer(text):
        matched_string = match.group(0)
        expression = matched_string.strip('`')
        evaluated = evaluate_prompt_expression(expression, frame_idx, max_frames)
        result = result.replace(matched_string, evaluated)
    return result

# ============================================================================
# IMPURE FUNCTIONS (side effects: printing)
# ============================================================================

def prepare_prompt(prompt_series: str, max_frames: int, seed: int, frame_idx: int) -> str:
    """Evaluate prompt expressions and print formatted prompt with seed info."""
    prompt_parsed = substitute_prompt_expressions(prompt_series, frame_idx, max_frames)

    prompt_to_print, *after_neg = prompt_parsed.strip().split("--neg")
    prompt_to_print = prompt_to_print.strip()
    after_neg = "".join(after_neg).strip()

    print(f"{GREEN}Seed: {RESET_COLOR}{seed}")
    print(f"{PURPLE}Prompt: {RESET_COLOR}{prompt_to_print}")
    if after_neg and after_neg.strip():
        print(f"{RED}Neg Prompt: {RESET_COLOR}{after_neg}")
        prompt_to_print += f"--neg {after_neg}"

    return prompt_to_print
