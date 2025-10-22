import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Import pure functions from refactored utils module
from deforum.utils.prompt_utils import (
    check_is_number,
    evaluate_weight_expression,
    parse_weight,
    split_prompt_into_pos_neg,
    substitute_weight_expressions,
    split_weighted_subprompts,
    parse_keyframe_number,
    parse_animation_prompts_dict,
    calculate_interpolation_weights,
    build_weighted_prompt_part,
    build_interpolated_prompt,
    evaluate_prompt_expression,
    substitute_prompt_expressions,
)

# Re-export for backward compatibility
__all__ = [
    'check_is_number',
    'evaluate_weight_expression',
    'parse_weight',
    'split_prompt_into_pos_neg',
    'substitute_weight_expressions',
    'split_weighted_subprompts',
    'parse_keyframe_number',
    'parse_animation_prompts_dict',
    'calculate_interpolation_weights',
    'build_weighted_prompt_part',
    'build_interpolated_prompt',
    'evaluate_prompt_expression',
    'substitute_prompt_expressions',
    'interpolate_prompts',
    'prepare_prompt',
]

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
# COMPLEX PURE FUNCTIONS (not yet extracted)
# ============================================================================

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
