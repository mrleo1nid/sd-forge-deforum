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
    interpolate_prompts,
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
# All pure functions imported from deforum.utils.prompt_utils
# ============================================================================

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
