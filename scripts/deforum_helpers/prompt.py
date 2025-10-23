"""Legacy wrapper for prompt logic - imports from deforum.core.prompts and deforum.utils.prompt_utils.

This module now imports prompt scheduling logic from the refactored core module
and pure utility functions from the utils module.
"""

# Import core prompt scheduling logic
from deforum.core.prompts import (
    PromptScheduler,
    prepare_prompt,
)

# Import pure utility functions
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
    'PromptScheduler',
    'prepare_prompt',
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
]
