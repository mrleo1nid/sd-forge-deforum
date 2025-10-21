"""Pure functions for prompt parsing and interpolation.

This module contains prompt-related pure functions extracted from
scripts/deforum_helpers/prompt.py, following functional programming principles
with no side effects.
"""

import re
import numexpr
import pandas as pd
from typing import Tuple

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def check_is_number(value: str) -> bool:
    """Check if string represents a valid float number.

    Uses regex to properly validate float format.

    Args:
        value: String to check

    Returns:
        True if string is a valid float number format
    """
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value) is not None


# ============================================================================
# WEIGHT EXPRESSION EVALUATION
# ============================================================================


def evaluate_weight_expression(expr: str, frame: int, max_frames: int) -> float:
    """Evaluate mathematical expression with frame variables.

    Supported variables: t (normalized time 0-1), frame

    Args:
        expr: Mathematical expression string
        frame: Current frame number
        max_frames: Total frames in animation

    Returns:
        Evaluated weight as float
    """
    t = frame / (max_frames - 1) if max_frames > 1 else 0
    return float(numexpr.evaluate(expr, local_dict={"t": t, "frame": frame}))


def parse_weight(weight_str: str | None, frame: int, max_frames: int) -> float:
    """Parse weight string into float value.

    Handles numeric strings, expressions, and None.

    Args:
        weight_str: Weight as string, expression, or None
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Parsed weight (1.0 if None)
    """
    if weight_str is None or weight_str.strip() == "":
        return 1.0

    if check_is_number(weight_str):
        return float(weight_str)

    try:
        return evaluate_weight_expression(weight_str, frame, max_frames)
    except Exception:
        return 1.0


# ============================================================================
# PROMPT PARSING
# ============================================================================


def split_prompt_into_pos_neg(text: str) -> Tuple[str, str]:
    """Split prompt text into positive and negative parts.

    Uses '--neg' delimiter to separate positive from negative prompts.

    Args:
        text: Prompt text potentially containing '--neg' delimiter

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    parts = text.split("--neg", 1)
    if len(parts) > 1:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""


def substitute_weight_expressions(text: str, frame: int, max_frames: int) -> str:
    """Substitute `{expression}` patterns with evaluated weights.

    Args:
        text: Prompt text with embedded weight expressions
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Text with expressions replaced by numeric values
    """
    pattern = r"`([^`]+)`"

    def replace_expr(match):
        expr = match.group(1)
        try:
            value = evaluate_weight_expression(expr, frame, max_frames)
            return str(value)
        except Exception:
            return match.group(0)

    return re.sub(pattern, replace_expr, text)


def split_weighted_subprompts(text: str, frame: int, max_frames: int) -> list[Tuple[str, float]]:
    """Split prompt into weighted subprompts.

    Handles both positive and negative prompts with optional weights.

    Args:
        text: Full prompt text
        frame: Current frame number
        max_frames: Total frames

    Returns:
        List of (prompt_text, weight) tuples
    """
    positive, negative = split_prompt_into_pos_neg(text)
    parts = []

    # Add positive prompt
    if positive:
        parts.append((positive, 1.0))

    # Add negative prompt
    if negative:
        parts.append((negative, -1.0))

    return parts


# ============================================================================
# KEYFRAME PARSING
# ============================================================================


def parse_keyframe_number(key_str: str, max_frames: int) -> int:
    """Parse keyframe number from string (numeric or expression).

    Args:
        key_str: Keyframe as string or expression
        max_frames: Total frames for expression evaluation

    Returns:
        Keyframe number as integer
    """
    if check_is_number(key_str):
        return int(key_str)

    try:
        # Evaluate expression with max_f variable
        return int(numexpr.evaluate(key_str, local_dict={"max_f": max_frames}))
    except Exception:
        return 0


def parse_animation_prompts_dict(prompt_dict: dict, max_frames: int) -> dict[int, str]:
    """Parse animation prompts dictionary with expression support.

    Converts string keys (including expressions) to integer frame numbers.

    Args:
        prompt_dict: Dict with string keys and prompt values
        max_frames: Total frames for expression evaluation

    Returns:
        Dict with integer keys and prompt values
    """
    result = {}
    for key_str, prompt in prompt_dict.items():
        frame_num = parse_keyframe_number(str(key_str), max_frames)
        result[frame_num] = prompt
    return result


# ============================================================================
# PROMPT INTERPOLATION
# ============================================================================


def calculate_interpolation_weights(
    frame: int, start_frame: int, end_frame: int
) -> Tuple[float, float]:
    """Calculate current and next weights for linear interpolation.

    Args:
        frame: Current frame number
        start_frame: Starting keyframe
        end_frame: Ending keyframe

    Returns:
        Tuple of (current_weight, next_weight) summing to 1.0
    """
    weight_step = 1.0 / (end_frame - start_frame)
    next_weight = weight_step * (frame - start_frame)
    current_weight = 1.0 - next_weight
    return current_weight, next_weight


def build_weighted_prompt_part(
    current: str | None,
    next_val: str | None,
    current_weight: float,
    next_weight: float,
) -> str:
    """Build weighted prompt part using composable diffusion syntax.

    Args:
        current: Current keyframe prompt
        next_val: Next keyframe prompt
        current_weight: Weight for current prompt
        next_weight: Weight for next prompt

    Returns:
        Weighted prompt string like "(prompt1):0.7 AND (prompt2):0.3"
    """
    parts = []
    if current:
        parts.append(f"({current}):{current_weight:.3f}")
    if next_val:
        parts.append(f"({next_val}):{next_weight:.3f}")
    return " AND ".join(parts)


def build_interpolated_prompt(
    current_positive: str | None,
    next_positive: str | None,
    current_negative: str | None,
    next_negative: str | None,
    current_weight: float,
    next_weight: float,
) -> Tuple[str, str]:
    """Build interpolated positive and negative prompts.

    Args:
        current_positive: Current positive prompt
        next_positive: Next positive prompt
        current_negative: Current negative prompt
        next_negative: Next negative prompt
        current_weight: Weight for current prompts
        next_weight: Weight for next prompts

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    positive = build_weighted_prompt_part(
        current_positive, next_positive, current_weight, next_weight
    )
    negative = build_weighted_prompt_part(
        current_negative, next_negative, current_weight, next_weight
    )
    return positive, negative


# ============================================================================
# EXPRESSION SUBSTITUTION
# ============================================================================


def evaluate_prompt_expression(expr: str, frame: int, max_frames: int) -> str:
    """Evaluate backtick expression in prompt.

    Supports: t (normalized time), frame, max_f (max frames)

    Args:
        expr: Expression to evaluate
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Evaluated result as string
    """
    try:
        t = frame / (max_frames - 1) if max_frames > 1 else 0
        result = numexpr.evaluate(
            expr, local_dict={"t": t, "frame": frame, "max_f": max_frames}
        )
        return str(result)
    except Exception:
        return expr


def substitute_prompt_expressions(text: str, frame: int, max_frames: int) -> str:
    """Substitute all backtick expressions in prompt text.

    Args:
        text: Prompt text with `expression` patterns
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Text with all expressions evaluated and substituted
    """
    pattern = r"`([^`]+)`"

    def replace_expr(match):
        expr = match.group(1)
        return evaluate_prompt_expression(expr, frame, max_frames)

    return re.sub(pattern, replace_expr, text)
