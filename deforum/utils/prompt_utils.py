"""Pure functions for prompt parsing and interpolation.

This module contains prompt-related pure functions extracted from
scripts/deforum_helpers/prompt.py, following functional programming principles
with no side effects.
"""

import re
import numexpr
import pandas as pd
import numpy as np
from typing import Tuple, Dict

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


def parse_weight(match: re.Match, frame: int = 0, max_frames: int = 0) -> float:
    """Parse weight from regex match - returns float weight value.

    Args:
        match: Regex match object with 'weight' group
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Parsed weight (1.0 if None or invalid)
    """
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
    """Replace all weight expressions in text with evaluated values.

    Args:
        text: Prompt text with embedded weight expressions in backticks
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Text with expressions replaced by numeric values
    """
    math_parser = re.compile(r"(?P<weight>(`[\S\s]*?`))", re.VERBOSE)
    return re.sub(math_parser, lambda m: str(parse_weight(m, frame, max_frames)), text)


def split_weighted_subprompts(text: str, frame: int = 0, max_frames: int = 0) -> Tuple[str, str]:
    """Split prompt into positive/negative after evaluating weight expressions.

    Args:
        text: Full prompt text
        frame: Current frame number
        max_frames: Total frames

    Returns:
        Tuple of (positive_prompt, negative_prompt) with weights evaluated
    """
    parsed_prompt = substitute_weight_expressions(text, frame, max_frames)
    return split_prompt_into_pos_neg(parsed_prompt)


# ============================================================================
# KEYFRAME PARSING
# ============================================================================


def parse_keyframe_number(key: str, max_frames: int) -> int:
    """Parse keyframe number - handles both numeric strings and math expressions.

    Args:
        key: Keyframe as string or expression
        max_frames: Total frames for expression evaluation

    Returns:
        Keyframe number as integer
    """
    if check_is_number(key):
        return int(float(key))
    # Evaluate expression (max_f variable available in numexpr context)
    max_f = max_frames  # Used by numexpr
    return int(numexpr.evaluate(key))


def parse_animation_prompts_dict(animation_prompts: Dict[str, str], max_frames: int) -> Dict[int, str]:
    """Convert string keys to integer frame numbers.

    Args:
        animation_prompts: Dict with string keys and prompt values
        max_frames: Total frames for expression evaluation

    Returns:
        Dict with integer keys and prompt values
    """
    parsed = {}
    for key, value in animation_prompts.items():
        frame_num = parse_keyframe_number(str(key), max_frames)
        parsed[frame_num] = value
    return parsed


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
    """Build weighted prompt using composable diffusion syntax.

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
    """Build full interpolated prompt with positive and negative parts.

    Args:
        current_prompt: Current keyframe prompt (may contain --neg)
        next_prompt: Next keyframe prompt (may contain --neg)
        current_weight: Weight for current prompts
        next_weight: Weight for next prompts

    Returns:
        Complete interpolated prompt string with "--neg" if needed
    """
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


# ============================================================================
# EXPRESSION SUBSTITUTION
# ============================================================================


def evaluate_prompt_expression(expression: str, frame_idx: int, max_frames: int) -> str:
    """Evaluate math expression in prompt (replaces t and max_f variables).

    Args:
        expression: Expression to evaluate
        frame_idx: Current frame number
        max_frames: Total frames

    Returns:
        Evaluated result as string
    """
    max_f = max_frames - 1
    t = frame_idx / max_f if max_f > 0 else 0
    try:
        result = numexpr.evaluate(expression, local_dict={'frame': frame_idx, 'max_f': max_frames, 't': t})
        return str(result)
    except Exception:
        return expression


def substitute_prompt_expressions(text: str, frame_idx: int, max_frames: int) -> str:
    """Substitute all backtick expressions in prompt text.

    Args:
        text: Prompt text with `expression` patterns
        frame_idx: Current frame number
        max_frames: Total frames

    Returns:
        Text with all expressions evaluated and substituted
    """
    pattern = r"`([^`]+)`"

    def replace_expr(match):
        expr = match.group(1)
        return evaluate_prompt_expression(expr, frame_idx, max_frames)

    return re.sub(pattern, replace_expr, text)

# ============================================================================
# PROMPT INTERPOLATION
# ============================================================================


def interpolate_prompts(animation_prompts: Dict[str, str], max_frames: int) -> pd.Series:
    """Interpolate prompts between keyframes using composable diffusion.

    Takes a dictionary of frame-indexed prompts and creates a smooth
    interpolation between keyframes using weighted composable diffusion.

    Args:
        animation_prompts: Dictionary mapping frame numbers (as strings) to prompt text
        max_frames: Total number of frames in animation

    Returns:
        pandas Series with interpolated prompts for each frame

    Examples:
        >>> prompts = {"0": "cat", "10": "dog"}
        >>> series = interpolate_prompts(prompts, 20)
        >>> series[0]
        'cat'
        >>> series[10]
        'dog'
        >>> # Frames 1-9 will have interpolated weights between cat and dog
    """
    parsed_prompts = parse_animation_prompts_dict(animation_prompts, max_frames)
    sorted_prompts = sorted(parsed_prompts.items(), key=lambda item: int(item[0]))

    # Use dtype=object to allow string values (fixes pandas FutureWarning)
    prompt_series = pd.Series([np.nan for _ in range(max_frames)], dtype=object)

    # Interpolate between consecutive keyframes
    for i in range(len(sorted_prompts) - 1):
        current_frame = int(sorted_prompts[i][0])
        next_frame = int(sorted_prompts[i + 1][0])

        if current_frame >= next_frame:
            continue  # Skip invalid ordering

        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        for f in range(current_frame, next_frame):
            current_weight, next_weight = calculate_interpolation_weights(
                f, current_frame, next_frame
            )
            prompt_series[f] = build_interpolated_prompt(
                current_prompt, next_prompt, current_weight, next_weight
            )

    # Set keyframe prompts (overwrite interpolated values)
    for frame_num, prompt in parsed_prompts.items():
        prompt_series[int(frame_num)] = prompt

    return prompt_series.ffill().bfill()
