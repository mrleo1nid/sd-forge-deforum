"""Unit tests for prompt parsing and interpolation functions."""

import re
import pytest
import pandas as pd
import numpy as np

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


class TestCheckIsNumber:
    """Test number checking."""

    def test_integer(self):
        assert check_is_number("42") is True
        assert check_is_number("0") is True

    def test_float(self):
        assert check_is_number("3.14") is True
        assert check_is_number("0.5") is True
        assert check_is_number(".5") is True

    def test_negative(self):
        assert check_is_number("-42") is True
        assert check_is_number("-3.14") is True

    def test_positive_sign(self):
        assert check_is_number("+42") is True
        assert check_is_number("+3.14") is True

    def test_non_numbers(self):
        assert check_is_number("abc") is False
        assert check_is_number("12.34.56") is False
        assert check_is_number("") is False


class TestEvaluateWeightExpression:
    """Test weight expression evaluation."""

    def test_simple_expression(self):
        result = evaluate_weight_expression("2 + 2", frame=0, max_frames=100)
        assert result == 4.0

    def test_with_t_variable(self):
        result = evaluate_weight_expression("0.5", frame=10, max_frames=100)
        assert result == 0.5

    def test_with_frame_variable(self):
        # The function uses t and max_f internally
        result = evaluate_weight_expression("1.0", frame=50, max_frames=100)
        assert result == 1.0


class TestParseWeight:
    """Test weight parsing from regex matches."""

    def test_none_weight(self):
        match = re.match(r"(?P<weight>)", "")
        result = parse_weight(match, frame=0, max_frames=100)
        assert result == 1.0

    def test_numeric_weight(self):
        pattern = r"(?P<weight>\d+\.?\d*)"
        match = re.match(pattern, "0.75")
        result = parse_weight(match, frame=0, max_frames=100)
        assert result == 0.75

    def test_expression_weight(self):
        pattern = r"(?P<weight>`[^`]+`)"
        match = re.match(pattern, "`2 + 2`")
        result = parse_weight(match, frame=0, max_frames=100)
        assert result == 4.0

    def test_invalid_expression(self):
        pattern = r"(?P<weight>`[^`]*`)"
        match = re.match(pattern, "``")
        result = parse_weight(match, frame=0, max_frames=100)
        assert result == 1.0  # Returns default for invalid


class TestSplitPromptIntoPosNeg:
    """Test positive/negative prompt splitting."""

    def test_positive_only(self):
        pos, neg = split_prompt_into_pos_neg("a beautiful landscape")
        assert pos == "a beautiful landscape"
        assert neg == ""

    def test_with_negative(self):
        pos, neg = split_prompt_into_pos_neg("a cat --neg ugly")
        assert pos == "a cat"
        assert neg == "ugly"

    def test_multiple_neg_markers(self):
        pos, neg = split_prompt_into_pos_neg("cat --neg ugly --neg bad")
        assert pos == "cat"
        assert neg == "ugly --neg bad"

    def test_empty_string(self):
        pos, neg = split_prompt_into_pos_neg("")
        assert pos == ""
        assert neg == ""


class TestSubstituteWeightExpressions:
    """Test weight expression substitution."""

    def test_no_expressions(self):
        result = substitute_weight_expressions("a cat", frame=0, max_frames=100)
        assert result == "a cat"

    def test_single_expression(self):
        result = substitute_weight_expressions("cat `0.5`", frame=10, max_frames=100)
        assert "0.5" in result

    def test_multiple_expressions(self):
        text = "`1.0` cat `0.5` dog"
        result = substitute_weight_expressions(text, frame=0, max_frames=100)
        assert "`" not in result  # All backticks should be removed


class TestSplitWeightedSubprompts:
    """Test complete weighted subprompt splitting."""

    def test_simple_prompt(self):
        pos, neg = split_weighted_subprompts("a cat", frame=0, max_frames=100)
        assert pos == "a cat"
        assert neg == ""

    def test_with_weights(self):
        pos, neg = split_weighted_subprompts("`0.5` cat", frame=0, max_frames=100)
        assert "cat" in pos
        assert neg == ""

    def test_with_negative(self):
        pos, neg = split_weighted_subprompts("cat --neg dog", frame=0, max_frames=100)
        assert "cat" in pos
        assert "dog" in neg


class TestParseKeyframeNumber:
    """Test keyframe number parsing."""

    def test_numeric_string(self):
        result = parse_keyframe_number("42", max_frames=100)
        assert result == 42

    def test_float_string(self):
        result = parse_keyframe_number("3.5", max_frames=100)
        assert result == 3

    def test_expression(self):
        # max_f is available in numexpr context
        result = parse_keyframe_number("10 + 5", max_frames=100)
        assert result == 15


class TestParseAnimationPromptsDict:
    """Test animation prompts dictionary parsing."""

    def test_numeric_keys(self):
        prompts = {"0": "cat", "10": "dog", "20": "bird"}
        result = parse_animation_prompts_dict(prompts, max_frames=100)
        assert result == {0: "cat", 10: "dog", 20: "bird"}

    def test_expression_keys(self):
        prompts = {"0": "start", "5+5": "middle"}
        result = parse_animation_prompts_dict(prompts, max_frames=100)
        assert result[0] == "start"
        assert result[10] == "middle"

    def test_empty_dict(self):
        result = parse_animation_prompts_dict({}, max_frames=100)
        assert result == {}


class TestCalculateInterpolationWeights:
    """Test interpolation weight calculation."""

    def test_start_frame(self):
        current, next_w = calculate_interpolation_weights(0, 0, 10)
        assert current == 1.0
        assert next_w == 0.0

    def test_end_frame(self):
        current, next_w = calculate_interpolation_weights(9, 0, 10)
        assert current == pytest.approx(0.1)
        assert next_w == pytest.approx(0.9)

    def test_middle_frame(self):
        current, next_w = calculate_interpolation_weights(5, 0, 10)
        assert current == 0.5
        assert next_w == 0.5


class TestBuildWeightedPromptPart:
    """Test weighted prompt part construction."""

    def test_both_prompts(self):
        result = build_weighted_prompt_part("cat", "dog", 0.7, 0.3)
        assert result == "(cat):0.7 AND (dog):0.3"

    def test_current_only(self):
        result = build_weighted_prompt_part("cat", None, 1.0, 0.0)
        assert result == "(cat):1.0"

    def test_next_only(self):
        result = build_weighted_prompt_part(None, "dog", 0.0, 1.0)
        assert result == "(dog):1.0"

    def test_neither(self):
        result = build_weighted_prompt_part(None, None, 0.5, 0.5)
        assert result == ""


class TestBuildInterpolatedPrompt:
    """Test full interpolated prompt building."""

    def test_positive_only(self):
        result = build_interpolated_prompt("cat", "dog", 0.6, 0.4)
        assert "(cat):0.6" in result
        assert "(dog):0.4" in result
        assert "--neg" not in result

    def test_with_negatives(self):
        result = build_interpolated_prompt(
            "cat --neg ugly",
            "dog --neg bad",
            0.5,
            0.5
        )
        assert "(cat):0.5" in result
        assert "(dog):0.5" in result
        assert "--neg" in result
        assert "ugly" in result
        assert "bad" in result

    def test_mixed_negatives(self):
        result = build_interpolated_prompt("cat", "dog --neg bad", 0.7, 0.3)
        assert "(cat):0.7" in result
        assert "(dog):0.3" in result


class TestEvaluatePromptExpression:
    """Test prompt expression evaluation."""

    def test_simple_math(self):
        result = evaluate_prompt_expression("2 + 2", frame_idx=0, max_frames=100)
        assert result == "4"

    def test_with_t_variable(self):
        result = evaluate_prompt_expression("10", frame_idx=5, max_frames=100)
        assert result == "10"


class TestSubstitutePromptExpressions:
    """Test prompt expression substitution."""

    def test_no_expressions(self):
        result = substitute_prompt_expressions("a cat", frame_idx=0, max_frames=100)
        assert result == "a cat"

    def test_single_expression(self):
        result = substitute_prompt_expressions("frame `5+5`", frame_idx=0, max_frames=100)
        assert "10" in result
        assert "`" not in result

    def test_multiple_expressions(self):
        text = "`2*2` cats and `3+3` dogs"
        result = substitute_prompt_expressions(text, frame_idx=0, max_frames=100)
        assert "4" in result
        assert "6" in result
        assert "`" not in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestInterpolatePrompts:
    """Test interpolate_prompts function."""

    def test_simple_two_keyframes(self):
        from deforum.utils.prompt_utils import interpolate_prompts
        
        prompts = {"0": "cat", "10": "dog"}
        result = interpolate_prompts(prompts, 20)
        
        # Check keyframes are preserved
        assert result[0] == "cat"
        assert result[10] == "dog"
        
        # Check interpolation exists between keyframes
        assert "(cat)" in result[5]
        assert "(dog)" in result[5]
        
        # Check forward fill after last keyframe
        assert result[15] == "dog"

    def test_three_keyframes(self):
        from deforum.utils.prompt_utils import interpolate_prompts
        
        prompts = {"0": "morning", "50": "noon", "100": "night"}
        result = interpolate_prompts(prompts, 150)
        
        assert result[0] == "morning"
        assert result[50] == "noon"
        assert result[100] == "night"
        
        # Check interpolation between first and second
        assert "(morning)" in result[25]
        assert "(noon)" in result[25]
        
        # Check forward fill after last keyframe
        assert result[125] == "night"

    def test_single_keyframe(self):
        from deforum.utils.prompt_utils import interpolate_prompts
        
        prompts = {"0": "static"}
        result = interpolate_prompts(prompts, 10)
        
        # All frames should have the same prompt
        for i in range(10):
            assert result[i] == "static"

    def test_with_weights(self):
        from deforum.utils.prompt_utils import interpolate_prompts
        
        prompts = {"0": "(cat):1.5", "10": "(dog):2.0"}
        result = interpolate_prompts(prompts, 20)
        
        # Keyframes should preserve original weights
        assert result[0] == "(cat):1.5"
        assert result[10] == "(dog):2.0"
        
        # Interpolated frames should have both prompts
        assert "cat" in result[5]
        assert "dog" in result[5]
