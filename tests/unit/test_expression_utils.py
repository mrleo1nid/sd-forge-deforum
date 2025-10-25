"""Unit tests for deforum.utils.expression_utils module."""

import pytest

from deforum.utils.parsing.expressions import (
    parse_embedded_expressions,
    parse_frame_expression,
    has_embedded_expressions,
    extract_expressions,
    validate_expression,
)


class TestParseEmbeddedExpressions:
    """Tests for parse_embedded_expressions function."""

    def test_simple_expression(self):
        """Should evaluate simple arithmetic expression."""
        result = parse_embedded_expressions("value `1+1`", {})
        assert result == "value 2"

    def test_variable_substitution(self):
        """Should substitute variables in expression."""
        result = parse_embedded_expressions("frame `t*2`", {"t": 5})
        assert result == "frame 10"

    def test_multiple_variables(self):
        """Should substitute multiple variables."""
        result = parse_embedded_expressions(
            "calc `t + max_f`",
            {"t": 10, "max_f": 100}
        )
        assert result == "calc 110"

    def test_multiple_expressions(self):
        """Should evaluate multiple expressions in one string."""
        result = parse_embedded_expressions(
            "`t` out of `max_f`",
            {"t": 25, "max_f": 100}
        )
        assert result == "25 out of 100"

    def test_no_expression(self):
        """Should return unchanged string if no expressions."""
        result = parse_embedded_expressions("static text", {"t": 5})
        assert result == "static text"

    def test_floating_point_expression(self):
        """Should handle floating point calculations."""
        result = parse_embedded_expressions("progress `t/max_f`", {"t": 25, "max_f": 100})
        assert result == "progress 0.25"

    def test_complex_expression(self):
        """Should evaluate complex arithmetic."""
        result = parse_embedded_expressions(
            "`(t*2 + max_f/10)**2`",
            {"t": 5, "max_f": 100}
        )
        assert result == "400.0"  # (5*2 + 100/10)**2 = (10+10)**2 = 400

    def test_expression_with_spaces(self):
        """Should handle expressions with spaces."""
        result = parse_embedded_expressions(
            "value ` t + 10 `",
            {"t": 5}
        )
        assert result == "value 15"

    def test_empty_expression(self):
        """Should handle empty backticks."""
        # numexpr.evaluate with empty string might raise error
        with pytest.raises(ValueError):
            parse_embedded_expressions("empty ``", {})

    def test_invalid_expression_raises_error(self):
        """Should raise ValueError for invalid expression."""
        with pytest.raises(ValueError, match="Failed to evaluate"):
            parse_embedded_expressions("bad `invalid syntax !`", {})


class TestParseFrameExpression:
    """Tests for parse_frame_expression function."""

    def test_frame_index_substitution(self):
        """Should substitute t with frame index."""
        result = parse_frame_expression("frame `t`", 10, 100)
        assert result == "frame 10"

    def test_max_frames_substitution(self):
        """Should substitute max_f with max frames."""
        result = parse_frame_expression("total `max_f`", 10, 100)
        assert result == "total 100"

    def test_both_variables(self):
        """Should substitute both t and max_f."""
        result = parse_frame_expression("progress `t/max_f`", 25, 100)
        assert result == "progress 0.25"

    def test_zero_frame_index(self):
        """Should handle frame index of 0."""
        result = parse_frame_expression("`t` `max_f`", 0, 50)
        assert result == "0 50"

    def test_last_frame_index(self):
        """Should handle last frame index."""
        result = parse_frame_expression("`t` of `max_f`", 99, 100)
        assert result == "99 of 100"

    def test_complex_frame_expression(self):
        """Should evaluate complex frame-based expression."""
        result = parse_frame_expression("`t*2 + max_f/10`", 5, 100)
        assert result == "20.0"  # 5*2 + 100/10 = 10 + 10 = 20

    def test_no_expression(self):
        """Should return unchanged string without expressions."""
        result = parse_frame_expression("static prompt", 10, 100)
        assert result == "static prompt"


class TestHasEmbeddedExpressions:
    """Tests for has_embedded_expressions function."""

    def test_with_expression(self):
        """Should return True for string with expression."""
        assert has_embedded_expressions("frame `t`")

    def test_without_expression(self):
        """Should return False for string without expression."""
        assert not has_embedded_expressions("static text")

    def test_multiple_expressions(self):
        """Should return True for multiple expressions."""
        assert has_embedded_expressions("`expr1` and `expr2`")

    def test_empty_string(self):
        """Should return False for empty string."""
        assert not has_embedded_expressions("")

    def test_single_backtick(self):
        """Should return False for unpaired backtick."""
        assert not has_embedded_expressions("one ` but not two")

    def test_nested_backticks_in_expression(self):
        """Should handle backticks inside expression correctly."""
        # This tests the non-greedy regex behavior
        result = has_embedded_expressions("`a` text `b`")
        assert result

    def test_empty_expression(self):
        """Should return True for empty backticks."""
        assert has_embedded_expressions("empty ``")


class TestExtractExpressions:
    """Tests for extract_expressions function."""

    def test_single_expression(self):
        """Should extract single expression."""
        result = extract_expressions("frame `t`")
        assert result == ['t']

    def test_multiple_expressions(self):
        """Should extract multiple expressions."""
        result = extract_expressions("`t` out of `max_f`")
        assert result == ['t', 'max_f']

    def test_no_expressions(self):
        """Should return empty list for no expressions."""
        result = extract_expressions("static text")
        assert result == []

    def test_complex_expression(self):
        """Should extract complex expression content."""
        result = extract_expressions("calc `t*2 + max_f/10`")
        assert result == ['t*2 + max_f/10']

    def test_empty_expression(self):
        """Should extract empty string from empty backticks."""
        result = extract_expressions("empty ``")
        assert result == ['']

    def test_unpaired_backtick(self):
        """Should not extract unpaired backticks."""
        result = extract_expressions("one ` but not two")
        assert result == []

    def test_expression_with_spaces(self):
        """Should preserve spaces in extracted expressions."""
        result = extract_expressions("value ` t + 10 `")
        assert result == [' t + 10 ']


class TestValidateExpression:
    """Tests for validate_expression function."""

    def test_valid_simple_expression(self):
        """Should return True for valid arithmetic."""
        assert validate_expression("1 + 1")

    def test_valid_complex_expression(self):
        """Should return True for complex valid expression."""
        assert validate_expression("(x*2 + y/10)**2", {"x": 5, "y": 100})

    def test_invalid_syntax(self):
        """Should return False for invalid syntax."""
        assert not validate_expression("invalid syntax !")

    def test_empty_expression(self):
        """Should return True for empty expression."""
        assert validate_expression("")

    def test_whitespace_expression(self):
        """Should return True for whitespace-only expression."""
        assert validate_expression("   ")

    def test_division_by_zero(self):
        """Should return False for division by zero."""
        # numexpr should raise an error for division by zero
        assert not validate_expression("1/0")

    def test_undefined_variable(self):
        """Should return False for undefined variable."""
        assert not validate_expression("undefined_var * 2")

    def test_variable_substitution(self):
        """Should validate with variable substitution."""
        assert validate_expression("t * 2", {"t": 5})

    def test_multiple_operations(self):
        """Should validate complex mathematical operations."""
        # numexpr supports basic arithmetic but not all numpy functions
        assert validate_expression("(t**2 + max_f) / 2", {"t": 5, "max_f": 100})


class TestExpressionUtilsIntegration:
    """Integration tests combining multiple expression utilities."""

    def test_parse_extract_validate_pipeline(self):
        """Should work through extract, validate, and parse pipeline."""
        text = "frame `t` progress `t/max_f`"

        # Check for expressions
        assert has_embedded_expressions(text)

        # Extract expressions
        expressions = extract_expressions(text)
        assert len(expressions) == 2
        assert expressions[0] == 't'
        assert expressions[1] == 't/max_f'

        # Validate expressions
        variables = {"t": 10, "max_f": 100}
        for expr in expressions:
            assert validate_expression(expr, variables)

        # Parse full text
        result = parse_embedded_expressions(text, variables)
        assert result == "frame 10 progress 0.1"

    def test_frame_expression_workflow(self):
        """Should handle typical frame expression workflow."""
        prompt_template = "A scene at frame `t` (`(t/max_f)*100`% complete)"

        # Check if needs parsing
        if has_embedded_expressions(prompt_template):
            # Parse for specific frame
            parsed = parse_frame_expression(prompt_template, 25, 100)
            assert "frame 25" in parsed
            assert "25.0% complete" in parsed

    def test_validate_before_parse(self):
        """Should validate expressions before parsing."""
        template = "`t*2` and `invalid !`"
        expressions = extract_expressions(template)

        # First expression is valid
        assert validate_expression(expressions[0], {"t": 5})

        # Second expression is invalid
        assert not validate_expression(expressions[1])

        # Parsing should fail on invalid expression
        with pytest.raises(ValueError):
            parse_embedded_expressions(template, {"t": 5})

    def test_conditional_parsing(self):
        """Should conditionally parse based on expression presence."""
        static_text = "no expressions here"
        dynamic_text = "frame `t`"

        # Skip parsing if no expressions
        if not has_embedded_expressions(static_text):
            assert static_text == static_text  # No parsing needed

        # Parse if expressions present
        if has_embedded_expressions(dynamic_text):
            parsed = parse_frame_expression(dynamic_text, 10, 100)
            assert parsed == "frame 10"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
