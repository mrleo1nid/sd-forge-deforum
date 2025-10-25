"""Unit tests for string manipulation and formatting functions."""

import re
import pytest

from deforum.utils.parsing.strings import (
    get_os,
    custom_placeholder_format,
    clean_gradio_path_strings,
    tick_or_cross,
    sanitize_keyframe_value,
)


class TestGetOs:
    """Test get_os function."""

    def test_returns_valid_os_name(self):
        result = get_os()
        assert result in ["Windows", "Linux", "Mac", "Unknown"]

    def test_returns_string(self):
        result = get_os()
        assert isinstance(result, str)


class TestCustomPlaceholderFormat:
    """Test custom_placeholder_format function."""

    def test_simple_string_value(self):
        match = re.match(r'{(\w+)}', '{name}')
        result = custom_placeholder_format({'name': 'test'}, match)
        assert result == 'test'

    def test_none_value_returns_underscore(self):
        match = re.match(r'{(\w+)}', '{name}')
        result = custom_placeholder_format({'name': None}, match)
        assert result == '_'

    def test_missing_key_returns_key(self):
        match = re.match(r'{(\w+)}', '{missing}')
        result = custom_placeholder_format({}, match)
        assert result == 'missing'

    def test_dict_value_uses_first_key(self):
        match = re.match(r'{(\w+)}', '{config}')
        value_dict = {'config': {'option1': 'value1', 'option2': 'value2'}}
        result = custom_placeholder_format(value_dict, match)
        assert result in ['value1', 'value2']  # Dict order may vary

    def test_dict_value_with_list(self):
        match = re.match(r'{(\w+)}', '{config}')
        value_dict = {'config': {'option1': ['first', 'second']}}
        result = custom_placeholder_format(value_dict, match)
        assert result == 'first'

    def test_truncates_long_values(self):
        match = re.match(r'{(\w+)}', '{long}')
        long_value = 'a' * 100
        result = custom_placeholder_format({'long': long_value}, match)
        assert len(result) == 50
        assert result == 'a' * 50

    def test_case_insensitive_key_lookup(self):
        match = re.match(r'{(\w+)}', '{NAME}')
        result = custom_placeholder_format({'name': 'value'}, match)
        assert result == 'value'

    def test_integer_value(self):
        match = re.match(r'{(\w+)}', '{count}')
        result = custom_placeholder_format({'count': 42}, match)
        assert result == '42'


class TestCleanGradioPathStrings:
    """Test clean_gradio_path_strings function."""

    def test_removes_surrounding_quotes(self):
        result = clean_gradio_path_strings('"/path/to/file"')
        assert result == '/path/to/file'

    def test_preserves_unquoted_string(self):
        result = clean_gradio_path_strings('regular string')
        assert result == 'regular string'

    def test_preserves_partial_quotes(self):
        result = clean_gradio_path_strings('"starts with quote')
        assert result == '"starts with quote'

        result = clean_gradio_path_strings('ends with quote"')
        assert result == 'ends with quote"'

    def test_preserves_empty_string(self):
        result = clean_gradio_path_strings('')
        assert result == ''

    def test_removes_quotes_from_empty_quoted_string(self):
        result = clean_gradio_path_strings('""')
        assert result == ''

    def test_preserves_non_string_types(self):
        assert clean_gradio_path_strings(42) == 42
        assert clean_gradio_path_strings(None) is None
        assert clean_gradio_path_strings([1, 2, 3]) == [1, 2, 3]

    def test_handles_nested_quotes(self):
        result = clean_gradio_path_strings('"path "with" quotes"')
        assert result == 'path "with" quotes'


class TestTickOrCross:
    """Test tick_or_cross function."""

    def test_true_returns_tick(self):
        result = tick_or_cross(True)
        assert result == '✔'

    def test_false_returns_cross(self):
        result = tick_or_cross(False)
        assert result == '✖'

    def test_true_with_emoji(self):
        result = tick_or_cross(True, use_simple_symbols=False)
        assert result == '\U00002705'

    def test_false_with_emoji(self):
        result = tick_or_cross(False, use_simple_symbols=False)
        assert result == '\U0000274C'

    def test_default_uses_simple_symbols(self):
        result_true = tick_or_cross(True)
        result_false = tick_or_cross(False)
        assert result_true == '✔'
        assert result_false == '✖'


class TestSanitizeKeyframeValue:
    """Test sanitize_keyframe_value function."""

    def test_removes_single_quotes(self):
        result = sanitize_keyframe_value("'test'")
        assert result == 'test'

    def test_removes_double_quotes(self):
        result = sanitize_keyframe_value('"test"')
        assert result == 'test'

    def test_removes_parentheses(self):
        result = sanitize_keyframe_value('(test)')
        assert result == 'test'

    def test_removes_all_special_chars(self):
        result = sanitize_keyframe_value("'(test)'")
        assert result == 'test'

        result = sanitize_keyframe_value('"(value)"')
        assert result == 'value'

    def test_removes_nested_special_chars(self):
        result = sanitize_keyframe_value("'(\"test\")'")
        assert result == 'test'

    def test_preserves_other_characters(self):
        result = sanitize_keyframe_value("'test-value_123'")
        assert result == 'test-value_123'

    def test_preserves_numbers(self):
        result = sanitize_keyframe_value("'(3.14)'")
        assert result == '3.14'

        result = sanitize_keyframe_value('"(-42)"')
        assert result == '-42'

    def test_handles_empty_string(self):
        result = sanitize_keyframe_value('')
        assert result == ''

    def test_handles_whitespace(self):
        result = sanitize_keyframe_value("'( test )'")
        assert result == ' test '


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_os_and_tick_cross(self):
        """Test that OS detection and tick/cross work together."""
        os_name = get_os()
        is_known_os = os_name in ["Windows", "Linux", "Mac"]
        symbol = tick_or_cross(is_known_os)
        assert symbol in ['✔', '✖']

    def test_sanitize_and_clean_paths(self):
        """Test sanitization followed by path cleaning."""
        # First sanitize quotes from keyframe value
        value = sanitize_keyframe_value("'/path/to/file'")
        # Then clean Gradio path wrapping (simulated)
        value_wrapped = f'"{value}"'
        final = clean_gradio_path_strings(value_wrapped)
        assert final == '/path/to/file'

    def test_placeholder_with_sanitized_value(self):
        """Test placeholder formatting with sanitized input."""
        sanitized = sanitize_keyframe_value("'(test-value)'")
        match = re.match(r'{(\w+)}', '{key}')
        result = custom_placeholder_format({'key': sanitized}, match)
        assert result == 'test-value'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
