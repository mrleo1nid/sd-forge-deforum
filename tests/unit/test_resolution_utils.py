"""Unit tests for resolution calculation utilities."""

import pytest

from deforum.utils.resolution_utils import (
    calculate_upscaled_resolution,
    get_scale_factor_for_model,
    calculate_upscaled_resolution_by_model,
    parse_resolution_string,
    format_resolution_string,
    calculate_aspect_ratio,
)


class TestCalculateUpscaledResolution:
    """Test calculate_upscaled_resolution function."""

    def test_standard_hd_2x(self):
        """HD resolution scaled by 2."""
        assert calculate_upscaled_resolution("1920*1080", 2) == "3840*2160"

    def test_standard_hd_4x(self):
        """HD resolution scaled by 4."""
        assert calculate_upscaled_resolution("1920*1080", 4) == "7680*4320"

    def test_sd_resolution_2x(self):
        """SD resolution scaled by 2."""
        assert calculate_upscaled_resolution("640*480", 2) == "1280*960"

    def test_sd_resolution_4x(self):
        """SD resolution scaled by 4."""
        assert calculate_upscaled_resolution("640*480", 4) == "2560*1920"

    def test_square_resolution(self):
        """Square resolution scaling."""
        assert calculate_upscaled_resolution("1024*1024", 2) == "2048*2048"

    def test_scale_by_3(self):
        """Scaling by factor of 3."""
        assert calculate_upscaled_resolution("640*480", 3) == "1920*1440"

    def test_placeholder_resolution(self):
        """Placeholder '---' should pass through unchanged."""
        assert calculate_upscaled_resolution("---", 2) == "---"

    def test_empty_string(self):
        """Empty string should return '---'."""
        assert calculate_upscaled_resolution("", 2) == "---"

    def test_none_input(self):
        """None input should return '---'."""
        assert calculate_upscaled_resolution(None, 2) == "---"

    def test_zero_scale_factor_raises_error(self):
        """Zero scale factor should raise ValueError."""
        with pytest.raises(ValueError, match="scale_factor must be positive integer"):
            calculate_upscaled_resolution("1920*1080", 0)

    def test_negative_scale_factor_raises_error(self):
        """Negative scale factor should raise ValueError."""
        with pytest.raises(ValueError, match="scale_factor must be positive integer"):
            calculate_upscaled_resolution("1920*1080", -2)

    def test_float_scale_factor_raises_error(self):
        """Float scale factor should raise ValueError."""
        with pytest.raises(ValueError, match="scale_factor must be positive integer"):
            calculate_upscaled_resolution("1920*1080", 2.5)

    def test_invalid_format_single_number(self):
        """Single number without delimiter should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid input_resolution format"):
            calculate_upscaled_resolution("1920", 2)

    def test_invalid_format_wrong_delimiter(self):
        """Wrong delimiter should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid input_resolution format"):
            calculate_upscaled_resolution("1920x1080", 2)

    def test_invalid_format_non_numeric(self):
        """Non-numeric values should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid input_resolution format"):
            calculate_upscaled_resolution("abc*def", 2)


class TestGetScaleFactorForModel:
    """Test get_scale_factor_for_model function."""

    def test_anime_model_returns_2(self):
        """realesr-animevideov3 should return scale factor of 2."""
        assert get_scale_factor_for_model("realesr-animevideov3") == 2

    def test_x4plus_model_returns_4(self):
        """realesrgan-x4plus should return scale factor of 4."""
        assert get_scale_factor_for_model("realesrgan-x4plus") == 4

    def test_unknown_model_defaults_to_4(self):
        """Unknown models should default to scale factor of 4."""
        assert get_scale_factor_for_model("unknown-model") == 4

    def test_empty_string_returns_4(self):
        """Empty string should return default scale factor of 4."""
        assert get_scale_factor_for_model("") == 4

    def test_none_returns_4(self):
        """None should return default scale factor of 4."""
        assert get_scale_factor_for_model(None) == 4

    def test_case_sensitive(self):
        """Model name matching should be case-sensitive."""
        # Different case should not match
        assert get_scale_factor_for_model("RealESR-AnimeVideoV3") == 4
        assert get_scale_factor_for_model("REALESR-ANIMEVIDEOV3") == 4


class TestCalculateUpscaledResolutionByModel:
    """Test calculate_upscaled_resolution_by_model function."""

    def test_anime_model_2x_scaling(self):
        """Anime model should apply 2x scaling."""
        result = calculate_upscaled_resolution_by_model("1920*1080", "realesr-animevideov3")
        assert result == "3840*2160"

    def test_x4plus_model_4x_scaling(self):
        """X4plus model should apply 4x scaling."""
        result = calculate_upscaled_resolution_by_model("640*480", "realesrgan-x4plus")
        assert result == "2560*1920"

    def test_unknown_model_4x_default(self):
        """Unknown model should default to 4x scaling."""
        result = calculate_upscaled_resolution_by_model("512*512", "unknown")
        assert result == "2048*2048"

    def test_placeholder_with_valid_model(self):
        """Placeholder '---' should pass through even with valid model."""
        result = calculate_upscaled_resolution_by_model("---", "realesr-animevideov3")
        assert result == "---"

    def test_empty_model_name(self):
        """Empty model name should return '---'."""
        result = calculate_upscaled_resolution_by_model("1920*1080", "")
        assert result == "---"

    def test_none_model_name(self):
        """None model name should return '---'."""
        result = calculate_upscaled_resolution_by_model("1920*1080", None)
        assert result == "---"


class TestParseResolutionString:
    """Test parse_resolution_string function."""

    def test_hd_resolution(self):
        """Parse HD resolution."""
        assert parse_resolution_string("1920*1080") == (1920, 1080)

    def test_sd_resolution(self):
        """Parse SD resolution."""
        assert parse_resolution_string("640*480") == (640, 480)

    def test_square_resolution(self):
        """Parse square resolution."""
        assert parse_resolution_string("1024*1024") == (1024, 1024)

    def test_4k_resolution(self):
        """Parse 4K resolution."""
        assert parse_resolution_string("3840*2160") == (3840, 2160)

    def test_invalid_format_single_number(self):
        """Single number should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution_string("1920")

    def test_invalid_format_wrong_delimiter(self):
        """Wrong delimiter should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution_string("1920x1080")

    def test_invalid_format_non_numeric(self):
        """Non-numeric values should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution_string("abc*def")

    def test_none_input(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution_string(None)

    def test_empty_string(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            parse_resolution_string("")


class TestFormatResolutionString:
    """Test format_resolution_string function."""

    def test_hd_resolution(self):
        """Format HD resolution."""
        assert format_resolution_string(1920, 1080) == "1920*1080"

    def test_sd_resolution(self):
        """Format SD resolution."""
        assert format_resolution_string(640, 480) == "640*480"

    def test_square_resolution(self):
        """Format square resolution."""
        assert format_resolution_string(1024, 1024) == "1024*1024"

    def test_4k_resolution(self):
        """Format 4K resolution."""
        assert format_resolution_string(3840, 2160) == "3840*2160"

    def test_zero_width_raises_error(self):
        """Zero width should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            format_resolution_string(0, 1080)

    def test_zero_height_raises_error(self):
        """Zero height should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            format_resolution_string(1920, 0)

    def test_negative_width_raises_error(self):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            format_resolution_string(-1920, 1080)

    def test_negative_height_raises_error(self):
        """Negative height should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            format_resolution_string(1920, -1080)

    def test_float_width_raises_error(self):
        """Float width should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be integers"):
            format_resolution_string(1920.5, 1080)

    def test_float_height_raises_error(self):
        """Float height should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be integers"):
            format_resolution_string(1920, 1080.5)


class TestCalculateAspectRatio:
    """Test calculate_aspect_ratio function."""

    def test_hd_16_9_ratio(self):
        """HD resolution should have 16:9 aspect ratio."""
        ratio = calculate_aspect_ratio(1920, 1080)
        assert pytest.approx(ratio, abs=0.01) == 16/9

    def test_sd_4_3_ratio(self):
        """SD resolution should have 4:3 aspect ratio."""
        ratio = calculate_aspect_ratio(640, 480)
        assert pytest.approx(ratio, abs=0.01) == 4/3

    def test_square_1_1_ratio(self):
        """Square resolution should have 1:1 aspect ratio."""
        ratio = calculate_aspect_ratio(1024, 1024)
        assert ratio == 1.0

    def test_ultrawide_21_9_ratio(self):
        """Ultrawide resolution should have approximately 21:9 aspect ratio."""
        ratio = calculate_aspect_ratio(2560, 1080)
        # 2560/1080 = 2.370 which is approximately 21/9 = 2.333
        # But they're not exactly equal, so we verify it's close
        assert 2.3 < ratio < 2.4
        assert pytest.approx(ratio, rel=0.02) == 21/9

    def test_portrait_orientation(self):
        """Portrait orientation should have ratio < 1."""
        ratio = calculate_aspect_ratio(1080, 1920)
        assert ratio < 1.0
        assert pytest.approx(ratio, abs=0.01) == 9/16

    def test_zero_height_raises_error(self):
        """Zero height should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            calculate_aspect_ratio(1920, 0)

    def test_zero_width_raises_error(self):
        """Zero width should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            calculate_aspect_ratio(0, 1080)

    def test_negative_dimensions_raise_error(self):
        """Negative dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be positive"):
            calculate_aspect_ratio(-1920, 1080)

    def test_float_dimensions_raise_error(self):
        """Float dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="width and height must be integers"):
            calculate_aspect_ratio(1920.5, 1080)


class TestResolutionUtilsIntegration:
    """Integration tests for resolution utility functions."""

    def test_parse_and_format_roundtrip(self):
        """Parsing and formatting should be inverse operations."""
        original = "1920*1080"
        w, h = parse_resolution_string(original)
        result = format_resolution_string(w, h)
        assert result == original

    def test_upscale_workflow(self):
        """Simulate full upscaling workflow."""
        input_res = "640*480"
        model = "realesr-animevideov3"  # 2x model

        # Calculate upscaled resolution
        output_res = calculate_upscaled_resolution_by_model(input_res, model)
        assert output_res == "1280*960"

        # Parse to verify dimensions
        w, h = parse_resolution_string(output_res)
        assert w == 1280
        assert h == 960

        # Verify aspect ratio preserved
        input_w, input_h = parse_resolution_string(input_res)
        input_ratio = calculate_aspect_ratio(input_w, input_h)
        output_ratio = calculate_aspect_ratio(w, h)
        assert pytest.approx(input_ratio, abs=0.01) == output_ratio

    def test_multiple_upscaling_steps(self):
        """Apply upscaling multiple times."""
        resolution = "512*512"

        # First upscale 2x
        resolution = calculate_upscaled_resolution(resolution, 2)
        assert resolution == "1024*1024"

        # Second upscale 2x
        resolution = calculate_upscaled_resolution(resolution, 2)
        assert resolution == "2048*2048"

        # Verify final dimensions
        w, h = parse_resolution_string(resolution)
        assert w == 2048
        assert h == 2048

    def test_model_based_upscaling_chain(self):
        """Chain different model-based upscaling."""
        resolution = "640*480"

        # Upscale with anime model (2x)
        resolution = calculate_upscaled_resolution_by_model(resolution, "realesr-animevideov3")
        assert resolution == "1280*960"

        # Upscale again with 4x model
        resolution = calculate_upscaled_resolution_by_model(resolution, "realesrgan-x4plus")
        assert resolution == "5120*3840"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
