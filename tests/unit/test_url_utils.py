"""Unit tests for deforum.utils.url_utils module."""

import pytest

from deforum.utils.url_utils import (
    extract_extension_from_url,
    extract_extension_from_path,
    parse_content_disposition_filename,
    extract_extension_from_content_disposition,
    is_url,
    validate_extension,
    get_file_extension_or_default,
)


class TestExtractExtensionFromUrl:
    """Tests for extract_extension_from_url function."""

    def test_simple_url(self):
        """Should extract extension from simple URL."""
        assert extract_extension_from_url('http://example.com/file.mp4') == 'mp4'
        assert extract_extension_from_url('https://example.com/video.avi') == 'avi'

    def test_url_with_query_params(self):
        """Should remove query parameters before extracting extension."""
        assert extract_extension_from_url('https://example.com/video.mp4?token=abc123') == 'mp4'
        assert extract_extension_from_url('http://example.com/file.txt?key=value&foo=bar') == 'txt'

    def test_uppercase_extension(self):
        """Should convert extension to lowercase."""
        assert extract_extension_from_url('http://example.com/VIDEO.MP4') == 'mp4'
        assert extract_extension_from_url('https://example.com/FILE.PNG') == 'png'

    def test_url_without_extension(self):
        """Should return empty string if no extension."""
        assert extract_extension_from_url('http://example.com/file') == ''
        assert extract_extension_from_url('https://example.com/path/') == ''

    def test_complex_path(self):
        """Should handle complex paths."""
        assert extract_extension_from_url('https://example.com/path/to/deep/file.mov') == 'mov'
        assert extract_extension_from_url('http://cdn.example.com/assets/video.webm?v=2') == 'webm'

    def test_multiple_dots(self):
        """Should extract last extension for multi-dot filenames."""
        assert extract_extension_from_url('http://example.com/archive.tar.gz') == 'gz'
        assert extract_extension_from_url('https://example.com/file.backup.mp4') == 'mp4'


class TestExtractExtensionFromPath:
    """Tests for extract_extension_from_path function."""

    def test_simple_filename(self):
        """Should extract extension from simple filename."""
        assert extract_extension_from_path('video.mp4') == 'mp4'
        assert extract_extension_from_path('image.png') == 'png'

    def test_absolute_path(self):
        """Should extract extension from absolute path."""
        assert extract_extension_from_path('/path/to/file.mp4') == 'mp4'
        assert extract_extension_from_path('/home/user/video.avi') == 'avi'

    def test_relative_path(self):
        """Should extract extension from relative path."""
        assert extract_extension_from_path('./video.mp4') == 'mp4'
        assert extract_extension_from_path('../file.txt') == 'txt'

    def test_uppercase_extension(self):
        """Should convert to lowercase."""
        assert extract_extension_from_path('VIDEO.MP4') == 'mp4'
        assert extract_extension_from_path('/path/FILE.AVI') == 'avi'

    def test_no_extension(self):
        """Should return empty string if no extension."""
        assert extract_extension_from_path('file') == ''
        assert extract_extension_from_path('/path/to/noext') == ''

    def test_hidden_file(self):
        """Should handle hidden files."""
        assert extract_extension_from_path('.hidden.txt') == 'txt'
        assert extract_extension_from_path('/path/.config') == 'config'

    def test_multiple_dots(self):
        """Should extract last extension."""
        assert extract_extension_from_path('archive.tar.gz') == 'gz'
        assert extract_extension_from_path('file.backup.mp4') == 'mp4'


class TestParseContentDispositionFilename:
    """Tests for parse_content_disposition_filename function."""

    def test_quoted_filename(self):
        """Should parse quoted filename."""
        result = parse_content_disposition_filename('attachment; filename="video.mp4"')
        assert result == 'video.mp4'

    def test_unquoted_filename(self):
        """Should parse unquoted filename."""
        result = parse_content_disposition_filename('attachment; filename=document.pdf')
        assert result == 'document.pdf'

    def test_inline_disposition(self):
        """Should work with inline disposition."""
        result = parse_content_disposition_filename('inline; filename="image.png"')
        assert result == 'image.png'

    def test_filename_with_spaces(self):
        """Should handle filenames with spaces."""
        result = parse_content_disposition_filename('attachment; filename="my file.txt"')
        assert result == 'my file.txt'

    def test_no_filename(self):
        """Should return None if no filename."""
        assert parse_content_disposition_filename('attachment') is None
        assert parse_content_disposition_filename('inline') is None

    def test_empty_string(self):
        """Should return None for empty string."""
        assert parse_content_disposition_filename('') is None

    def test_complex_header(self):
        """Should handle complex headers."""
        result = parse_content_disposition_filename(
            'attachment; filename="report.pdf"; creation-date="Mon, 01 Jan 2024 00:00:00 GMT"'
        )
        assert result == 'report.pdf'


class TestExtractExtensionFromContentDisposition:
    """Tests for extract_extension_from_content_disposition function."""

    def test_extract_from_quoted_filename(self):
        """Should extract extension from quoted filename."""
        ext = extract_extension_from_content_disposition('attachment; filename="video.mp4"')
        assert ext == 'mp4'

    def test_extract_from_unquoted_filename(self):
        """Should extract extension from unquoted filename."""
        ext = extract_extension_from_content_disposition('attachment; filename=document.PDF')
        assert ext == 'pdf'

    def test_lowercase_conversion(self):
        """Should convert to lowercase."""
        ext = extract_extension_from_content_disposition('attachment; filename="FILE.AVI"')
        assert ext == 'avi'

    def test_no_filename(self):
        """Should return empty string if no filename."""
        assert extract_extension_from_content_disposition('attachment') == ''

    def test_empty_string(self):
        """Should return empty string for empty input."""
        assert extract_extension_from_content_disposition('') == ''

    def test_filename_without_extension(self):
        """Should return empty string if filename has no extension."""
        assert extract_extension_from_content_disposition('attachment; filename="noext"') == ''

    def test_multi_dot_filename(self):
        """Should extract last extension."""
        ext = extract_extension_from_content_disposition('attachment; filename="archive.tar.gz"')
        assert ext == 'gz'


class TestIsUrl:
    """Tests for is_url function."""

    def test_http_url(self):
        """Should recognize http URLs."""
        assert is_url('http://example.com/file.mp4')
        assert is_url('http://localhost/video')

    def test_https_url(self):
        """Should recognize https URLs."""
        assert is_url('https://example.com/file.mp4')
        assert is_url('https://secure.site.com/path')

    def test_local_path(self):
        """Should reject local paths."""
        assert not is_url('/path/to/file.mp4')
        assert not is_url('./relative/path')
        assert not is_url('../parent/dir')

    def test_filename(self):
        """Should reject plain filenames."""
        assert not is_url('file.mp4')
        assert not is_url('video.avi')

    def test_other_protocols(self):
        """Should reject other protocols."""
        assert not is_url('ftp://example.com/file')
        assert not is_url('file:///path/to/file')
        assert not is_url('ssh://server/path')

    def test_edge_cases(self):
        """Should handle edge cases."""
        assert not is_url('')
        assert not is_url('http')
        assert not is_url('https')


class TestValidateExtension:
    """Tests for validate_extension function."""

    def test_valid_extension(self):
        """Should validate matching extension."""
        is_valid, ext = validate_extension('mp4', ['mp4', 'avi', 'mov'])
        assert is_valid
        assert ext == 'mp4'

    def test_invalid_extension(self):
        """Should reject non-matching extension."""
        is_valid, ext = validate_extension('txt', ['mp4', 'avi'])
        assert not is_valid
        assert ext == 'txt'

    def test_case_insensitive(self):
        """Should handle uppercase input."""
        is_valid, ext = validate_extension('MP4', ['mp4', 'avi'])
        assert is_valid
        assert ext == 'mp4'

    def test_empty_extension(self):
        """Should reject empty extension."""
        is_valid, ext = validate_extension('', ['mp4'])
        assert not is_valid
        assert ext == ''

    def test_empty_acceptable_list(self):
        """Should reject when no extensions acceptable."""
        is_valid, ext = validate_extension('mp4', [])
        assert not is_valid

    def test_multiple_acceptable(self):
        """Should validate against multiple extensions."""
        assert validate_extension('mp4', ['mp4', 'avi', 'mov', 'webm'])[0]
        assert validate_extension('avi', ['mp4', 'avi', 'mov', 'webm'])[0]
        assert not validate_extension('txt', ['mp4', 'avi', 'mov', 'webm'])[0]


class TestGetFileExtensionOrDefault:
    """Tests for get_file_extension_or_default function."""

    def test_url_with_extension(self):
        """Should extract extension from URL."""
        ext = get_file_extension_or_default('http://example.com/file.mp4', 'mp3')
        assert ext == 'mp4'

    def test_url_without_extension(self):
        """Should return default for URL without extension."""
        ext = get_file_extension_or_default('http://example.com/file', 'mp3')
        assert ext == 'mp3'

    def test_path_with_extension(self):
        """Should extract extension from path."""
        ext = get_file_extension_or_default('/path/to/video.avi', 'mp3')
        assert ext == 'avi'

    def test_path_without_extension(self):
        """Should return default for path without extension."""
        ext = get_file_extension_or_default('noextension', 'wav')
        assert ext == 'wav'

    def test_default_empty_string(self):
        """Should handle empty default."""
        ext = get_file_extension_or_default('noext', '')
        assert ext == ''

    def test_url_with_query_params(self):
        """Should handle URLs with query parameters."""
        ext = get_file_extension_or_default('https://example.com/video.mp4?token=abc', 'mp3')
        assert ext == 'mp4'

    def test_uppercase_conversion(self):
        """Should convert to lowercase."""
        ext = get_file_extension_or_default('FILE.MP4', 'mp3')
        assert ext == 'mp4'


class TestIntegration:
    """Integration tests for URL utilities."""

    def test_url_validation_pipeline(self):
        """Test complete URL validation pipeline."""
        url = 'https://example.com/video.mp4?token=abc123'

        # Check if it's a URL
        assert is_url(url)

        # Extract extension
        ext = extract_extension_from_url(url)
        assert ext == 'mp4'

        # Validate extension
        is_valid, ext_lower = validate_extension(ext, ['mp4', 'avi', 'mov'])
        assert is_valid
        assert ext_lower == 'mp4'

    def test_content_disposition_pipeline(self):
        """Test Content-Disposition header parsing pipeline."""
        header = 'attachment; filename="movie.MP4"'

        # Parse filename
        filename = parse_content_disposition_filename(header)
        assert filename == 'movie.MP4'

        # Extract extension
        ext = extract_extension_from_content_disposition(header)
        assert ext == 'mp4'

        # Validate
        is_valid, _ = validate_extension(ext, ['mp4', 'avi'])
        assert is_valid

    def test_fallback_logic(self):
        """Test fallback to default extension."""
        # URL without extension
        url_no_ext = 'http://example.com/download'
        ext = get_file_extension_or_default(url_no_ext, 'mp3')
        assert ext == 'mp3'

        # URL with extension
        url_with_ext = 'http://example.com/audio.wav'
        ext = get_file_extension_or_default(url_with_ext, 'mp3')
        assert ext == 'wav'

    def test_real_world_scenarios(self):
        """Test real-world use cases."""
        # S3 presigned URL
        s3_url = 'https://bucket.s3.amazonaws.com/file.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256'
        ext = extract_extension_from_url(s3_url)
        assert ext == 'mp4'

        # CDN URL
        cdn_url = 'https://cdn.example.com/path/to/video.webm?v=1.2.3'
        ext = extract_extension_from_url(cdn_url)
        assert ext == 'webm'

        # Local file path
        local_path = '/home/user/Videos/vacation.MOV'
        ext = extract_extension_from_path(local_path)
        assert ext == 'mov'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
