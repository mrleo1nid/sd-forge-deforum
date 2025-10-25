"""Unit tests for deforum.utils.hash_utils module."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from deforum.utils.conversion.hashing import (
    compute_file_checksum,
    compute_file_checksum_with_factory,
    compute_string_hash,
    verify_file_checksum,
    get_supported_hash_algorithms,
)


class TestComputeFileChecksum:
    """Tests for compute_file_checksum function."""

    def test_compute_sha256_checksum(self, tmp_path):
        """Should compute SHA256 checksum of file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Compute checksum
        result = compute_file_checksum(test_file, 'sha256')

        # Verify (known SHA256 of "hello world")
        expected = 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        assert result == expected

    def test_compute_md5_checksum(self, tmp_path):
        """Should compute MD5 checksum of file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = compute_file_checksum(test_file, 'md5')

        # Known MD5 of "test"
        expected = '098f6bcd4621d373cade4e832627b4f6'
        assert result == expected

    def test_compute_blake2b_checksum(self, tmp_path):
        """Should compute BLAKE2b checksum of file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        result = compute_file_checksum(test_file, 'blake2b')

        # Verify it's a valid blake2b hash (128 hex characters)
        assert len(result) == 128
        assert all(c in '0123456789abcdef' for c in result)

    def test_empty_file(self, tmp_path):
        """Should handle empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = compute_file_checksum(test_file, 'sha256')

        # Known SHA256 of empty string
        expected = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        assert result == expected

    def test_binary_file(self, tmp_path):
        """Should handle binary file."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b'\x00\x01\x02\x03\xff')

        result = compute_file_checksum(test_file, 'sha256')

        # Verify it computed something
        assert len(result) == 64  # SHA256 is 64 hex chars

    def test_large_file(self, tmp_path):
        """Should handle large file with chunked reading."""
        test_file = tmp_path / "large.txt"
        # Write 1MB of data
        test_file.write_bytes(b'a' * (1024 * 1024))

        result = compute_file_checksum(test_file, 'sha256', chunk_size=4096)

        # Verify it computed correctly
        assert len(result) == 64

    def test_file_not_found_raises_error(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_checksum('/nonexistent/file.txt')

    def test_directory_raises_error(self, tmp_path):
        """Should raise ValueError for directory."""
        with pytest.raises(ValueError, match="not a file"):
            compute_file_checksum(tmp_path)

    def test_unsupported_algorithm_raises_error(self, tmp_path):
        """Should raise ValueError for unsupported algorithm."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_file_checksum(test_file, 'nonexistent_algorithm')

    def test_string_path(self, tmp_path):
        """Should handle string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        result = compute_file_checksum(str(test_file), 'sha256')
        assert len(result) == 64


class TestComputeFileChecksumWithFactory:
    """Tests for compute_file_checksum_with_factory function."""

    def test_with_md5_factory(self, tmp_path):
        """Should compute checksum with MD5 factory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = compute_file_checksum_with_factory(test_file, hashlib.md5)

        expected = '098f6bcd4621d373cade4e832627b4f6'
        assert result == expected

    def test_with_sha256_factory(self, tmp_path):
        """Should compute checksum with SHA256 factory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = compute_file_checksum_with_factory(test_file, hashlib.sha256)

        expected = 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        assert result == expected

    def test_with_blake2b_factory(self, tmp_path):
        """Should compute checksum with BLAKE2b factory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = compute_file_checksum_with_factory(test_file, hashlib.blake2b)

        # Verify it's a valid blake2b hash
        assert len(result) == 128

    def test_custom_chunk_blocks(self, tmp_path):
        """Should use custom chunk block size."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b'a' * 10000)

        result = compute_file_checksum_with_factory(
            test_file, hashlib.sha256, chunk_num_blocks=64
        )

        assert len(result) == 64

    def test_file_not_found_raises_error(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_checksum_with_factory('/nonexistent/file.txt', hashlib.md5)


class TestComputeStringHash:
    """Tests for compute_string_hash function."""

    def test_hash_string_sha256(self):
        """Should hash string with SHA256."""
        result = compute_string_hash("hello world", 'sha256')

        expected = 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        assert result == expected

    def test_hash_string_md5(self):
        """Should hash string with MD5."""
        result = compute_string_hash("test", 'md5')

        expected = '098f6bcd4621d373cade4e832627b4f6'
        assert result == expected

    def test_hash_bytes(self):
        """Should hash bytes."""
        result = compute_string_hash(b"test", 'sha256')

        # Known SHA256 of "test"
        expected = '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'
        assert result == expected

    def test_empty_string(self):
        """Should hash empty string."""
        result = compute_string_hash("", 'sha256')

        expected = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        assert result == expected

    def test_unicode_string(self):
        """Should hash unicode string."""
        result = compute_string_hash("こんにちは", 'sha256')

        # Verify it computed something (actual value depends on encoding)
        assert len(result) == 64

    def test_unsupported_algorithm_raises_error(self):
        """Should raise ValueError for unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_string_hash("test", 'invalid_algorithm')


class TestVerifyFileChecksum:
    """Tests for verify_file_checksum function."""

    def test_verify_correct_checksum(self, tmp_path):
        """Should return True for correct checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Compute expected checksum
        expected = '098f6bcd4621d373cade4e832627b4f6'

        assert verify_file_checksum(test_file, expected, 'md5')

    def test_verify_incorrect_checksum(self, tmp_path):
        """Should return False for incorrect checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        assert not verify_file_checksum(test_file, 'wrong_checksum', 'md5')

    def test_verify_case_insensitive(self, tmp_path):
        """Should be case insensitive."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Use uppercase checksum
        expected = '098F6BCD4621D373CADE4E832627B4F6'

        assert verify_file_checksum(test_file, expected, 'md5')

    def test_verify_nonexistent_file(self):
        """Should return False for nonexistent file."""
        assert not verify_file_checksum('/nonexistent/file.txt', 'abc123', 'md5')

    def test_verify_with_different_algorithms(self, tmp_path):
        """Should verify with different algorithms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        # SHA256 checksum of "hello"
        sha256_sum = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
        assert verify_file_checksum(test_file, sha256_sum, 'sha256')

        # MD5 checksum of "hello"
        md5_sum = '5d41402abc4b2a76b9719d911017c592'
        assert verify_file_checksum(test_file, md5_sum, 'md5')


class TestGetSupportedHashAlgorithms:
    """Tests for get_supported_hash_algorithms function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_supported_hash_algorithms()
        assert isinstance(result, list)

    def test_contains_common_algorithms(self):
        """Should contain common hash algorithms."""
        algorithms = get_supported_hash_algorithms()

        assert 'sha256' in algorithms
        assert 'md5' in algorithms
        assert 'sha1' in algorithms

    def test_list_is_sorted(self):
        """Should return sorted list."""
        algorithms = get_supported_hash_algorithms()
        assert algorithms == sorted(algorithms)

    def test_all_strings(self):
        """Should contain only strings."""
        algorithms = get_supported_hash_algorithms()
        assert all(isinstance(alg, str) for alg in algorithms)


class TestHashUtilsIntegration:
    """Integration tests combining multiple hash utilities."""

    def test_compute_and_verify_workflow(self, tmp_path):
        """Should compute and verify checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("important data")

        # Compute checksum
        checksum = compute_file_checksum(test_file, 'sha256')

        # Verify checksum
        assert verify_file_checksum(test_file, checksum, 'sha256')

        # Modify file
        test_file.write_text("modified data")

        # Verification should now fail
        assert not verify_file_checksum(test_file, checksum, 'sha256')

    def test_multiple_algorithms_same_file(self, tmp_path):
        """Should compute different checksums with different algorithms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        sha256 = compute_file_checksum(test_file, 'sha256')
        md5 = compute_file_checksum(test_file, 'md5')
        sha1 = compute_file_checksum(test_file, 'sha1')

        # All should be different
        assert sha256 != md5
        assert sha256 != sha1
        assert md5 != sha1

        # All should have expected lengths
        assert len(sha256) == 64
        assert len(md5) == 32
        assert len(sha1) == 40

    def test_file_vs_string_hash_equivalence(self, tmp_path):
        """Should produce same hash for file and string."""
        content = "test content"

        # Hash string
        string_hash = compute_string_hash(content, 'sha256')

        # Hash file with same content
        test_file = tmp_path / "test.txt"
        test_file.write_text(content)
        file_hash = compute_file_checksum(test_file, 'sha256')

        assert string_hash == file_hash

    def test_factory_vs_algorithm_equivalence(self, tmp_path):
        """Should produce same hash with factory and algorithm name."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        hash1 = compute_file_checksum(test_file, 'sha256')
        hash2 = compute_file_checksum_with_factory(test_file, hashlib.sha256)

        assert hash1 == hash2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
