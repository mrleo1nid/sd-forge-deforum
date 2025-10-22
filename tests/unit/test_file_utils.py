"""Unit tests for file and path utilities."""

import pytest
import os
import tempfile

from deforum.utils.file_utils import (
    get_max_path_length,
    count_files_in_folder,
)


class TestGetMaxPathLength:
    """Test get_max_path_length function."""

    def test_windows_without_long_path_support(self):
        base = "C:\\Users\\user"
        result = get_max_path_length(base, "Windows", False)
        expected = 260 - len(base) - 1
        assert result == expected

    def test_windows_with_long_path_support(self):
        base = "C:\\Users\\user"
        result = get_max_path_length(base, "Windows", True)
        expected = 32767 - len(base) - 1
        assert result == expected

    def test_linux_path_length(self):
        base = "/home/user"
        result = get_max_path_length(base, "Linux", False)
        expected = 4096 - len(base) - 1
        assert result == expected

    def test_mac_path_length(self):
        base = "/Users/user"
        result = get_max_path_length(base, "Mac", False)
        expected = 4096 - len(base) - 1
        assert result == expected

    def test_longer_base_path_reduces_max(self):
        short_base = "/home/user"
        long_base = "/home/user/documents/projects/deforum/output"

        result_short = get_max_path_length(short_base, "Linux", False)
        result_long = get_max_path_length(long_base, "Linux", False)

        assert result_long < result_short
        difference = result_short - result_long
        assert difference == len(long_base) - len(short_base)

    def test_empty_base_path(self):
        result = get_max_path_length("", "Linux", False)
        expected = 4096 - 1
        assert result == expected

    def test_windows_long_path_significantly_longer(self):
        base = "C:\\test"
        without_support = get_max_path_length(base, "Windows", False)
        with_support = get_max_path_length(base, "Windows", True)
        assert with_support > without_support
        assert with_support > 30000  # Should be ~32760

    def test_non_windows_ignores_long_path_flag(self):
        base = "/home/test"
        result_false = get_max_path_length(base, "Linux", False)
        result_true = get_max_path_length(base, "Linux", True)
        assert result_false == result_true


class TestCountFilesInFolder:
    """Test count_files_in_folder function."""

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            count = count_files_in_folder(tmpdir)
            assert count == 0

    def test_folder_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 files
            for i in range(3):
                filepath = os.path.join(tmpdir, f"file{i}.txt")
                open(filepath, 'w').close()

            count = count_files_in_folder(tmpdir)
            assert count == 3

    def test_folder_with_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 files
            for i in range(2):
                filepath = os.path.join(tmpdir, f"file{i}.txt")
                open(filepath, 'w').close()

            # Create 2 subdirectories
            for i in range(2):
                subdir = os.path.join(tmpdir, f"subdir{i}")
                os.makedirs(subdir)

            # Count should include both files and directories
            count = count_files_in_folder(tmpdir)
            assert count == 4  # 2 files + 2 dirs

    def test_folder_with_different_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            extensions = ['.txt', '.py', '.json', '.png']
            for ext in extensions:
                filepath = os.path.join(tmpdir, f"file{ext}")
                open(filepath, 'w').close()

            count = count_files_in_folder(tmpdir)
            assert count == len(extensions)

    def test_folder_with_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create regular file
            regular = os.path.join(tmpdir, "regular.txt")
            open(regular, 'w').close()

            # Create hidden file (Unix-style)
            hidden = os.path.join(tmpdir, ".hidden")
            open(hidden, 'w').close()

            count = count_files_in_folder(tmpdir)
            # Note: glob.glob("*") doesn't match hidden files (starting with .) by default
            assert count == 1  # Only regular file is counted

    def test_non_recursive_count(self):
        """Verify count is not recursive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in root
            root_file = os.path.join(tmpdir, "root.txt")
            open(root_file, 'w').close()

            # Create subdirectory with files
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            sub_file = os.path.join(subdir, "sub.txt")
            open(sub_file, 'w').close()

            # Should only count root file + subdir (not sub_file)
            count = count_files_in_folder(tmpdir)
            assert count == 2


class TestIntegration:
    """Integration tests combining file utilities."""

    def test_path_length_calculation_realistic(self):
        """Test realistic path length calculations."""
        # Simulate Windows path
        base = "C:\\Users\\username\\Documents\\Projects\\Deforum\\outputs"
        max_len = get_max_path_length(base, "Windows", False)

        # Should have reasonable space left
        assert max_len > 100
        assert max_len < 260

        # Check math
        expected = 260 - len(base) - 1
        assert max_len == expected

    def test_path_length_with_very_long_base(self):
        """Test with very long base path."""
        # Create artificially long path
        long_segments = ["very_long_folder_name_" + str(i) for i in range(10)]
        base = "/".join(long_segments)

        max_len = get_max_path_length(base, "Linux", False)

        # Should still be positive
        assert max_len > 0
        assert max_len == 4096 - len(base) - 1

    def test_count_after_file_operations(self):
        """Test counting after creating and deleting files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Start empty
            assert count_files_in_folder(tmpdir) == 0

            # Add 3 files
            for i in range(3):
                filepath = os.path.join(tmpdir, f"file{i}.txt")
                open(filepath, 'w').close()
            assert count_files_in_folder(tmpdir) == 3

            # Remove 1 file
            os.remove(os.path.join(tmpdir, "file0.txt"))
            assert count_files_in_folder(tmpdir) == 2

            # Add 2 more
            for i in range(3, 5):
                filepath = os.path.join(tmpdir, f"file{i}.txt")
                open(filepath, 'w').close()
            assert count_files_in_folder(tmpdir) == 4

    def test_os_specific_path_limits(self):
        """Test that different OSes have appropriate limits."""
        base = "/test/path"

        windows_short = get_max_path_length(base, "Windows", False)
        windows_long = get_max_path_length(base, "Windows", True)
        linux_len = get_max_path_length(base, "Linux", False)

        # Windows short path should be smallest
        assert windows_short < linux_len
        assert windows_short < windows_long

        # Linux should be between Windows short and long
        assert linux_len > windows_short
        assert linux_len < windows_long

        # Windows long should be largest
        assert windows_long > linux_len
        assert windows_long > windows_short


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
