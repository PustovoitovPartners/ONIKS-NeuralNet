"""Unit tests for the ListFilesTool class."""

import pytest
import os
import tempfile
import stat
from pathlib import Path
from unittest.mock import patch, Mock

from oniks.tools.fs_tools import ListFilesTool, WriteFileTool


class TestListFilesToolInitialization:
    """Test ListFilesTool initialization."""
    
    def test_list_files_tool_initialization(self):
        """Test ListFilesTool initialization sets correct attributes."""
        tool = ListFilesTool()
        
        assert tool.name == "list_files"
        assert "Recursively lists all files and directories" in tool.description
        assert "path" in tool.description
        assert "ignore_patterns" in tool.description
        assert isinstance(tool.description, str)
    
    def test_list_files_tool_string_representation(self):
        """Test ListFilesTool string representation."""
        tool = ListFilesTool()
        
        assert str(tool) == "ListFilesTool(name='list_files')"
        assert repr(tool) == "ListFilesTool(name='list_files')"
    
    def test_list_files_tool_default_ignore_patterns(self):
        """Test ListFilesTool has proper default ignore patterns."""
        tool = ListFilesTool()
        
        expected_patterns = [
            ".git", "__pycache__", "venv", ".venv", "env", ".env",
            "node_modules", ".pytest_cache", ".mypy_cache", ".coverage",
            "htmlcov", ".tox", "dist", "build", "*.egg-info",
            ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.pyd",
            ".idea", ".vscode", "*.log", ".logs"
        ]
        
        for pattern in expected_patterns:
            assert pattern in tool._default_ignore_patterns
        
        assert isinstance(tool._default_ignore_patterns, list)
        assert len(tool._default_ignore_patterns) > 0


class TestListFilesToolExecution:
    """Test ListFilesTool execution with various scenarios."""
    
    @pytest.fixture
    def simple_temp_dir(self):
        """Create a simple temporary directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files
            (temp_path / "file1.txt").write_text("content1")
            (temp_path / "file2.py").write_text("content2")
            
            # Create subdirectory with files
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "file3.md").write_text("content3")
            (sub_dir / "file4.log").write_text("content4")
            
            # Create nested subdirectory
            nested_dir = sub_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file5.json").write_text('{"key": "value"}')
            
            yield temp_path
    
    @pytest.fixture
    def temp_dir_with_ignored_files(self):
        """Create a temporary directory with files that should be ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create regular files
            (temp_path / "main.py").write_text("print('hello')")
            (temp_path / "README.md").write_text("# Project")
            
            # Create directories that should be ignored
            git_dir = temp_path / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("git config")
            
            pycache_dir = temp_path / "__pycache__"
            pycache_dir.mkdir()
            (pycache_dir / "module.cpython-39.pyc").write_text("bytecode")
            
            venv_dir = temp_path / "venv"
            venv_dir.mkdir()
            (venv_dir / "pyvenv.cfg").write_text("venv config")
            
            # Create files that should be ignored
            (temp_path / "file.pyc").write_text("bytecode")
            (temp_path / "debug.log").write_text("debug info")
            (temp_path / ".DS_Store").write_text("macos metadata")
            
            yield temp_path
    
    def test_list_files_tool_successful_simple_structure(self, simple_temp_dir):
        """Test successful listing of simple directory structure."""
        tool = ListFilesTool()
        
        result = tool.execute(path=str(simple_temp_dir))
        
        assert isinstance(result, str)
        assert str(simple_temp_dir) in result
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir/" in result
        assert "file3.md" in result
        assert "nested/" in result
        assert "file5.json" in result
        
        # Check tree structure symbols
        assert "├── " in result or "└── " in result
        assert "│   " in result or "    " in result
    
    def test_list_files_tool_default_ignore_patterns(self, temp_dir_with_ignored_files):
        """Test that default ignore patterns work correctly."""
        tool = ListFilesTool()
        
        result = tool.execute(path=str(temp_dir_with_ignored_files))
        
        # Should include regular files
        assert "main.py" in result
        assert "README.md" in result
        
        # Should ignore default patterns
        assert ".git" not in result
        assert "__pycache__" not in result
        assert "venv" not in result
        assert ".pyc" not in result
        assert ".log" not in result
        assert ".DS_Store" not in result
    
    def test_list_files_tool_custom_ignore_patterns(self, simple_temp_dir):
        """Test custom ignore patterns functionality."""
        tool = ListFilesTool()
        
        # Test with custom patterns
        result = tool.execute(
            path=str(simple_temp_dir),
            ignore_patterns=["*.py", "*.log", "nested"]
        )
        
        # Should include files not matching custom patterns
        assert "file1.txt" in result
        assert "file3.md" in result
        
        # Should ignore files matching custom patterns
        assert "file2.py" not in result
        assert "file4.log" not in result
        assert "nested/" not in result
        assert "file5.json" not in result  # Inside ignored nested directory
    
    def test_list_files_tool_combined_ignore_patterns(self, temp_dir_with_ignored_files):
        """Test combination of default and custom ignore patterns."""
        tool = ListFilesTool()
        
        result = tool.execute(
            path=str(temp_dir_with_ignored_files),
            ignore_patterns=["*.md", "main.*"]
        )
        
        # Should ignore both default and custom patterns
        assert ".git" not in result  # Default pattern
        assert "__pycache__" not in result  # Default pattern
        assert "README.md" not in result  # Custom pattern
        assert "main.py" not in result  # Custom pattern
    
    def test_list_files_tool_empty_directory(self):
        """Test listing empty directory.""" 
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ListFilesTool()
            
            result = tool.execute(path=temp_dir)
            
            assert isinstance(result, str)
            assert temp_dir in result
            # Should only contain the root directory
            lines = result.strip().split('\n')
            assert len(lines) == 1
            assert lines[0].endswith('/')
    
    def test_list_files_tool_single_file_directory(self):
        """Test directory with single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "single_file.txt").write_text("content")
            
            tool = ListFilesTool()
            result = tool.execute(path=temp_dir)
            
            assert "single_file.txt" in result
            assert "└── single_file.txt" in result


class TestListFilesToolArgumentValidation:
    """Test ListFilesTool argument validation."""
    
    def test_list_files_tool_missing_path_argument(self):
        """Test execution without path argument."""
        tool = ListFilesTool()
        
        result = tool.execute()
        
        assert "Error: Missing required argument 'path'" in result
    
    def test_list_files_tool_path_wrong_type(self):
        """Test execution with non-string path argument."""
        tool = ListFilesTool()
        
        # Test with integer
        result1 = tool.execute(path=123)
        assert "Error: 'path' must be a string, got int" in result1
        
        # Test with list
        result2 = tool.execute(path=["/some/path"])
        assert "Error: 'path' must be a string, got list" in result2
        
        # Test with None
        result3 = tool.execute(path=None)
        assert "Error: 'path' must be a string, got NoneType" in result3
    
    def test_list_files_tool_empty_path(self):
        """Test execution with empty path."""
        tool = ListFilesTool()
        
        # Test with empty string
        result1 = tool.execute(path="")
        assert "Error: 'path' cannot be empty" in result1
        
        # Test with whitespace only
        result2 = tool.execute(path="   ")
        assert "Error: 'path' cannot be empty" in result2
        
        # Test with tab and newline
        result3 = tool.execute(path="\t\n")
        assert "Error: 'path' cannot be empty" in result3
    
    def test_list_files_tool_ignore_patterns_wrong_type(self):
        """Test execution with non-list ignore_patterns argument."""
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with string instead of list
            result1 = tool.execute(path=temp_dir, ignore_patterns="*.py")
            assert "Error: 'ignore_patterns' must be a list, got str" in result1
            
            # Test with integer
            result2 = tool.execute(path=temp_dir, ignore_patterns=123)
            assert "Error: 'ignore_patterns' must be a list, got int" in result2
    
    def test_list_files_tool_ignore_patterns_invalid_elements(self):
        """Test execution with non-string elements in ignore_patterns."""
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with mixed types in list
            result1 = tool.execute(path=temp_dir, ignore_patterns=["*.py", 123])
            assert "Error: All ignore patterns must be strings, got int" in result1
            
            # Test with None in list
            result2 = tool.execute(path=temp_dir, ignore_patterns=["*.py", None])
            assert "Error: All ignore patterns must be strings, got NoneType" in result2
    
    def test_list_files_tool_ignore_patterns_none(self):
        """Test execution with None ignore_patterns (should work)."""
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")
            
            result = tool.execute(path=temp_dir, ignore_patterns=None)
            
            # Should work and use empty list instead of None
            assert "Error:" not in result
            assert "test.txt" in result
    
    def test_list_files_tool_empty_ignore_patterns(self):
        """Test execution with empty ignore_patterns list."""
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.pyc").write_text("bytecode")
            
            result = tool.execute(path=temp_dir, ignore_patterns=[])
            
            # Should still apply default ignore patterns
            assert "Error:" not in result
            assert "test.pyc" not in result  # Should be ignored by default patterns


class TestListFilesToolErrorHandling:
    """Test ListFilesTool error handling scenarios."""
    
    def test_list_files_tool_nonexistent_path(self):
        """Test listing nonexistent directory."""
        tool = ListFilesTool()
        nonexistent_path = "/this/path/does/not/exist"
        
        result = tool.execute(path=nonexistent_path)
        
        assert "Error: Path" in result
        assert "not found" in result
        assert nonexistent_path in result
    
    def test_list_files_tool_file_instead_of_directory(self):
        """Test listing a file instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"content")
            temp_file_path = temp_file.name
        
        try:
            tool = ListFilesTool()
            result = tool.execute(path=temp_file_path)
            
            assert "Error:" in result
            assert "is not a directory" in result
            assert temp_file_path in result
            
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_list_files_tool_no_read_permission(self):
        """Test listing directory without read permission (Unix-like systems only)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")
            
            tool = ListFilesTool()
            
            try:
                # Remove read permission
                os.chmod(temp_dir, 0o000)
                
                result = tool.execute(path=temp_dir)
                
                assert "Error:" in result
                assert "permission" in result.lower()
                assert temp_dir in result
                
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_list_files_tool_permission_denied_subdirectory(self):
        """Test handling permission denied for subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create accessible files
            (temp_path / "accessible.txt").write_text("content")
            
            # Create subdirectory
            sub_dir = temp_path / "restricted"
            sub_dir.mkdir()
            (sub_dir / "hidden.txt").write_text("hidden content")
            
            tool = ListFilesTool()
            
            try:
                # Remove read permission from subdirectory
                os.chmod(sub_dir, 0o000)
                
                result = tool.execute(path=temp_dir)
                
                # Should handle gracefully
                assert "accessible.txt" in result
                assert "restricted/" in result
                assert "[Permission Denied]" in result or "[Access Error]" in result
                
            finally:
                # Restore permissions for cleanup
                os.chmod(sub_dir, 0o755)
    
    @patch('os.access')
    def test_list_files_tool_os_access_permission_error(self, mock_access):
        """Test handling OSError from os.access check."""
        mock_access.return_value = False
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool.execute(path=temp_dir)
            
            assert "Error: No read permission" in result
            assert temp_dir in result
    
    @patch('pathlib.Path.resolve')
    def test_list_files_tool_path_resolve_error(self, mock_resolve):
        """Test handling errors during path resolution."""
        mock_resolve.side_effect = OSError("Path resolution failed")
        tool = ListFilesTool()
        
        result = tool.execute(path="/some/path")
        
        assert "Error: System error" in result
        assert "Path resolution failed" in result
    
    def test_list_files_tool_iterdir_error(self):
        """Test handling errors during directory iteration in _build_tree method."""
        tool = ListFilesTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock the _build_tree method to simulate PermissionError during iteration
            original_build_tree = tool._build_tree
            
            def mock_build_tree(directory, prefix, tree_lines, ignore_patterns, is_root=False):
                if is_root:
                    # Simulate permission error during root directory iteration
                    raise PermissionError("Cannot list directory")
                else:
                    return original_build_tree(directory, prefix, tree_lines, ignore_patterns, is_root)
            
            with patch.object(tool, '_build_tree', side_effect=mock_build_tree):
                result = tool.execute(path=temp_dir)
                
                assert "Error: Permission denied" in result
                assert temp_dir in result
    
    def test_list_files_tool_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        tool = ListFilesTool()
        
        with patch('pathlib.Path.resolve', side_effect=Exception("Unexpected error")):
            result = tool.execute(path="/some/path")
            
            assert "Error: Unexpected error" in result
            assert "Unexpected error" in result


class TestListFilesToolTreeStructure:
    """Test ListFilesTool tree structure formatting."""
    
    @pytest.fixture
    def complex_temp_dir(self):
        """Create a complex temporary directory structure for tree testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create various files and directories
            (temp_path / "a_file.txt").write_text("content")
            (temp_path / "z_file.py").write_text("content")
            
            # Create multiple subdirectories
            dir_a = temp_path / "a_dir"
            dir_a.mkdir()
            (dir_a / "file1.md").write_text("content")
            (dir_a / "file2.json").write_text("content")
            
            dir_z = temp_path / "z_dir"
            dir_z.mkdir()
            (dir_z / "nested_file.txt").write_text("content")
            
            # Create nested structure
            nested = dir_z / "nested_dir"
            nested.mkdir()
            (nested / "deep_file.log").write_text("content")
            
            yield temp_path
    
    def test_list_files_tool_tree_structure_symbols(self, complex_temp_dir):
        """Test that tree structure uses correct symbols."""
        tool = ListFilesTool()
        
        result = tool.execute(path=str(complex_temp_dir))
        
        # Should use proper tree symbols
        assert "├── " in result
        assert "└── " in result
        assert "│   " in result
        
        # Check structure format
        lines = result.split('\n')
        root_line = lines[0]
        assert root_line.endswith('/')
        
        # Should have proper indentation
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                assert line.startswith('├── ') or line.startswith('└── ') or \
                       line.startswith('│   ') or line.startswith('    ')
    
    def test_list_files_tool_directory_sorting(self, complex_temp_dir):
        """Test that directories and files are sorted correctly."""
        tool = ListFilesTool()
        
        result = tool.execute(path=str(complex_temp_dir))
        lines = result.split('\n')
        
        # Extract file/directory names from first level only (ignoring tree symbols and indentation)
        first_level_items = []
        for line in lines[1:]:  # Skip root directory line
            if line.strip() and ('├── ' in line or '└── ' in line):
                # Only consider first level items (no additional indentation)
                if line.startswith('├── ') or line.startswith('└── '):
                    # Extract the name after the tree symbol
                    if line.startswith('├── '):
                        name = line[4:]  # Remove '├── '
                    elif line.startswith('└── '):
                        name = line[4:]  # Remove '└── '
                    first_level_items.append(name)
        
        # Check that directories come before files (directories end with '/')
        # and both are sorted alphabetically
        dirs = [item for item in first_level_items if item.endswith('/')]
        files = [item for item in first_level_items if not item.endswith('/')]
        
        # Verify directories come before files in the output
        if dirs and files:
            # Find positions of first directory and first file
            first_dir_pos = first_level_items.index(dirs[0]) if dirs else float('inf')
            first_file_pos = first_level_items.index(files[0]) if files else float('inf')
            assert first_dir_pos < first_file_pos, "Directories should come before files"
        
        # Directories should be sorted
        if len(dirs) > 1:
            assert dirs == sorted(dirs, key=str.lower), f"Directories not sorted: {dirs}"
        
        # Files should be sorted 
        if len(files) > 1:
            assert files == sorted(files, key=str.lower), f"Files not sorted: {files}"
    
    def test_list_files_tool_directory_markers(self, complex_temp_dir):
        """Test that directories are marked with trailing slash."""
        tool = ListFilesTool()
        
        result = tool.execute(path=str(complex_temp_dir))
        
        # Directories should have trailing slash
        assert "a_dir/" in result
        assert "z_dir/" in result
        assert "nested_dir/" in result
        
        # Files should not have trailing slash
        assert "a_file.txt" in result and "a_file.txt/" not in result
        assert "file1.md" in result and "file1.md/" not in result


class TestListFilesToolIgnorePatternMatching:
    """Test ListFilesTool ignore pattern matching functionality."""
    
    @pytest.fixture
    def pattern_test_dir(self):
        """Create directory structure for testing pattern matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with various patterns
            (temp_path / "test.py").write_text("python")
            (temp_path / "test.pyc").write_text("bytecode")
            (temp_path / "test.pyo").write_text("optimized")
            (temp_path / "script.js").write_text("javascript")
            (temp_path / "style.css").write_text("styles")
            (temp_path / "data.json").write_text("data")
            (temp_path / "README.md").write_text("readme")
            (temp_path / "debug.log").write_text("logs")
            (temp_path / "info.log").write_text("more logs")
            (temp_path / "image.png").write_text("binary")
            
            # Create directories
            (temp_path / "logs").mkdir()
            (temp_path / "cache").mkdir()
            (temp_path / "temp_data").mkdir()
            
            yield temp_path
    
    def test_list_files_tool_glob_patterns(self, pattern_test_dir):
        """Test glob pattern matching."""
        tool = ListFilesTool()
        
        # Test wildcard patterns
        result = tool.execute(
            path=str(pattern_test_dir),
            ignore_patterns=["*.py*", "*.log"]
        )
        
        # Should ignore .py, .pyc, .pyo, and .log files
        assert "test.py" not in result
        assert "test.pyc" not in result
        assert "test.pyo" not in result
        assert "debug.log" not in result
        assert "info.log" not in result
        
        # Should include other files
        assert "script.js" in result
        assert "style.css" in result
        assert "data.json" in result
    
    def test_list_files_tool_exact_match_patterns(self, pattern_test_dir):
        """Test exact match patterns."""
        tool = ListFilesTool()
        
        result = tool.execute(
            path=str(pattern_test_dir),
            ignore_patterns=["logs", "cache", "README.md"]
        )
        
        # Should ignore exact matches
        assert "logs/" not in result
        assert "cache/" not in result
        assert "README.md" not in result
        
        # Should include non-matching items
        assert "temp_data/" in result
        assert "test.py" in result
    
    def test_list_files_tool_case_sensitivity(self, pattern_test_dir):
        """Test case sensitivity in pattern matching."""
        # Create a clean test directory for better case sensitivity testing
        with tempfile.TemporaryDirectory() as clean_dir:
            clean_path = Path(clean_dir)
            
            # Create files with distinctly different names to avoid filesystem case issues
            (clean_path / "lowercase.txt").write_text("content")
            (clean_path / "UPPERCASE.TXT").write_text("content")
            (clean_path / "MixedCase.Txt").write_text("content")
            
            tool = ListFilesTool()
            
            # Test exact match case sensitivity
            result_exact = tool.execute(
                path=clean_dir,
                ignore_patterns=["lowercase.txt"]  # Should only match exact case
            )
            
            # Should ignore only the exact lowercase match
            assert "lowercase.txt" not in result_exact
            assert "UPPERCASE.TXT" in result_exact  # Different name, should be included
            assert "MixedCase.Txt" in result_exact  # Different name, should be included
            
            # Test glob pattern case sensitivity  
            result_glob = tool.execute(
                path=clean_dir,
                ignore_patterns=["*.txt"]  # lowercase extension pattern
            )
            
            # Should only match the lowercase extension
            assert "lowercase.txt" not in result_glob  # Matches *.txt
            # Different extensions should be included (case sensitive)
            assert "UPPERCASE.TXT" in result_glob  # .TXT != .txt
            assert "MixedCase.Txt" in result_glob   # .Txt != .txt
    
    def test_list_files_tool_complex_patterns(self, pattern_test_dir):
        """Test complex glob patterns."""
        tool = ListFilesTool()
        
        # Test character classes and ranges
        result = tool.execute(
            path=str(pattern_test_dir),
            ignore_patterns=["test.[pj]*", "*temp*", "*.m[da]"]
        )
        
        # Should ignore files matching complex patterns
        assert "test.py" not in result  # Matches test.[pj]*
        assert "test.pyc" not in result  # Matches test.[pj]*
        assert "temp_data/" not in result  # Matches *temp*
        assert "README.md" not in result  # Matches *.m[da]
        
        # Should include files not matching patterns
        assert "script.js" in result
        assert "style.css" in result
        assert "data.json" in result


class TestListFilesToolEdgeCases:
    """Test ListFilesTool edge cases and boundary conditions."""
    
    def test_list_files_tool_very_deep_nesting(self):
        """Test handling of very deep directory nesting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create deeply nested structure
            current = temp_path
            for i in range(10):  # Create 10 levels deep
                current = current / f"level_{i}"
                current.mkdir()
                (current / f"file_{i}.txt").write_text(f"content {i}")
            
            tool = ListFilesTool()
            result = tool.execute(path=temp_dir)
            
            # Should handle deep nesting
            assert "level_0/" in result
            assert "level_9/" in result
            assert "file_0.txt" in result
            assert "file_9.txt" in result
            
            # Check that we have proper nesting structure
            lines = result.split('\n')
            
            # Count the depth by looking at how many levels we have
            level_count = 0
            for line in lines:
                if "level_" in line and line.strip().endswith('/'):
                    level_count += 1
            
            # Should have all 10 levels
            assert level_count == 10
            
            # Check that deepest file appears (indicating proper traversal)
            assert "file_9.txt" in result
    
    def test_list_files_tool_many_files_in_directory(self):
        """Test handling of directory with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create many files
            for i in range(100):
                (temp_path / f"file_{i:03d}.txt").write_text(f"content {i}")
            
            tool = ListFilesTool()
            result = tool.execute(path=temp_dir)
            
            # Should handle many files
            assert "file_000.txt" in result
            assert "file_099.txt" in result
            
            # Should be sorted properly
            lines = result.split('\n')
            file_lines = [line for line in lines if "file_" in line]
            
            # Extract file numbers and check sorting
            file_numbers = []
            for line in file_lines:
                if "file_" in line:
                    # Extract the number from file_XXX.txt
                    start = line.find("file_") + 5
                    end = line.find(".txt")
                    if start < end:
                        file_numbers.append(int(line[start:end]))
            
            # Should be sorted numerically (as strings)
            assert file_numbers == sorted(file_numbers)
    
    def test_list_files_tool_unicode_filenames(self):
        """Test handling of unicode characters in filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with unicode names
            try:
                (temp_path / "файл.txt").write_text("russian")
                (temp_path / "文件.py").write_text("chinese")
                (temp_path / "archivo_señor.md").write_text("spanish")
                (temp_path / "🚀_rocket.json").write_text("emoji")
                
                tool = ListFilesTool()
                result = tool.execute(path=temp_dir)
                
                # Should handle unicode filenames
                assert "файл.txt" in result
                assert "文件.py" in result
                assert "archivo_señor.md" in result
                assert "🚀_rocket.json" in result
                
            except (UnicodeError, OSError):
                # Skip if filesystem doesn't support unicode
                pytest.skip("Filesystem doesn't support unicode filenames")
    
    def test_list_files_tool_special_characters_in_filenames(self):
        """Test handling of special characters in filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with special characters (where supported)
            special_files = [
                "file with spaces.txt",
                "file-with-dashes.py",
                "file_with_underscores.md",
                "file.with.dots.json",
                "file&with&ampersands.txt"  # Changed from .log to .txt to avoid default ignore
            ]
            
            created_files = []
            for filename in special_files:
                try:
                    (temp_path / filename).write_text("content")
                    created_files.append(filename)
                except (OSError, ValueError):
                    # Skip files that can't be created on this system
                    continue
            
            if created_files:
                tool = ListFilesTool()
                result = tool.execute(path=temp_dir)
                
                # Should handle special characters (except those filtered by default patterns)
                for filename in created_files:
                    # Check if the file should be ignored by default patterns
                    should_be_ignored = any(
                        filename.endswith(ext) for ext in ['.log', '.pyc', '.pyo', '.pyd']
                    ) or any(
                        pattern in filename for pattern in ['.DS_Store', 'Thumbs.db']
                    )
                    
                    if not should_be_ignored:
                        assert filename in result, f"File {filename} should be in result but isn't"
    
    def test_list_files_tool_symlink_handling(self):
        """Test handling of symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create target file and directory
            target_file = temp_path / "target.txt"
            target_file.write_text("target content")
            
            target_dir = temp_path / "target_dir"
            target_dir.mkdir()
            (target_dir / "nested.txt").write_text("nested content")
            
            try:
                # Create symbolic links
                file_link = temp_path / "file_link.txt"
                dir_link = temp_path / "dir_link"
                
                os.symlink(target_file, file_link)
                os.symlink(target_dir, dir_link)
                
                tool = ListFilesTool()
                result = tool.execute(path=temp_dir)
                
                # Should include symlinks in listing
                assert "file_link.txt" in result
                assert "dir_link/" in result
                assert "target.txt" in result
                assert "target_dir/" in result
                
                # Should be able to traverse symlinked directories
                assert "nested.txt" in result
                
            except (OSError, NotImplementedError):
                # Skip on systems that don't support symlinks
                pytest.skip("Symbolic links not supported on this system")
    
    def test_list_files_tool_circular_symlinks(self):
        """Test handling of circular symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Create circular symlink
                circular_link = temp_path / "circular"
                os.symlink(temp_path, circular_link)
                
                tool = ListFilesTool()
                result = tool.execute(path=temp_dir)
                
                # Should handle circular symlinks gracefully
                assert isinstance(result, str)
                assert "circular/" in result
                # Should not get stuck in infinite loop
                
            except (OSError, NotImplementedError):
                # Skip on systems that don't support symlinks
                pytest.skip("Symbolic links not supported on this system")
    
    def test_list_files_tool_additional_arguments(self):
        """Test that additional arguments are ignored gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")
            
            tool = ListFilesTool()
            
            result = tool.execute(
                path=temp_dir,
                ignore_patterns=[],
                extra_arg="ignored",
                another_arg=123,
                yet_another={"key": "value"}
            )
            
            # Should work normally and ignore extra arguments
            assert "Error:" not in result
            assert "test.txt" in result
    
    def test_list_files_tool_relative_vs_absolute_paths(self):
        """Test handling of relative vs absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")
            
            tool = ListFilesTool()
            
            # Test absolute path
            result1 = tool.execute(path=temp_dir)
            assert "test.txt" in result1
            
            # Test relative path (from current working directory)
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_path.parent)
                relative_path = temp_path.name
                
                result2 = tool.execute(path=relative_path)
                assert "test.txt" in result2
                
            finally:
                os.chdir(original_cwd)


class TestListFilesToolPerformance:
    """Test ListFilesTool performance characteristics."""
    
    def test_list_files_tool_large_directory_structure(self):
        """Test performance with large directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create moderately large structure
            for i in range(50):  # 50 directories
                dir_path = temp_path / f"dir_{i:02d}"
                dir_path.mkdir()
                
                for j in range(20):  # 20 files per directory
                    (dir_path / f"file_{j:02d}.txt").write_text(f"content {i}:{j}")
            
            tool = ListFilesTool()
            
            import time
            start_time = time.time()
            result = tool.execute(path=temp_dir)
            end_time = time.time()
            
            # Should complete in reasonable time (< 5 seconds)
            assert end_time - start_time < 5.0
            
            # Should produce correct output
            assert "dir_00/" in result
            assert "dir_49/" in result
            assert "file_00.txt" in result
            assert "file_19.txt" in result
            
            # Count total items (approximately 50 dirs + 1000 files)
            lines = result.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            assert len(non_empty_lines) > 1000  # Should have many items
    
    def test_list_files_tool_memory_efficiency(self):
        """Test memory efficiency with large file lists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create many files with long names
            for i in range(1000):
                long_name = f"very_long_filename_with_many_characters_{i:04d}.txt"
                (temp_path / long_name).write_text("content")
            
            tool = ListFilesTool()
            result = tool.execute(path=temp_dir)
            
            # Should handle large output efficiently
            assert isinstance(result, str)
            assert len(result) > 50000  # Should be substantial output
            
            # Should contain all files
            assert "very_long_filename_with_many_characters_0000.txt" in result
            assert "very_long_filename_with_many_characters_0999.txt" in result


class TestWriteFileToolInitialization:
    """Test WriteFileTool initialization."""
    
    def test_write_file_tool_initialization(self):
        """Test WriteFileTool initialization sets correct attributes."""
        tool = WriteFileTool()
        
        assert tool.name == "write_file"
        assert "Записывает или перезаписывает контент" in tool.description
        assert "file_path" in tool.description
        assert "content" in tool.description
        assert isinstance(tool.description, str)
    
    def test_write_file_tool_string_representation(self):
        """Test WriteFileTool string representation."""
        tool = WriteFileTool()
        
        assert str(tool) == "WriteFileTool(name='write_file')"
        assert repr(tool) == "WriteFileTool(name='write_file')"


class TestWriteFileToolExecution:
    """Test WriteFileTool execution with various scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_write_file_tool_create_new_file(self, temp_workspace):
        """Test successful creation of a new file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "new_file.txt"
        content = "Hello, World!"
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        assert "13 bytes" in result  # UTF-8 byte count for "Hello, World!"
        assert str(target_file) in result
        
        # Verify file was created with correct content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content
    
    def test_write_file_tool_overwrite_existing_file(self, temp_workspace):
        """Test overwriting an existing file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "existing_file.txt"
        original_content = "Original content"
        new_content = "New content"
        
        # Create file with original content
        target_file.write_text(original_content, encoding='utf-8')
        assert target_file.read_text(encoding='utf-8') == original_content
        
        # Overwrite with new content
        result = tool.execute(file_path=str(target_file), content=new_content)
        
        # Check success message
        assert "Successfully wrote" in result
        assert "11 bytes" in result  # UTF-8 byte count for "New content"
        assert str(target_file) in result
        
        # Verify file was overwritten
        assert target_file.read_text(encoding='utf-8') == new_content
    
    def test_write_file_tool_create_parent_directories(self, temp_workspace):
        """Test automatic creation of parent directories."""
        tool = WriteFileTool()
        target_file = temp_workspace / "deep" / "nested" / "structure" / "file.txt"
        content = "Content in nested file"
        
        # Ensure parent directories don't exist
        assert not target_file.parent.exists()
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        assert "22 bytes" in result  # UTF-8 byte count for "Content in nested file"
        assert str(target_file) in result
        
        # Verify parent directories were created
        assert target_file.parent.exists()
        assert target_file.parent.is_dir()
        
        # Verify file was created with correct content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content
    
    def test_write_file_tool_empty_content(self, temp_workspace):
        """Test writing empty content to a file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "empty_file.txt"
        content = ""
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        assert "0 bytes" in result
        assert str(target_file) in result
        
        # Verify file was created with empty content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == ""
        assert target_file.stat().st_size == 0
    
    def test_write_file_tool_unicode_content(self, temp_workspace):
        """Test writing Unicode content to a file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "unicode_file.txt"
        content = "Привет, мир! 🌍 文字 ñoño"
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message (Unicode characters take more bytes than characters)
        assert "Successfully wrote" in result
        content_bytes = len(content.encode('utf-8'))
        assert f"{content_bytes} bytes" in result
        assert str(target_file) in result
        
        # Verify file was created with correct Unicode content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content
    
    def test_write_file_tool_large_content(self, temp_workspace):
        """Test writing large content to a file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "large_file.txt"
        # Create content with 10,000 characters
        content = "A" * 10000
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        assert "10000 bytes" in result
        assert str(target_file) in result
        
        # Verify file was created with correct content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content
        assert target_file.stat().st_size == 10000
    
    def test_write_file_tool_multiline_content(self, temp_workspace):
        """Test writing multiline content to a file."""
        tool = WriteFileTool()
        target_file = temp_workspace / "multiline_file.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        content_bytes = len(content.encode('utf-8'))
        assert f"{content_bytes} bytes" in result
        assert str(target_file) in result
        
        # Verify file was created with correct multiline content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content
        lines = target_file.read_text(encoding='utf-8').splitlines()
        assert len(lines) == 3
        assert lines == ["Line 1", "Line 2", "Line 3"]
    
    def test_write_file_tool_special_characters_content(self, temp_workspace):
        """Test writing content with special characters."""
        tool = WriteFileTool()
        target_file = temp_workspace / "special_chars.txt"
        content = "Special chars: \t\n\\\"'`!@#$%^&*()[]{}|;:,.<>?/~"
        
        result = tool.execute(file_path=str(target_file), content=content)
        
        # Check success message
        assert "Successfully wrote" in result
        content_bytes = len(content.encode('utf-8'))
        assert f"{content_bytes} bytes" in result
        
        # Verify file was created with correct content
        assert target_file.exists()
        assert target_file.read_text(encoding='utf-8') == content


class TestWriteFileToolArgumentValidation:
    """Test WriteFileTool argument validation."""
    
    def test_write_file_tool_missing_file_path_argument(self):
        """Test execution without file_path argument."""
        tool = WriteFileTool()
        
        result = tool.execute(content="test content")
        
        assert "Error: Missing required argument 'file_path'" in result
    
    def test_write_file_tool_missing_content_argument(self):
        """Test execution without content argument."""
        tool = WriteFileTool()
        
        result = tool.execute(file_path="/tmp/test.txt")
        
        assert "Error: Missing required argument 'content'" in result
    
    def test_write_file_tool_missing_both_arguments(self):
        """Test execution without any required arguments."""
        tool = WriteFileTool()
        
        result = tool.execute()
        
        assert "Error: Missing required argument 'file_path'" in result
    
    def test_write_file_tool_file_path_wrong_type(self):
        """Test execution with non-string file_path argument."""
        tool = WriteFileTool()
        
        # Test with integer
        result1 = tool.execute(file_path=123, content="test")
        assert "Error: 'file_path' must be a string, got int" in result1
        
        # Test with list
        result2 = tool.execute(file_path=["/tmp/test.txt"], content="test")
        assert "Error: 'file_path' must be a string, got list" in result2
        
        # Test with None
        result3 = tool.execute(file_path=None, content="test")
        assert "Error: 'file_path' must be a string, got NoneType" in result3
        
        # Test with dictionary
        result4 = tool.execute(file_path={"path": "/tmp/test.txt"}, content="test")
        assert "Error: 'file_path' must be a string, got dict" in result4
    
    def test_write_file_tool_content_wrong_type(self):
        """Test execution with non-string content argument."""
        tool = WriteFileTool()
        
        # Test with integer
        result1 = tool.execute(file_path="/tmp/test.txt", content=123)
        assert "Error: 'content' must be a string, got int" in result1
        
        # Test with list
        result2 = tool.execute(file_path="/tmp/test.txt", content=["test", "content"])
        assert "Error: 'content' must be a string, got list" in result2
        
        # Test with None
        result3 = tool.execute(file_path="/tmp/test.txt", content=None)
        assert "Error: 'content' must be a string, got NoneType" in result3
        
        # Test with bytes
        result4 = tool.execute(file_path="/tmp/test.txt", content=b"test")
        assert "Error: 'content' must be a string, got bytes" in result4
        
        # Test with boolean
        result5 = tool.execute(file_path="/tmp/test.txt", content=True)
        assert "Error: 'content' must be a string, got bool" in result5
    
    def test_write_file_tool_empty_file_path(self):
        """Test execution with empty file_path."""
        tool = WriteFileTool()
        
        # Test with empty string
        result1 = tool.execute(file_path="", content="test")
        assert "Error: 'file_path' cannot be empty" in result1
        
        # Test with whitespace only
        result2 = tool.execute(file_path="   ", content="test")
        assert "Error: 'file_path' cannot be empty" in result2
        
        # Test with tab and newline
        result3 = tool.execute(file_path="\t\n", content="test")
        assert "Error: 'file_path' cannot be empty" in result3
    
    def test_write_file_tool_additional_arguments(self):
        """Test that additional arguments are ignored gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.txt"
            
            tool = WriteFileTool()
            
            result = tool.execute(
                file_path=str(target_file),
                content="test content",
                extra_arg="ignored",
                another_arg=123,
                yet_another={"key": "value"}
            )
            
            # Should work normally and ignore extra arguments
            assert "Successfully wrote" in result
            assert target_file.exists()
            assert target_file.read_text(encoding='utf-8') == "test content"


class TestWriteFileToolErrorHandling:
    """Test WriteFileTool error handling scenarios."""
    
    def test_write_file_tool_invalid_file_path_characters(self):
        """Test handling of invalid file path characters."""
        tool = WriteFileTool()
        
        # Test with null character (invalid in most filesystems)
        result1 = tool.execute(file_path="/tmp/test\x00file.txt", content="test")
        assert "Error:" in result1
        
        # Results may vary by filesystem, but should handle gracefully
        assert ("Invalid file path" in result1 or "Cannot" in result1 or 
                "Unexpected error" in result1 or "embedded null character" in result1)
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_write_file_tool_no_write_permission_directory(self):
        """Test handling permission denied for parent directory (Unix-like systems only)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            restricted_dir = temp_path / "restricted"
            restricted_dir.mkdir()
            
            tool = WriteFileTool()
            target_file = restricted_dir / "test.txt"
            
            try:
                # Remove write permission from parent directory
                os.chmod(restricted_dir, 0o444)  # Read-only
                
                result = tool.execute(file_path=str(target_file), content="test content")
                
                assert "Error:" in result
                assert "permission" in result.lower()
                
            finally:
                # Restore permissions for cleanup
                os.chmod(restricted_dir, 0o755)
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_write_file_tool_no_write_permission_existing_file(self):
        """Test handling permission denied for existing file (Unix-like systems only)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target_file = temp_path / "readonly_file.txt"
            
            # Create file with initial content
            target_file.write_text("original content")
            
            tool = WriteFileTool()
            
            try:
                # Remove write permission from file
                os.chmod(target_file, 0o444)  # Read-only
                
                result = tool.execute(file_path=str(target_file), content="new content")
                
                assert "Error:" in result
                assert "Permission denied" in result or "permission" in result.lower()
                
                # Original content should remain unchanged
                assert target_file.read_text(encoding='utf-8') == "original content"
                
            finally:
                # Restore permissions for cleanup
                os.chmod(target_file, 0o644)
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_write_file_tool_cannot_create_parent_directories(self):
        """Test handling permission denied when creating parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            restricted_parent = temp_path / "restricted"
            restricted_parent.mkdir()
            
            tool = WriteFileTool()
            target_file = restricted_parent / "nested" / "test.txt"
            
            try:
                # Remove write permission from parent directory
                os.chmod(restricted_parent, 0o444)  # Read-only
                
                result = tool.execute(file_path=str(target_file), content="test content")
                
                assert "Error:" in result
                assert ("Permission denied when creating parent directories" in result or
                        "permission" in result.lower())
                
                # File should not exist (use try/except to handle permission errors)
                try:
                    file_exists = target_file.exists()
                    assert not file_exists
                except PermissionError:
                    # If we can't even check if file exists due to permissions, that's also valid
                    pass
                
            finally:
                # Restore permissions for cleanup
                os.chmod(restricted_parent, 0o755)
    
    def test_write_file_tool_path_resolution_error(self):
        """Test handling errors during path resolution."""
        tool = WriteFileTool()
        
        with patch('pathlib.Path.resolve', side_effect=OSError("Path resolution failed")):
            result = tool.execute(file_path="/some/path/test.txt", content="test")
            
            assert "Error:" in result
            assert "Invalid file path" in result
            assert "Path resolution failed" in result
    
    def test_write_file_tool_mkdir_os_error(self):
        """Test handling OSError during parent directory creation."""
        tool = WriteFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "deep" / "nested" / "test.txt"
            
            with patch('pathlib.Path.mkdir', side_effect=OSError("Cannot create directory")):
                result = tool.execute(file_path=str(target_file), content="test")
                
                assert "Error:" in result
                assert "Cannot create parent directories" in result
                assert "Cannot create directory" in result
    
    def test_write_file_tool_file_write_os_error(self):
        """Test handling OSError during file writing."""
        tool = WriteFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.txt"
            
            with patch('builtins.open', side_effect=OSError("Disk full")):
                result = tool.execute(file_path=str(target_file), content="test")
                
                assert "Error:" in result
                assert "Cannot write to file" in result
                assert "Disk full" in result
    
    def test_write_file_tool_content_encoding_handling(self):
        """Test that content is properly encoded to UTF-8."""
        tool = WriteFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "encoding_test.txt"
            
            # Test various encodable content
            test_contents = [
                "Simple ASCII text",
                "Unicode: 文字 ñoño 🚀",
                "Special: \t\n\"'",  # Removed \r as it gets normalized on some systems
                "",  # Empty content
            ]
            
            for i, content in enumerate(test_contents):
                test_file = Path(temp_dir) / f"test_{i}.txt"
                result = tool.execute(file_path=str(test_file), content=content)
                
                assert "Successfully wrote" in result
                expected_bytes = len(content.encode('utf-8'))
                assert f"{expected_bytes} bytes" in result
                
                # Verify file content matches exactly
                assert test_file.read_text(encoding='utf-8') == content
    
    def test_write_file_tool_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        tool = WriteFileTool()
        
        with patch('pathlib.Path.resolve', side_effect=Exception("Unexpected error")):
            result = tool.execute(file_path="/some/path/test.txt", content="test")
            
            assert "Error:" in result
            assert "Unexpected error when writing to file" in result
            assert "Unexpected error" in result
    
    def test_write_file_tool_access_check_error(self):
        """Test handling errors during directory access check."""
        tool = WriteFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.txt"
            
            with patch('os.access', return_value=False):
                result = tool.execute(file_path=str(target_file), content="test")
                
                assert "Error:" in result
                assert "No write permission for directory" in result


class TestWriteFileToolSpecialScenarios:
    """Test WriteFileTool special scenarios and edge cases."""
    
    def test_write_file_tool_absolute_vs_relative_paths(self):
        """Test handling of absolute vs relative file paths."""
        tool = WriteFileTool()
        content = "test content"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test absolute path
            abs_file = temp_path / "absolute_test.txt"
            result1 = tool.execute(file_path=str(abs_file), content=content)
            
            assert "Successfully wrote" in result1
            assert abs_file.exists()
            assert abs_file.read_text(encoding='utf-8') == content
            
            # Test relative path (from current working directory)
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                rel_file = "relative_test.txt"
                
                result2 = tool.execute(file_path=rel_file, content=content)
                
                assert "Successfully wrote" in result2
                assert (temp_path / rel_file).exists()
                assert (temp_path / rel_file).read_text(encoding='utf-8') == content
                
            finally:
                os.chdir(original_cwd)
    
    def test_write_file_tool_unicode_filenames(self):
        """Test handling of Unicode characters in filenames."""
        tool = WriteFileTool()
        content = "Unicode filename test"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test various Unicode filenames
            unicode_files = [
                "файл.txt",       # Russian
                "文件.py",        # Chinese
                "archivo_señor.md",  # Spanish with ñ
                "🚀_rocket.json"     # Emoji
            ]
            
            for filename in unicode_files:
                try:
                    target_file = temp_path / filename
                    result = tool.execute(file_path=str(target_file), content=content)
                    
                    # Should handle Unicode filenames
                    assert "Successfully wrote" in result
                    assert target_file.exists()
                    assert target_file.read_text(encoding='utf-8') == content
                    
                except (UnicodeError, OSError):
                    # Skip if filesystem doesn't support unicode
                    pytest.skip(f"Filesystem doesn't support unicode filename: {filename}")
    
    def test_write_file_tool_special_characters_in_filenames(self):
        """Test handling of special characters in filenames."""
        tool = WriteFileTool()
        content = "Special filename test"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test special characters in filenames (where supported)
            special_files = [
                "file with spaces.txt",
                "file-with-dashes.py",
                "file_with_underscores.md",
                "file.with.dots.json",
                "file(with)parentheses.txt",
                "file[with]brackets.py"
            ]
            
            for filename in special_files:
                try:
                    target_file = temp_path / filename
                    result = tool.execute(file_path=str(target_file), content=content)
                    
                    assert "Successfully wrote" in result
                    assert target_file.exists()
                    assert target_file.read_text(encoding='utf-8') == content
                    
                except (OSError, ValueError):
                    # Skip files that can't be created on this system
                    continue
    
    def test_write_file_tool_very_long_filename(self):
        """Test handling of very long filenames."""
        tool = WriteFileTool()
        content = "Long filename test"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a filename that's close to but under most filesystem limits
            long_name = "a" * 200 + ".txt"  # 200 characters + extension
            target_file = temp_path / long_name
            
            try:
                result = tool.execute(file_path=str(target_file), content=content)
                
                # Should handle long filenames or provide appropriate error
                if "Successfully wrote" in result:
                    assert target_file.exists()
                    assert target_file.read_text(encoding='utf-8') == content
                else:
                    # Should provide meaningful error message
                    assert "Error:" in result
                    
            except OSError:
                # Some filesystems have stricter limits
                pytest.skip("Filesystem doesn't support this filename length")
    
    def test_write_file_tool_deeply_nested_directories(self):
        """Test creation of deeply nested directory structures."""
        tool = WriteFileTool()
        content = "Deep nesting test"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a deeply nested path (20 levels)
            nested_path = temp_path
            for i in range(20):
                nested_path = nested_path / f"level_{i:02d}"
            
            target_file = nested_path / "deep_file.txt"
            
            result = tool.execute(file_path=str(target_file), content=content)
            
            assert "Successfully wrote" in result
            assert target_file.exists()
            assert target_file.read_text(encoding='utf-8') == content
            
            # Verify all intermediate directories were created
            current = target_file.parent
            level_count = 0
            while current != temp_path:
                assert current.exists()
                assert current.is_dir()
                level_count += 1
                current = current.parent
            
            assert level_count == 20
    
    def test_write_file_tool_byte_count_accuracy(self):
        """Test accuracy of byte count reporting for various content types."""
        tool = WriteFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            test_cases = [
                ("", 0),  # Empty content
                ("a", 1),  # Single ASCII character
                ("hello", 5),  # ASCII string
                ("hello\n", 6),  # ASCII with newline
                ("Привет", 12),  # Cyrillic (2 bytes per char in UTF-8)
                ("🚀", 4),  # Emoji (4 bytes in UTF-8)
                ("Hello 🌍 World", 16),  # Mixed ASCII and emoji
                ("文字", 6),  # Chinese characters (3 bytes each in UTF-8)
                ("café", 5),  # Accented characters
            ]
            
            for i, (content, expected_bytes) in enumerate(test_cases):
                target_file = temp_path / f"test_{i}.txt"
                
                result = tool.execute(file_path=str(target_file), content=content)
                
                assert "Successfully wrote" in result
                assert f"{expected_bytes} bytes" in result
                
                # Verify actual file size matches reported size
                actual_size = target_file.stat().st_size
                assert actual_size == expected_bytes
                
                # Verify content encoding matches expectation
                encoded_size = len(content.encode('utf-8'))
                assert encoded_size == expected_bytes
    
    def test_write_file_tool_concurrent_directory_creation(self):
        """Test handling of concurrent directory creation scenarios."""
        tool = WriteFileTool()
        content = "Concurrent test"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target_file = temp_path / "new_dir" / "test.txt"
            
            # Simulate race condition where directory is created between exist check and mkdir
            original_mkdir = Path.mkdir
            def mock_mkdir(self, mode=0o777, parents=False, exist_ok=False):
                if not exist_ok and self.exists():
                    raise FileExistsError("Directory already exists")
                return original_mkdir(self, mode, parents, exist_ok)
            
            with patch('pathlib.Path.mkdir', mock_mkdir):
                result = tool.execute(file_path=str(target_file), content=content)
                
                # Should handle gracefully due to exist_ok=True
                assert "Successfully wrote" in result
                assert target_file.exists()
                assert target_file.read_text(encoding='utf-8') == content