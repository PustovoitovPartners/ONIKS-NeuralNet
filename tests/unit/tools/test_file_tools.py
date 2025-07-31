"""Unit tests for the ReadFileTool class."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from oniks.tools.file_tools import ReadFileTool


class TestReadFileToolInitialization:
    """Test ReadFileTool initialization."""
    
    def test_read_file_tool_initialization(self):
        """Test ReadFileTool initialization sets correct attributes."""
        tool = ReadFileTool()
        
        assert tool.name == "read_file"
        assert "Reads the entire content of a specified file" in tool.description
        assert "file_path" in tool.description
        assert isinstance(tool.description, str)
    
    def test_read_file_tool_string_representation(self):
        """Test ReadFileTool string representation."""
        tool = ReadFileTool()
        
        assert str(tool) == "ReadFileTool(name='read_file')"
        assert repr(tool) == "ReadFileTool(name='read_file')"


class TestReadFileToolExecution:
    """Test ReadFileTool execution with various scenarios."""
    
    @pytest.fixture
    def temp_file_with_content(self):
        """Create a temporary file with test content."""
        content = "This is a test file.\nSecond line of content.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        yield temp_path, content
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def empty_temp_file(self):
        """Create an empty temporary file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_read_file_tool_successful_read(self, temp_file_with_content):
        """Test successful file reading."""
        temp_path, expected_content = temp_file_with_content
        tool = ReadFileTool()
        
        result = tool.execute(file_path=temp_path)
        
        assert result == expected_content
        assert "This is a test file." in result
        assert "Second line of content." in result
    
    def test_read_file_tool_empty_file(self, empty_temp_file):
        """Test reading empty file."""
        tool = ReadFileTool()
        
        result = tool.execute(file_path=empty_temp_file)
        
        assert result == ""
    
    def test_read_file_tool_missing_file_path_argument(self):
        """Test execution without file_path argument."""
        tool = ReadFileTool()
        
        result = tool.execute()
        
        assert "Error: Missing required argument 'file_path'" in result
    
    def test_read_file_tool_file_path_wrong_type(self):
        """Test execution with non-string file_path argument."""
        tool = ReadFileTool()
        
        # Test with integer
        result1 = tool.execute(file_path=123)
        assert "Error: 'file_path' must be a string, got int" in result1
        
        # Test with list
        result2 = tool.execute(file_path=["/path/to/file"])
        assert "Error: 'file_path' must be a string, got list" in result2
        
        # Test with None
        result3 = tool.execute(file_path=None)
        assert "Error: 'file_path' must be a string, got NoneType" in result3
    
    def test_read_file_tool_empty_file_path(self):
        """Test execution with empty file_path."""
        tool = ReadFileTool()
        
        # Test with empty string
        result1 = tool.execute(file_path="")
        assert "Error: 'file_path' cannot be empty" in result1
        
        # Test with whitespace only
        result2 = tool.execute(file_path="   ")
        assert "Error: 'file_path' cannot be empty" in result2
        
        # Test with tab and newline
        result3 = tool.execute(file_path="\t\n")
        assert "Error: 'file_path' cannot be empty" in result3
    
    def test_read_file_tool_nonexistent_file(self):
        """Test reading nonexistent file."""
        tool = ReadFileTool()
        nonexistent_path = "/this/path/does/not/exist.txt"
        
        result = tool.execute(file_path=nonexistent_path)
        
        assert "Error: File" in result
        assert "not found" in result
        assert nonexistent_path in result
    
    def test_read_file_tool_directory_instead_of_file(self):
        """Test reading a directory instead of a file."""
        tool = ReadFileTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool.execute(file_path=temp_dir)
            
            assert "Error:" in result
            assert "is not a file" in result
            assert temp_dir in result
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific permission test")
    def test_read_file_tool_no_read_permission(self, temp_file_with_content):
        """Test reading file without read permission (Unix-like systems only)."""
        temp_path, _ = temp_file_with_content
        tool = ReadFileTool()
        
        try:
            # Remove read permission
            os.chmod(temp_path, 0o000)
            
            result = tool.execute(file_path=temp_path)
            
            assert "Error:" in result
            assert "permission" in result.lower()
            assert temp_path in result
            
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_path, 0o644)
    
    def test_read_file_tool_unicode_content(self):
        """Test reading file with unicode content."""
        unicode_content = "Hello 世界!\nПривет мир!\n🌍🚀\náñüñÑ"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(unicode_content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            assert result == unicode_content
            assert "世界" in result
            assert "Привет" in result
            assert "🌍🚀" in result
            assert "áñüñÑ" in result
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_tool_large_file(self):
        """Test reading large file."""
        large_content = "Line {}\n".format("x" * 1000) * 1000  # ~1MB content
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(large_content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            assert len(result) == len(large_content)
            assert result == large_content
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_tool_binary_file_handling(self):
        """Test handling of binary files."""
        binary_content = b'\x00\x01\x02\x03\xff\xfe\xfd'
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(binary_content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            assert "Error:" in result
            assert "binary data" in result or "encoding" in result
            assert temp_path in result
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_tool_different_encodings(self):
        """Test handling of files with different encodings."""
        # Create file with latin-1 encoding
        content = "Special chars: àáâãäå"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='latin-1', delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            # Should either read correctly (if system handles it) or report encoding error
            if "Error:" in result:
                assert "encoding" in result.lower() or "binary" in result.lower()
            else:
                # If it reads successfully, content should be preserved
                assert "Special chars:" in result
                
        finally:
            os.unlink(temp_path)
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_read_file_tool_permission_error_exception(self, mock_open_func):
        """Test handling of PermissionError exception."""
        tool = ReadFileTool()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True):
            
            result = tool.execute(file_path="/test/file.txt")
            
            assert "Error: Permission denied" in result
            assert "/test/file.txt" in result
    
    @patch('builtins.open', side_effect=OSError("System error"))
    def test_read_file_tool_os_error_exception(self, mock_open_func):
        """Test handling of OSError exception."""
        tool = ReadFileTool()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True):
            
            result = tool.execute(file_path="/test/file.txt")
            
            assert "Error: System error" in result
            assert "System error" in result
    
    @patch('builtins.open', side_effect=Exception("Unexpected error"))
    def test_read_file_tool_unexpected_exception(self, mock_open_func):
        """Test handling of unexpected exceptions."""
        tool = ReadFileTool()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True):
            
            result = tool.execute(file_path="/test/file.txt")
            
            assert "Error: Unexpected error" in result
            assert "Unexpected error" in result
    
    def test_read_file_tool_with_additional_arguments(self, temp_file_with_content):
        """Test that additional arguments are ignored."""
        temp_path, expected_content = temp_file_with_content
        tool = ReadFileTool()
        
        result = tool.execute(
            file_path=temp_path,
            extra_arg="ignored",
            another_arg=123
        )
        
        # Should work normally and ignore extra arguments
        assert result == expected_content
    
    def test_read_file_tool_multiline_content(self):
        """Test reading file with various line endings."""
        content_unix = "Line 1\nLine 2\nLine 3\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', newline='') as tmp:
            tmp.write(content_unix)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            assert "Line 1" in result
            assert "Line 2" in result
            assert "Line 3" in result
            # Verify newlines are preserved
            lines = result.split('\n')
            assert len(lines) == 4  # 3 lines + empty string after final newline
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_tool_special_characters_in_path(self):
        """Test handling of special characters in file path."""
        tool = ReadFileTool()
        
        # Test paths with special characters
        special_paths = [
            "/path with spaces/file.txt",
            "/path/with-dashes/file.txt",
            "/path/with_underscores/file.txt",
            "/path/with.dots/file.txt",
            "/path/with(parentheses)/file.txt"
        ]
        
        for path in special_paths:
            result = tool.execute(file_path=path)
            # Should handle gracefully (file doesn't exist, but path is processed)
            assert "Error: File" in result
            assert "not found" in result
            assert path in result


class TestReadFileToolEdgeCases:
    """Test ReadFileTool edge cases and boundary conditions."""
    
    def test_read_file_tool_very_long_path(self):
        """Test with very long file path."""
        tool = ReadFileTool()
        
        # Create a very long path (beyond typical OS limits)
        long_path = "/" + "very_long_directory_name" * 20 + "/file.txt"
        
        result = tool.execute(file_path=long_path)
        
        assert "Error:" in result
        assert "not found" in result
    
    def test_read_file_tool_path_traversal_attempts(self):
        """Test behavior with path traversal attempts."""
        tool = ReadFileTool()
        
        # Various path traversal patterns
        traversal_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/../../../../../../etc/passwd",
            "file://etc/passwd",
            "http://example.com/file.txt"
        ]
        
        for path in traversal_paths:
            result = tool.execute(file_path=path)
            # Should handle as regular file paths (most will result in "not found")
            assert isinstance(result, str)
            # The exact error depends on the system, but should be handled gracefully
    
    def test_read_file_tool_concurrent_access(self):
        """Test concurrent access to the same file."""
        import threading
        import time
        import tempfile
        import os
        
        # Create test file
        expected_content = "Test content for concurrent access\nLine 2\n"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(expected_content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            results = []
            errors = []
            
            def read_worker():
                try:
                    result = tool.execute(file_path=temp_path)
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
            
            # Create multiple threads to read the same file
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=read_worker)
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 10
            
            # All results should be the same
            for result in results:
                assert result == expected_content
                
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_file_tool_file_modification_during_read(self):
        """Test behavior when file is modified during read operation."""
        content = "Initial content\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            
            # Read the file
            result1 = tool.execute(file_path=temp_path)
            assert result1 == content
            
            # Modify the file
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write("Modified content\n")
            
            # Read again
            result2 = tool.execute(file_path=temp_path)
            assert result2 == "Modified content\n"
            assert result1 != result2
            
        finally:
            os.unlink(temp_path)
    
    def test_read_file_tool_symlink_handling(self):
        """Test handling of symbolic links (where supported)."""
        content = "Symlink target content\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            target_path = tmp.name
        
        try:
            # Create symlink (skip on systems that don't support it)
            symlink_path = target_path + "_link"
            try:
                os.symlink(target_path, symlink_path)
                
                tool = ReadFileTool()
                result = tool.execute(file_path=symlink_path)
                
                # Should read the content of the target file
                assert result == content
                
                os.unlink(symlink_path)
                
            except (OSError, NotImplementedError):
                # Skip test on systems that don't support symlinks
                pytest.skip("Symbolic links not supported on this system")
                
        finally:
            os.unlink(target_path)
    
    def test_read_file_tool_memory_efficiency_large_file(self):
        """Test memory efficiency with large files."""
        # Create a moderately large file (1MB) to test memory handling
        line_content = "This is line {} with some padding content to make it longer.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            for i in range(10000):  # Write 10,000 lines (approximately 1MB)
                tmp.write(line_content.format(i))
            temp_path = tmp.name
        
        try:
            tool = ReadFileTool()
            result = tool.execute(file_path=temp_path)
            
            # Verify content integrity
            lines = result.split('\n')
            assert len(lines) == 10001  # 10,000 lines + empty string after final newline
            assert "This is line 0 with some padding" in lines[0]
            assert "This is line 9999 with some padding" in lines[9999]
            
        finally:
            os.unlink(temp_path)