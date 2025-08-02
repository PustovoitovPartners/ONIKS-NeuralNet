"""File operation tools for the ONIKS NeuralNet framework.

This module provides tools for performing file operations during graph execution.
These tools can be used by agents to read, write, and manipulate files as part
of their workflow execution.
"""

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from oniks.tools.base import Tool


class ReadFileTool(Tool):
    """Tool for reading the entire content of a specified file.
    
    This tool provides a safe way to read file contents during graph execution.
    It handles common file errors gracefully and returns appropriate error messages
    when files cannot be read.
    
    The tool takes a single argument 'file_path' which should be a string path
    to the file to be read. It returns the entire content of the file as a string,
    or an error message if the file cannot be read.
    
    Example:
        >>> tool = ReadFileTool()
        >>> result = tool.execute(file_path="/path/to/file.txt")
        >>> print(result)
        File content here...
        
        >>> result = tool.execute(file_path="/nonexistent/file.txt")
        >>> print(result)
        Error: File '/nonexistent/file.txt' not found
    """
    
    def __init__(self) -> None:
        """Initialize the ReadFileTool with name and description."""
        super().__init__()
        self.name = "read_file"
        self.description = (
            "Reads the entire content of a specified file. "
            "Arguments: {'file_path': 'str'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Read the entire content of the specified file.
        
        Args:
            **kwargs: Must contain 'file_path' key with string value pointing
                     to the file to be read.
        
        Returns:
            String containing the entire file content, or an error message
            if the file cannot be read.
        
        Raises:
            KeyError: If 'file_path' argument is not provided.
            TypeError: If 'file_path' is not a string.
        """
        # Validate required argument
        if 'file_path' not in kwargs:
            return "Error: Missing required argument 'file_path'"
        
        file_path = kwargs['file_path']
        
        # Validate argument type
        if not isinstance(file_path, str):
            return f"Error: 'file_path' must be a string, got {type(file_path).__name__}"
        
        # Validate file path is not empty
        if not file_path.strip():
            return "Error: 'file_path' cannot be empty"
        
        try:
            # Check if file exists before attempting to read
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' not found"
            
            # Check if path is actually a file (not a directory)
            if not os.path.isfile(file_path):
                return f"Error: '{file_path}' is not a file"
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                return f"Error: No read permission for file '{file_path}'"
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return content
            
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except PermissionError:
            return f"Error: Permission denied when reading file '{file_path}'"
        except UnicodeDecodeError:
            # Try reading as binary and return a message about encoding
            try:
                with open(file_path, 'rb') as file:
                    # Just check if file can be opened
                    file.read(1024)  # Read first 1KB to check
                return f"Error: File '{file_path}' contains binary data or uses unsupported encoding"
            except Exception:
                return f"Error: Unable to read file '{file_path}' due to encoding issues"
        except OSError as e:
            return f"Error: System error when reading file '{file_path}': {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when reading file '{file_path}': {str(e)}"


class FileSearchReplaceTool(Tool):
    """Powerful, reliable and secure file search and replace tool.
    
    This tool provides advanced search and replace functionality with regex support,
    atomic file operations, backup creation, and comprehensive security measures.
    It supports both simple string replacement and complex regex patterns with
    various protection mechanisms against malicious inputs.
    
    Features:
    - Simple string and regex pattern matching
    - Atomic file operations using temporary files
    - Automatic backup creation
    - Dry run mode for testing changes
    - Protection against large files and excessive replacements
    - Regex timeout protection
    - Detailed reporting of all changes made
    
    Example:
        >>> tool = FileSearchReplaceTool()
        >>> result = tool.execute(
        ...     file_path="/path/to/file.txt",
        ...     search_pattern="old_text",
        ...     replace_with="new_text"
        ... )
        >>> print(result)
        Success: 3 replacements made in /path/to/file.txt
        Lines modified: 5, 12, 18
        File size changed from 1024 to 1036 bytes
    """
    
    # Security and performance limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_REGEX_TIME = 5.0  # 5 seconds
    DEFAULT_MAX_REPLACEMENTS = 1000
    SUPPORTED_ENCODINGS = ['utf-8', 'utf-16', 'latin-1', 'ascii']
    
    def __init__(self) -> None:
        """Initialize the FileSearchReplaceTool with name and description."""
        super().__init__()
        self.name = "file_search_replace"
        self.description = (
            "Performs powerful search and replace operations on text files with regex support. "
            "Arguments: {"
            "'file_path': 'str', "
            "'search_pattern': 'str', "
            "'replace_with': 'str', "
            "'is_regex': 'bool (default: False)', "
            "'regex_flags': 'int (default: 0)', "
            "'auto_backup': 'bool (default: True)', "
            "'dry_run': 'bool (default: False)', "
            "'max_replacements': 'int (default: 1000)'"
            "}"
        )
    
    def execute(
        self, 
        file_path: str, 
        search_pattern: str, 
        replace_with: str,
        is_regex: bool = False, 
        regex_flags: int = 0,
        auto_backup: bool = True, 
        dry_run: bool = False,
        max_replacements: int = DEFAULT_MAX_REPLACEMENTS,
        **kwargs: Any
    ) -> str:
        """Execute search and replace operation on the specified file.
        
        Args:
            file_path: Path to the file to modify
            search_pattern: Pattern to search for (string or regex)
            replace_with: Text to replace matches with
            is_regex: Whether search_pattern is a regular expression
            regex_flags: Regex flags to use (e.g., re.IGNORECASE, re.MULTILINE)
            auto_backup: Whether to create a backup before modifying
            dry_run: Whether to simulate changes without actually modifying the file
            max_replacements: Maximum number of replacements to prevent runaway operations
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            Detailed string report of the operation results
        """
        try:
            # Validate arguments
            validation_error = self._validate_arguments(
                file_path, search_pattern, replace_with, is_regex, 
                regex_flags, max_replacements
            )
            if validation_error:
                return validation_error
            
            # Check file existence and permissions
            file_check_error = self._check_file_access(file_path, dry_run)
            if file_check_error:
                return file_check_error
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.MAX_FILE_SIZE:
                return f"Error: File size ({file_size:,} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE:,} bytes)"
            
            # Read file content with encoding detection
            content, encoding = self._read_file_with_encoding(file_path)
            if content is None:
                return f"Error: Could not read file '{file_path}' with any supported encoding"
            
            # Prepare regex pattern
            if is_regex:
                try:
                    pattern = re.compile(search_pattern, regex_flags)
                except re.error as e:
                    return f"Error: Invalid regex pattern '{search_pattern}': {str(e)}"
            else:
                # Escape special regex characters for literal string search
                escaped_pattern = re.escape(search_pattern)
                pattern = re.compile(escaped_pattern, regex_flags)
            
            # Perform search and replace with timeout protection
            try:
                modified_content, replacements_info = self._perform_replacement(
                    content, pattern, replace_with, max_replacements
                )
            except TimeoutError:
                return f"Error: Regex operation timed out after {self.MAX_REGEX_TIME} seconds"
            except Exception as e:
                return f"Error: Failed to perform replacement: {str(e)}"
            
            # Check if any replacements were made
            if not replacements_info['count']:
                return "Pattern not found: No matches found for the specified pattern"
            
            # Generate report
            report = self._generate_report(
                file_path, replacements_info, len(content), len(modified_content), dry_run
            )
            
            # If dry run, return report without making changes
            if dry_run:
                return f"DRY RUN - {report}"
            
            # Create backup if requested
            if auto_backup:
                backup_error = self._create_backup(file_path)
                if backup_error:
                    return backup_error
            
            # Write modified content atomically
            write_error = self._write_file_atomically(file_path, modified_content, encoding)
            if write_error:
                return write_error
            
            return f"Success: {report}"
            
        except Exception as e:
            return f"Error: Unexpected error during file search and replace: {str(e)}"
    
    def _validate_arguments(
        self, 
        file_path: str, 
        search_pattern: str, 
        replace_with: str,
        is_regex: bool, 
        regex_flags: int, 
        max_replacements: int
    ) -> str:
        """Validate all input arguments.
        
        Args:
            file_path: Path to the file
            search_pattern: Pattern to search for
            replace_with: Replacement text
            is_regex: Whether pattern is regex
            regex_flags: Regex flags
            max_replacements: Maximum replacements
        
        Returns:
            Error message if validation fails, empty string if success
        """
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: file_path must be a non-empty string"
        
        if not isinstance(search_pattern, str):
            return "Error: search_pattern must be a string"
        
        if not search_pattern:
            return "Error: search_pattern cannot be empty"
        
        if not isinstance(replace_with, str):
            return "Error: replace_with must be a string"
        
        if not isinstance(is_regex, bool):
            return "Error: is_regex must be a boolean"
        
        if not isinstance(regex_flags, int) or regex_flags < 0:
            return "Error: regex_flags must be a non-negative integer"
        
        if not isinstance(max_replacements, int) or max_replacements <= 0:
            return "Error: max_replacements must be a positive integer"
        
        # Validate regex flags are within reasonable bounds
        if regex_flags > 0b111111111:  # All current re flags combined
            return "Error: regex_flags value is too large"
        
        return ""
    
    def _check_file_access(self, file_path: str, dry_run: bool) -> str:
        """Check if file exists and has appropriate permissions.
        
        Args:
            file_path: Path to the file
            dry_run: Whether this is a dry run
        
        Returns:
            Error message if check fails, empty string if success
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found"
        
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a regular file"
        
        if not os.access(file_path, os.R_OK):
            return f"Error: No read permission for file '{file_path}'"
        
        if not dry_run and not os.access(file_path, os.W_OK):
            return f"Error: No write permission for file '{file_path}'"
        
        return ""
    
    def _read_file_with_encoding(self, file_path: str) -> Tuple[str, str]:
        """Read file content with automatic encoding detection.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Tuple of (content, encoding) or (None, None) if failed
        """
        for encoding in self.SUPPORTED_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        return None, None
    
    def _perform_replacement(
        self, 
        content: str, 
        pattern: re.Pattern, 
        replace_with: str, 
        max_replacements: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Perform the actual search and replace operation with timeout protection.
        
        Args:
            content: File content
            pattern: Compiled regex pattern
            replace_with: Replacement text
            max_replacements: Maximum number of replacements
        
        Returns:
            Tuple of (modified_content, replacement_info)
        
        Raises:
            TimeoutError: If operation takes too long
        """
        start_time = time.time()
        replacements_made = 0
        modified_lines = []
        
        # Split content into lines for line-by-line processing and reporting
        lines = content.splitlines(keepends=True)
        modified_lines_content = []
        
        for line_num, line in enumerate(lines, 1):
            # Check timeout
            if time.time() - start_time > self.MAX_REGEX_TIME:
                raise TimeoutError("Regex operation timed out")
            
            # Count matches in this line
            matches = list(pattern.finditer(line))
            
            if matches and replacements_made + len(matches) <= max_replacements:
                # Perform replacement on this line
                new_line = pattern.sub(replace_with, line)
                modified_lines_content.append(new_line)
                modified_lines.append(line_num)
                replacements_made += len(matches)
            else:
                modified_lines_content.append(line)
                
                # Check if we've hit the replacement limit
                if replacements_made >= max_replacements:
                    break
        
        modified_content = ''.join(modified_lines_content)
        
        replacement_info = {
            'count': replacements_made,
            'lines_modified': modified_lines,
            'max_reached': replacements_made >= max_replacements
        }
        
        return modified_content, replacement_info
    
    def _generate_report(
        self, 
        file_path: str, 
        replacements_info: Dict[str, Any], 
        original_size: int, 
        new_size: int,
        dry_run: bool
    ) -> str:
        """Generate detailed report of the operation.
        
        Args:
            file_path: Path to the file
            replacements_info: Information about replacements made
            original_size: Original content size in characters
            new_size: New content size in characters
            dry_run: Whether this was a dry run
        
        Returns:
            Formatted report string
        """
        count = replacements_info['count']
        lines_modified = replacements_info['lines_modified']
        max_reached = replacements_info['max_reached']
        
        report_parts = []
        
        # Basic replacement info
        replacement_text = "replacement" if count == 1 else "replacements"
        would_text = "would be made" if dry_run else "made"
        report_parts.append(f"{count} {replacement_text} {would_text} in {file_path}")
        
        # Lines modified info
        if lines_modified:
            if len(lines_modified) <= 10:
                lines_str = ", ".join(map(str, lines_modified))
            else:
                lines_str = f"{', '.join(map(str, lines_modified[:10]))}... (+{len(lines_modified) - 10} more)"
            report_parts.append(f"Lines modified: {lines_str}")
        
        # File size change
        size_change = new_size - original_size
        if size_change != 0:
            change_text = "would change" if dry_run else "changed"
            direction = "increased" if size_change > 0 else "decreased"
            report_parts.append(f"File size {change_text} from {original_size:,} to {new_size:,} characters ({direction} by {abs(size_change):,})")
        
        # Warning if max replacements reached
        if max_reached:
            report_parts.append(f"Warning: Maximum replacement limit ({self.DEFAULT_MAX_REPLACEMENTS}) reached, additional matches were not processed")
        
        return "\n".join(report_parts)
    
    def _create_backup(self, file_path: str) -> str:
        """Create a backup of the file before modification.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Error message if backup fails, empty string if success
        """
        try:
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            return ""
        except Exception as e:
            return f"Error: Failed to create backup '{backup_path}': {str(e)}"
    
    def _write_file_atomically(self, file_path: str, content: str, encoding: str) -> str:
        """Write content to file atomically using temporary file.
        
        Args:
            file_path: Path to the target file
            content: Content to write
            encoding: Encoding to use
        
        Returns:
            Error message if write fails, empty string if success
        """
        try:
            # Get directory of target file
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            
            # Create temporary file in the same directory
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding=encoding, 
                dir=file_dir,
                prefix=f".{file_name}_tmp_",
                suffix=".tmp",
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content)
            
            # Atomically replace original file with temporary file
            if os.name == 'nt':  # Windows
                # On Windows, we need to remove the target first
                shutil.move(temp_path, file_path)
            else:  # Unix-like systems
                os.replace(temp_path, file_path)
            
            return ""
            
        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return f"Error: Failed to write file '{file_path}': {str(e)}"