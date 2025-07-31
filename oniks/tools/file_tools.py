"""File operation tools for the ONIKS NeuralNet framework.

This module provides tools for performing file operations during graph execution.
These tools can be used by agents to read, write, and manipulate files as part
of their workflow execution.
"""

import os
from typing import Any

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