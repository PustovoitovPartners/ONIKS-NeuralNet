"""File system operation tools for the ONIKS NeuralNet framework.

This module provides tools for file system operations during graph execution.
These tools can be used by agents to explore and analyze directory structures
as part of their workflow execution.
"""

import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

from oniks.tools.base import Tool


class ListFilesTool(Tool):
    """Tool for recursively listing files and directories in a tree-like format.
    
    This tool provides a comprehensive view of directory structures by recursively
    traversing all subdirectories and files, displaying them in a readable tree format.
    It includes built-in filtering capabilities to ignore common directories like
    version control, cache, and virtual environment folders.
    
    The tool takes a 'path' argument specifying the root directory to explore,
    and an optional 'ignore_patterns' list for custom filtering patterns.
    It returns a formatted string representation of the directory tree structure.
    
    Example:
        >>> tool = ListFilesTool()
        >>> result = tool.execute(path="/my/project")
        >>> print(result)
        /my/project/
        ├── src/
        │   ├── main.py
        │   └── utils.py
        ├── tests/
        │   └── test_main.py
        └── README.md
        
        >>> result = tool.execute(path="/my/project", ignore_patterns=["*.pyc", "logs"])
        >>> print(result)
        Directory tree with custom ignore patterns...
    """
    
    def __init__(self) -> None:
        """Initialize the ListFilesTool with name and description."""
        super().__init__()
        self.name = "list_files"
        self.description = (
            "Recursively lists all files and directories starting from the specified path. "
            "Arguments: {'path': 'str', 'ignore_patterns': 'Optional[List[str]]'}"
        )
        
        # Default patterns to ignore - common directories and files that are usually not relevant
        self._default_ignore_patterns = [
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            "env",
            ".env",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "htmlcov",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".idea",
            ".vscode",
            "*.log",
            ".logs"
        ]
    
    def execute(self, **kwargs: Any) -> str:
        """Recursively list files and directories in tree format.
        
        Args:
            **kwargs: Must contain 'path' key with string value pointing to the
                     root directory to explore. Optionally contains 'ignore_patterns'
                     key with a list of patterns to ignore in addition to defaults.
        
        Returns:
            String containing the formatted directory tree, or an error message
            if the directory cannot be accessed.
        
        Raises:
            KeyError: If 'path' argument is not provided.
            TypeError: If arguments have incorrect types.
        """
        # Validate required argument
        if 'path' not in kwargs:
            return "Error: Missing required argument 'path'"
        
        root_path = kwargs['path']
        
        # Validate argument type
        if not isinstance(root_path, str):
            return f"Error: 'path' must be a string, got {type(root_path).__name__}"
        
        # Validate path is not empty
        if not root_path.strip():
            return "Error: 'path' cannot be empty"
        
        # Handle optional ignore_patterns argument
        ignore_patterns = kwargs.get('ignore_patterns', [])
        if ignore_patterns is not None:
            if not isinstance(ignore_patterns, list):
                return f"Error: 'ignore_patterns' must be a list, got {type(ignore_patterns).__name__}"
            
            # Validate all patterns are strings
            for pattern in ignore_patterns:
                if not isinstance(pattern, str):
                    return f"Error: All ignore patterns must be strings, got {type(pattern).__name__}"
        else:
            ignore_patterns = []
        
        # Combine default and custom ignore patterns
        all_ignore_patterns = self._default_ignore_patterns + ignore_patterns
        
        try:
            # Convert to Path object for easier manipulation
            root_path_obj = Path(root_path).resolve()
            
            # Check if path exists
            if not root_path_obj.exists():
                return f"Error: Path '{root_path}' not found"
            
            # Check if path is a directory
            if not root_path_obj.is_dir():
                return f"Error: '{root_path}' is not a directory"
            
            # Check if directory is readable
            if not os.access(root_path_obj, os.R_OK):
                return f"Error: No read permission for directory '{root_path}'"
            
            # Generate the tree structure
            tree_lines = []
            tree_lines.append(f"{root_path_obj}/")
            
            self._build_tree(root_path_obj, "", tree_lines, all_ignore_patterns, True)
            
            return "\n".join(tree_lines)
            
        except PermissionError:
            return f"Error: Permission denied when accessing directory '{root_path}'"
        except OSError as e:
            return f"Error: System error when accessing directory '{root_path}': {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when listing directory '{root_path}': {str(e)}"
    
    def _build_tree(
        self, 
        directory: Path, 
        prefix: str, 
        tree_lines: List[str], 
        ignore_patterns: List[str],
        is_root: bool = False
    ) -> None:
        """Recursively build the tree structure for a directory.
        
        Args:
            directory: Path object representing the directory to process.
            prefix: String prefix for the current level of indentation.
            tree_lines: List to append formatted tree lines to.
            ignore_patterns: List of patterns to ignore during traversal.
            is_root: Whether this is the root directory (affects prefix handling).
        """
        try:
            # Get all items in the directory and sort them
            items = []
            for item in directory.iterdir():
                if not self._should_ignore(item, ignore_patterns):
                    items.append(item)
            
            # Sort items: directories first, then files, both alphabetically
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            # Process each item
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                
                # Determine the connector symbol
                if is_last:
                    connector = "└── "
                    new_prefix = prefix + "    "
                else:
                    connector = "├── "
                    new_prefix = prefix + "│   "
                
                # Format the item name
                if item.is_dir():
                    item_name = f"{item.name}/"
                else:
                    item_name = item.name
                
                # Add the formatted line
                tree_lines.append(f"{prefix}{connector}{item_name}")
                
                # Recursively process subdirectories
                if item.is_dir():
                    try:
                        self._build_tree(item, new_prefix, tree_lines, ignore_patterns)
                    except PermissionError:
                        # Add a note about permission denied subdirectories
                        tree_lines.append(f"{new_prefix}[Permission Denied]")
                    except OSError:
                        # Add a note about inaccessible subdirectories
                        tree_lines.append(f"{new_prefix}[Access Error]")
                        
        except PermissionError:
            # This shouldn't happen as we check permissions before calling
            tree_lines.append(f"{prefix}[Permission Denied]")
        except OSError:
            # Handle other OS-level errors
            tree_lines.append(f"{prefix}[Access Error]")
    
    def _should_ignore(self, path: Path, ignore_patterns: List[str]) -> bool:
        """Check if a path should be ignored based on the ignore patterns.
        
        Args:
            path: Path object to check.
            ignore_patterns: List of patterns to match against.
        
        Returns:
            True if the path should be ignored, False otherwise.
        """
        import fnmatch
        
        path_name = path.name
        
        # Check against all ignore patterns
        for pattern in ignore_patterns:
            # Support both exact matches and glob patterns
            if fnmatch.fnmatch(path_name, pattern) or path_name == pattern:
                return True
        
        return False


class WriteFileTool(Tool):
    """Tool for writing or overwriting content to a file.
    
    This tool provides the ability to write content to files, creating the file
    if it doesn't exist and creating parent directories as needed. It handles
    various error conditions gracefully and provides detailed feedback about
    the operation results.
    
    The tool takes 'file_path' and 'content' arguments and returns a success
    message with the number of bytes written, or an error message if the
    operation fails.
    
    Example:
        >>> tool = WriteFileTool()
        >>> result = tool.execute(file_path="/my/project/config.txt", content="key=value")
        >>> print(result)
        Successfully wrote 9 bytes to /my/project/config.txt
        
        >>> result = tool.execute(file_path="/invalid/path", content="test")
        >>> print(result)
        Error: Permission denied when writing to file '/invalid/path'
    """
    
    def __init__(self) -> None:
        """Initialize the WriteFileTool with name and description."""
        super().__init__()
        self.name = "write_file"
        self.description = (
            "Writes or overwrites content to the specified file. Creates file if it doesn't exist. "
            "Arguments: {'file_path': 'str', 'content': 'str'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Write or overwrite content to the specified file.
        
        Args:
            **kwargs: Must contain 'file_path' key with string value pointing to the
                     target file path, and 'content' key with string content to write.
        
        Returns:
            String containing success message with byte count and file path,
            or an error message if the operation fails.
        
        Raises:
            KeyError: If required arguments are not provided.
            TypeError: If arguments have incorrect types.
        """
        # Validate required arguments
        if 'file_path' not in kwargs:
            return "Error: Missing required argument 'file_path'"
        
        if 'content' not in kwargs:
            return "Error: Missing required argument 'content'"
        
        file_path = kwargs['file_path']
        content = kwargs['content']
        
        # Validate argument types
        if not isinstance(file_path, str):
            return f"Error: 'file_path' must be a string, got {type(file_path).__name__}"
        
        if not isinstance(content, str):
            return f"Error: 'content' must be a string, got {type(content).__name__}"
        
        # Validate file_path is not empty
        if not file_path.strip():
            return "Error: 'file_path' cannot be empty"
        
        try:
            # Convert to Path object for easier manipulation
            file_path_obj = Path(file_path).resolve()
            
            # Create parent directories if they don't exist
            parent_dir = file_path_obj.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return f"Error: Permission denied when creating parent directories for '{file_path}'"
                except OSError as e:
                    return f"Error: Cannot create parent directories for '{file_path}': {str(e)}"
            
            # Check if parent directory is writable
            if not os.access(parent_dir, os.W_OK):
                return f"Error: No write permission for directory '{parent_dir}'"
            
            # Write content to file
            try:
                with open(file_path_obj, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Calculate the number of bytes written
                content_bytes = len(content.encode('utf-8'))
                
                return f"Successfully wrote {content_bytes} bytes to {file_path_obj}"
                
            except PermissionError:
                return f"Error: Permission denied when writing to file '{file_path}'"
            except OSError as e:
                return f"Error: Cannot write to file '{file_path}': {str(e)}"
            except UnicodeEncodeError as e:
                return f"Error: Cannot encode content for file '{file_path}': {str(e)}"
                
        except OSError as e:
            return f"Error: Invalid file path '{file_path}': {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when writing to file '{file_path}': {str(e)}"


class CreateDirectoryTool(Tool):
    """Tool for creating directories with support for nested directory creation.
    
    This tool provides the ability to create directories, including all necessary
    parent directories if they don't exist. It handles various error conditions
    gracefully and provides detailed feedback about the operation results.
    
    The tool takes a 'path' argument specifying the directory to create and uses
    the parents=True and exist_ok=True options to handle nested directory creation
    and avoid errors when the directory already exists.
    
    Example:
        >>> tool = CreateDirectoryTool()
        >>> result = tool.execute(path="/my/project/nested/directory")
        >>> print(result)
        Directory '/my/project/nested/directory' created successfully.
        
        >>> result = tool.execute(path="/invalid/path")
        >>> print(result)
        Error: Permission denied when creating directory '/invalid/path'
    """
    
    def __init__(self) -> None:
        """Initialize the CreateDirectoryTool with name and description."""
        super().__init__()
        self.name = "create_directory"
        self.description = (
            "Creates a new directory at the specified path. "
            "Arguments: {'path': 'str'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Create a directory at the specified path with nested directory support.
        
        Args:
            **kwargs: Must contain 'path' key with string value pointing to the
                     directory path to create.
        
        Returns:
            String containing success message with the created directory path,
            or an error message if the operation fails.
        
        Raises:
            KeyError: If 'path' argument is not provided.
            TypeError: If arguments have incorrect types.
        """
        # Validate required argument
        if 'path' not in kwargs:
            return "Error: Missing required argument 'path'"
        
        directory_path = kwargs['path']
        
        # Validate argument type
        if not isinstance(directory_path, str):
            return f"Error: 'path' must be a string, got {type(directory_path).__name__}"
        
        # Validate path is not empty
        if not directory_path.strip():
            return "Error: 'path' cannot be empty"
        
        try:
            # Convert to Path object for easier manipulation
            path_obj = Path(directory_path).resolve()
            
            # Create the directory with parents=True and exist_ok=True
            path_obj.mkdir(parents=True, exist_ok=True)
            
            return f"Directory '{path_obj}' created successfully."
            
        except PermissionError:
            return f"Error: Permission denied when creating directory '{directory_path}'"
        except FileExistsError:
            # This shouldn't happen with exist_ok=True, but included for completeness
            return f"Error: A file already exists at path '{directory_path}'"
        except OSError as e:
            return f"Error: Cannot create directory '{directory_path}': {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when creating directory '{directory_path}': {str(e)}"


class CopyFileTool(Tool):
    """Tool for copying files from a source path to a destination path.
    
    This tool provides the ability to copy files from one location to another,
    creating parent directories for the destination if they don't exist. It handles
    various error conditions gracefully and provides detailed feedback about
    the operation results.
    
    The tool takes 'source_path' and 'destination_path' arguments and returns a
    success message indicating the copy operation, or an error message if the
    operation fails.
    
    Example:
        >>> tool = CopyFileTool()
        >>> result = tool.execute(source_path="/my/source.txt", destination_path="/my/backup/source.txt")
        >>> print(result)
        Successfully copied '/my/source.txt' to '/my/backup/source.txt'
        
        >>> result = tool.execute(source_path="/nonexistent.txt", destination_path="/backup.txt")
        >>> print(result)
        Error: Source file '/nonexistent.txt' not found
    """
    
    def __init__(self) -> None:
        """Initialize the CopyFileTool with name and description."""
        super().__init__()
        self.name = "copy_file"
        self.description = (
            "Copies a file from a source path to a destination path. "
            "Arguments: {'source_path': 'str', 'destination_path': 'str'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Copy a file from source path to destination path.
        
        Args:
            **kwargs: Must contain 'source_path' key with string value pointing to the
                     source file path, and 'destination_path' key with string value
                     pointing to the destination file path.
        
        Returns:
            String containing success message with source and destination paths,
            or an error message if the operation fails.
        
        Raises:
            KeyError: If required arguments are not provided.
            TypeError: If arguments have incorrect types.
        """
        # Validate required arguments
        if 'source_path' not in kwargs:
            return "Error: Missing required argument 'source_path'"
        
        if 'destination_path' not in kwargs:
            return "Error: Missing required argument 'destination_path'"
        
        source_path = kwargs['source_path']
        destination_path = kwargs['destination_path']
        
        # Validate argument types
        if not isinstance(source_path, str):
            return f"Error: 'source_path' must be a string, got {type(source_path).__name__}"
        
        if not isinstance(destination_path, str):
            return f"Error: 'destination_path' must be a string, got {type(destination_path).__name__}"
        
        # Validate paths are not empty
        if not source_path.strip():
            return "Error: 'source_path' cannot be empty"
        
        if not destination_path.strip():
            return "Error: 'destination_path' cannot be empty"
        
        try:
            # Convert to Path objects for easier manipulation
            source_path_obj = Path(source_path).resolve()
            destination_path_obj = Path(destination_path).resolve()
            
            # Check if source file exists
            if not source_path_obj.exists():
                return f"Error: Source file '{source_path}' not found"
            
            # Check if source is a file
            if not source_path_obj.is_file():
                return f"Error: Source path '{source_path}' is not a file"
            
            # Check if source file is readable
            if not os.access(source_path_obj, os.R_OK):
                return f"Error: No read permission for source file '{source_path}'"
            
            # Create parent directories for destination if they don't exist
            destination_parent = destination_path_obj.parent
            if not destination_parent.exists():
                try:
                    destination_parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return f"Error: Permission denied when creating parent directories for destination '{destination_path}'"
                except OSError as e:
                    return f"Error: Cannot create parent directories for destination '{destination_path}': {str(e)}"
            
            # Check if destination parent directory is writable
            if not os.access(destination_parent, os.W_OK):
                return f"Error: No write permission for destination directory '{destination_parent}'"
            
            # Check if destination already exists and is not a file
            if destination_path_obj.exists() and not destination_path_obj.is_file():
                return f"Error: Destination path '{destination_path}' exists but is not a file"
            
            # Perform the copy operation
            try:
                shutil.copy(source_path_obj, destination_path_obj)
                return f"Successfully copied '{source_path_obj}' to '{destination_path_obj}'"
                
            except PermissionError:
                return f"Error: Permission denied when copying to destination '{destination_path}'"
            except OSError as e:
                return f"Error: Cannot copy file: {str(e)}"
            except shutil.SameFileError:
                return f"Error: Source and destination refer to the same file"
                
        except OSError as e:
            return f"Error: Invalid file path: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when copying file: {str(e)}"


class RenameFileTool(Tool):
    """Tool for renaming or moving files from a source path to a destination path.
    
    This tool provides the ability to rename files or move them to different locations,
    creating parent directories for the destination if they don't exist. It handles
    various error conditions gracefully and provides detailed feedback about
    the operation results.
    
    The tool takes 'source_path' and 'destination_path' arguments and returns a
    success message indicating the rename/move operation, or an error message if the
    operation fails.
    
    Example:
        >>> tool = RenameFileTool()
        >>> result = tool.execute(source_path="/my/old_name.txt", destination_path="/my/new_name.txt")
        >>> print(result)
        Successfully renamed '/my/old_name.txt' to '/my/new_name.txt'
        
        >>> result = tool.execute(source_path="/nonexistent.txt", destination_path="/new.txt")
        >>> print(result)
        Error: Source file '/nonexistent.txt' not found
    """
    
    def __init__(self) -> None:
        """Initialize the RenameFileTool with name and description."""
        super().__init__()
        self.name = "rename_file"
        self.description = (
            "Renames or moves a file from a source path to a destination path. "
            "Arguments: {'source_path': 'str', 'destination_path': 'str'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Rename or move a file from source path to destination path.
        
        Args:
            **kwargs: Must contain 'source_path' key with string value pointing to the
                     source file path, and 'destination_path' key with string value
                     pointing to the destination file path.
        
        Returns:
            String containing success message with source and destination paths,
            or an error message if the operation fails.
        
        Raises:
            KeyError: If required arguments are not provided.
            TypeError: If arguments have incorrect types.
        """
        # Validate required arguments
        if 'source_path' not in kwargs:
            return "Error: Missing required argument 'source_path'"
        
        if 'destination_path' not in kwargs:
            return "Error: Missing required argument 'destination_path'"
        
        source_path = kwargs['source_path']
        destination_path = kwargs['destination_path']
        
        # Validate argument types
        if not isinstance(source_path, str):
            return f"Error: 'source_path' must be a string, got {type(source_path).__name__}"
        
        if not isinstance(destination_path, str):
            return f"Error: 'destination_path' must be a string, got {type(destination_path).__name__}"
        
        # Validate paths are not empty
        if not source_path.strip():
            return "Error: 'source_path' cannot be empty"
        
        if not destination_path.strip():
            return "Error: 'destination_path' cannot be empty"
        
        try:
            # Convert to Path objects for easier manipulation
            source_path_obj = Path(source_path).resolve()
            destination_path_obj = Path(destination_path).resolve()
            
            # Check if source file exists
            if not source_path_obj.exists():
                return f"Error: Source file '{source_path}' not found"
            
            # Check if source is a file
            if not source_path_obj.is_file():
                return f"Error: Source path '{source_path}' is not a file"
            
            # Check if source file is readable and writable (needed for rename)
            if not os.access(source_path_obj, os.R_OK):
                return f"Error: No read permission for source file '{source_path}'"
            
            if not os.access(source_path_obj.parent, os.W_OK):
                return f"Error: No write permission for source directory '{source_path_obj.parent}'"
            
            # Create parent directories for destination if they don't exist
            destination_parent = destination_path_obj.parent
            if not destination_parent.exists():
                try:
                    destination_parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return f"Error: Permission denied when creating parent directories for destination '{destination_path}'"
                except OSError as e:
                    return f"Error: Cannot create parent directories for destination '{destination_path}': {str(e)}"
            
            # Check if destination parent directory is writable
            if not os.access(destination_parent, os.W_OK):
                return f"Error: No write permission for destination directory '{destination_parent}'"
            
            # Check if trying to rename to the same path (check this first)
            if source_path_obj == destination_path_obj:
                return f"Error: Source and destination paths are the same"
            
            # Check if destination already exists
            if destination_path_obj.exists():
                return f"Error: Destination path '{destination_path}' already exists"
            
            # Perform the rename/move operation
            try:
                source_path_obj.rename(destination_path_obj)
                return f"Successfully renamed '{source_path_obj}' to '{destination_path_obj}'"
                
            except PermissionError:
                return f"Error: Permission denied when renaming to destination '{destination_path}'"
            except OSError as e:
                return f"Error: Cannot rename file: {str(e)}"
                
        except OSError as e:
            return f"Error: Invalid file path: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error when renaming file: {str(e)}"