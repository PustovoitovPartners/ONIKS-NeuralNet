"""Shell command execution tools for the ONIKS NeuralNet framework.

This module provides tools for secure execution of shell commands during graph execution.
The tool ensures commands are executed safely within the active Python virtual environment
and provides structured output for LLM analysis.
"""

import shlex
import subprocess
import sys
from typing import Any, Optional

from oniks.tools.base import Tool


class ExecuteBashCommandTool(Tool):
    """Tool for secure execution of shell commands with structured output.
    
    This tool provides a safe way to execute shell commands during graph execution
    while ensuring all commands run within the active Python virtual environment.
    It uses subprocess.run() with shell=False for security and captures stdout,
    stderr, and return codes for comprehensive analysis.
    
    Security Features:
    - Uses subprocess.run() with shell=False to prevent shell injection attacks
    - Commands are parsed into argument lists using shlex for safe execution
    - Timeout protection prevents hanging processes
    - All commands execute within the active Python virtual environment
    
    The tool takes a 'command' string and optional 'timeout' integer. It returns
    structured output containing exit code, stdout, and stderr for easy LLM analysis.
    
    Example:
        >>> tool = ExecuteBashCommandTool()
        >>> result = tool.execute(command="python --version")
        >>> print(result)
        Exit Code: 0
        --- STDOUT ---
        Python 3.11.0
        --- STDERR ---
        
        
        >>> result = tool.execute(command="nonexistent_command")
        >>> print(result)
        Exit Code: 127
        --- STDOUT ---
        
        --- STDERR ---
        /bin/sh: nonexistent_command: command not found
    """
    
    def __init__(self) -> None:
        """Initialize the ExecuteBashCommandTool with name and description."""
        super().__init__()
        self.name = "execute_bash_command"
        self.description = (
            "Executes shell command and returns its stdout, stderr and exit code. "
            "Arguments: {'command': 'str', 'timeout': 'Optional[int]'}"
        )
    
    def execute(self, **kwargs: Any) -> str:
        """Execute a shell command safely and return structured output.
        
        Args:
            **kwargs: Must contain:
                - 'command': String command to execute
                - 'timeout': Optional integer timeout in seconds (default: 60)
        
        Returns:
            Structured string containing exit code, stdout, and stderr formatted
            for LLM analysis in the format:
            "Exit Code: [code]\n--- STDOUT ---\n[stdout]\n--- STDERR ---\n[stderr]"
        
        Raises:
            KeyError: If 'command' argument is not provided.
            TypeError: If arguments are not of the correct type.
        """
        # Validate required argument
        if 'command' not in kwargs:
            return self._format_error("Missing required argument 'command'")
        
        command = kwargs['command']
        timeout = kwargs.get('timeout', 60)
        
        # Validate argument types
        if not isinstance(command, str):
            return self._format_error(
                f"'command' must be a string, got {type(command).__name__}"
            )
        
        if timeout is not None and not isinstance(timeout, int):
            return self._format_error(
                f"'timeout' must be an integer or None, got {type(timeout).__name__}"
            )
        
        # Validate command is not empty
        if not command.strip():
            return self._format_error("'command' cannot be empty")
        
        # Validate timeout is positive
        if timeout is not None and timeout <= 0:
            return self._format_error("'timeout' must be a positive integer")
        
        try:
            # Parse command into argument list for security (shell=False)
            try:
                args = shlex.split(command)
            except ValueError as e:
                return self._format_error(f"Invalid command syntax: {str(e)}")
            
            if not args:
                return self._format_error("No command arguments provided")
            
            # Ensure we're running in the active Python virtual environment
            env = None
            if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
                # We're in a virtual environment, preserve it
                import os
                env = os.environ.copy()
                # Ensure PATH includes the virtual environment
                venv_bin = os.path.join(sys.prefix, 'bin')
                if 'PATH' in env:
                    env['PATH'] = f"{venv_bin}:{env['PATH']}"
                else:
                    env['PATH'] = venv_bin
            
            # Execute command with security measures
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,  # Security: prevent shell injection
                env=env,  # Preserve virtual environment
                cwd=None  # Use current working directory
            )
            
            # Format structured output for LLM analysis
            return self._format_output(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            return self._format_error(
                f"Command timed out after {timeout} seconds",
                exit_code=124  # Standard timeout exit code
            )
        except FileNotFoundError:
            return self._format_error(
                f"Command not found: {args[0]}",
                exit_code=127  # Standard "command not found" exit code
            )
        except PermissionError:
            return self._format_error(
                f"Permission denied when executing: {args[0]}",
                exit_code=126  # Standard "permission denied" exit code
            )
        except OSError as e:
            return self._format_error(
                f"System error when executing command: {str(e)}",
                exit_code=1
            )
        except Exception as e:
            return self._format_error(
                f"Unexpected error when executing command: {str(e)}",
                exit_code=1
            )
    
    def _format_output(self, exit_code: int, stdout: str, stderr: str) -> str:
        """Format command output in structured format for LLM analysis.
        
        Args:
            exit_code: Command exit/return code
            stdout: Standard output from command
            stderr: Standard error from command
        
        Returns:
            Formatted string with structured output
        """
        return (
            f"Exit Code: {exit_code}\n"
            f"--- STDOUT ---\n"
            f"{stdout}\n"
            f"--- STDERR ---\n"
            f"{stderr}"
        )
    
    def _format_error(self, error_message: str, exit_code: Optional[int] = None) -> str:
        """Format error message in structured format.
        
        Args:
            error_message: Error description
            exit_code: Optional exit code (defaults to None for tool errors)
        
        Returns:
            Formatted error message in structured output format
        """
        if exit_code is not None:
            return self._format_output(
                exit_code=exit_code,
                stdout="",
                stderr=f"Tool Error: {error_message}"
            )
        else:
            return f"Tool Error: {error_message}"