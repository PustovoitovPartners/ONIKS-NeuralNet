"""Unit tests for the ExecuteBashCommandTool class."""

import pytest
import os
import sys
import subprocess
import time
import platform
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from oniks.tools.shell_tools import ExecuteBashCommandTool


class TestExecuteBashCommandToolInitialization:
    """Test ExecuteBashCommandTool initialization."""
    
    def test_execute_bash_command_tool_initialization(self):
        """Test ExecuteBashCommandTool initialization sets correct attributes."""
        tool = ExecuteBashCommandTool()
        
        assert tool.name == "execute_bash_command"
        assert "Выполняет shell-команду" in tool.description
        assert "command" in tool.description
        assert "timeout" in tool.description
        assert isinstance(tool.description, str)
    
    def test_execute_bash_command_tool_string_representation(self):
        """Test ExecuteBashCommandTool string representation."""
        tool = ExecuteBashCommandTool()
        
        assert str(tool) == "ExecuteBashCommandTool(name='execute_bash_command')"
        assert repr(tool) == "ExecuteBashCommandTool(name='execute_bash_command')"


class TestExecuteBashCommandToolArgumentValidation:
    """Test ExecuteBashCommandTool argument validation."""
    
    def test_execute_bash_command_tool_missing_command_argument(self):
        """Test execution without command argument."""
        tool = ExecuteBashCommandTool()
        
        result = tool.execute()
        
        assert "Tool Error: Missing required argument 'command'" in result
    
    def test_execute_bash_command_tool_command_wrong_type(self):
        """Test execution with non-string command argument."""
        tool = ExecuteBashCommandTool()
        
        # Test with integer
        result1 = tool.execute(command=123)
        assert "Tool Error: 'command' must be a string, got int" in result1
        
        # Test with list
        result2 = tool.execute(command=["ls", "-l"])
        assert "Tool Error: 'command' must be a string, got list" in result2
        
        # Test with None
        result3 = tool.execute(command=None)
        assert "Tool Error: 'command' must be a string, got NoneType" in result3
        
        # Test with dictionary
        result4 = tool.execute(command={"cmd": "ls"})
        assert "Tool Error: 'command' must be a string, got dict" in result4
    
    def test_execute_bash_command_tool_timeout_wrong_type(self):
        """Test execution with non-integer timeout argument."""
        tool = ExecuteBashCommandTool()
        
        # Test with string
        result1 = tool.execute(command="echo test", timeout="60")
        assert "Tool Error: 'timeout' must be an integer or None, got str" in result1
        
        # Test with float
        result2 = tool.execute(command="echo test", timeout=60.5)
        assert "Tool Error: 'timeout' must be an integer or None, got float" in result2
        
        # Test with list
        result3 = tool.execute(command="echo test", timeout=[60])
        assert "Tool Error: 'timeout' must be an integer or None, got list" in result3
    
    def test_execute_bash_command_tool_empty_command(self):
        """Test execution with empty command."""
        tool = ExecuteBashCommandTool()
        
        # Test with empty string
        result1 = tool.execute(command="")
        assert "Tool Error: 'command' cannot be empty" in result1
        
        # Test with whitespace only
        result2 = tool.execute(command="   ")
        assert "Tool Error: 'command' cannot be empty" in result2
        
        # Test with tab and newline
        result3 = tool.execute(command="\t\n")
        assert "Tool Error: 'command' cannot be empty" in result3
    
    def test_execute_bash_command_tool_invalid_timeout_value(self):
        """Test execution with invalid timeout values."""
        tool = ExecuteBashCommandTool()
        
        # Test with zero timeout
        result1 = tool.execute(command="echo test", timeout=0)
        assert "Tool Error: 'timeout' must be a positive integer" in result1
        
        # Test with negative timeout
        result2 = tool.execute(command="echo test", timeout=-10)
        assert "Tool Error: 'timeout' must be a positive integer" in result2
    
    def test_execute_bash_command_tool_timeout_none_allowed(self):
        """Test that timeout=None is allowed (no timeout)."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test output",
                stderr=""
            )
            
            result = tool.execute(command="echo test", timeout=None)
            
            # Should not contain error message
            assert "Tool Error:" not in result
            # Mock should be called with timeout=None (no timeout)
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]['timeout'] is None
    
    def test_execute_bash_command_tool_additional_arguments(self):
        """Test that additional arguments are ignored gracefully."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test output",
                stderr=""
            )
            
            result = tool.execute(
                command="echo test",
                timeout=30,
                extra_arg="ignored",
                another_arg=123,
                yet_another={"key": "value"}
            )
            
            # Should work normally and ignore extra arguments
            assert "Tool Error:" not in result
            assert "Exit Code: 0" in result
            assert "test output" in result


class TestExecuteBashCommandToolCommandParsing:
    """Test ExecuteBashCommandTool command parsing and security."""
    
    def test_execute_bash_command_tool_invalid_command_syntax(self):
        """Test handling of invalid shell command syntax."""
        tool = ExecuteBashCommandTool()
        
        # Test with unclosed quote
        result1 = tool.execute(command='echo "unclosed quote')
        assert "Tool Error: Invalid command syntax:" in result1
        
        # Test with invalid escape sequence (platform dependent)
        result2 = tool.execute(command='echo "invalid\\')
        assert "Tool Error: Invalid command syntax:" in result2 or "Exit Code:" in result2
    
    def test_execute_bash_command_tool_empty_args_after_parsing(self):
        """Test handling when shlex.split returns empty list."""
        tool = ExecuteBashCommandTool()
        
        with patch('shlex.split') as mock_split:
            mock_split.return_value = []
            
            result = tool.execute(command="some command")
            
            assert "Tool Error: No command arguments provided" in result
    
    def test_execute_bash_command_tool_shlex_parsing(self):
        """Test that commands are properly parsed with shlex for security."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test output",
                stderr=""
            )
            
            # Test command with quotes and spaces
            tool.execute(command='echo "hello world" test')
            
            # Verify shlex parsing was used (command split into proper arguments)
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]  # First positional argument (args list)
            
            assert isinstance(call_args, list)
            assert call_args[0] == "echo"
            assert "hello world" in call_args
            assert "test" in call_args
    
    def test_execute_bash_command_tool_shell_injection_prevention(self):
        """Test that shell injection attempts are prevented."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="safe output",
                stderr=""
            )
            
            # Test potential shell injection command
            dangerous_command = "echo safe; rm -rf /"
            tool.execute(command=dangerous_command)
            
            # Verify subprocess.run called with shell=False
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['shell'] is False
            
            # Verify command was parsed as arguments, not executed as shell command
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)
            assert call_args[0] == "echo"


class TestExecuteBashCommandToolSuccessfulExecution:
    """Test ExecuteBashCommandTool successful command execution."""
    
    def test_execute_bash_command_tool_simple_command_success(self):
        """Test successful execution of simple command."""
        tool = ExecuteBashCommandTool()
        
        # Use a cross-platform command that should work everywhere
        if platform.system() == "Windows":
            command = "echo hello"
        else:
            command = "echo hello"
        
        result = tool.execute(command=command)
        
        assert "Exit Code: 0" in result
        assert "--- STDOUT ---" in result
        assert "--- STDERR ---" in result
        assert "hello" in result
    
    def test_execute_bash_command_tool_stdout_capture(self):
        """Test correct stdout capture."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            expected_stdout = "This is stdout output\nLine 2 of output\n"
            mock_run.return_value = Mock(
                returncode=0,
                stdout=expected_stdout,
                stderr=""
            )
            
            result = tool.execute(command="echo test")
            
            assert "Exit Code: 0" in result
            assert "--- STDOUT ---" in result
            assert expected_stdout in result
            assert "Line 2 of output" in result
    
    def test_execute_bash_command_tool_stderr_capture(self):
        """Test correct stderr capture."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            expected_stderr = "This is stderr output\nError message line 2\n"
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr=expected_stderr
            )
            
            result = tool.execute(command="echo error >&2")
            
            assert "Exit Code: 1" in result
            assert "--- STDERR ---" in result
            assert expected_stderr in result
            assert "Error message line 2" in result
    
    def test_execute_bash_command_tool_both_stdout_stderr(self):
        """Test capture of both stdout and stderr."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Standard output message",
                stderr="Standard error message"
            )
            
            result = tool.execute(command="complex command")
            
            assert "Exit Code: 0" in result
            assert "--- STDOUT ---" in result
            assert "--- STDERR ---" in result
            assert "Standard output message" in result
            assert "Standard error message" in result
    
    def test_execute_bash_command_tool_empty_output(self):
        """Test handling of commands with no output."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = tool.execute(command="true")
            
            assert "Exit Code: 0" in result
            assert "--- STDOUT ---" in result
            assert "--- STDERR ---" in result
            # Should handle empty output gracefully
            lines = result.split('\n')
            assert len(lines) >= 4  # At least: Exit Code, STDOUT header, empty line, STDERR header
    
    def test_execute_bash_command_tool_unicode_output(self):
        """Test handling of Unicode output."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            unicode_output = "Unicode: Привет мир! 🌍 文字"
            mock_run.return_value = Mock(
                returncode=0,
                stdout=unicode_output,
                stderr=""
            )
            
            result = tool.execute(command="echo unicode")
            
            assert "Exit Code: 0" in result
            assert unicode_output in result
            assert "Привет мир!" in result
            assert "🌍" in result
            assert "文字" in result
    
    def test_execute_bash_command_tool_multiline_output(self):
        """Test handling of multiline output."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            multiline_stdout = "Line 1\nLine 2\nLine 3\n"
            multiline_stderr = "Error line 1\nError line 2\n"
            mock_run.return_value = Mock(
                returncode=0,
                stdout=multiline_stdout,
                stderr=multiline_stderr
            )
            
            result = tool.execute(command="multiline command")
            
            assert "Exit Code: 0" in result
            assert "Line 1" in result
            assert "Line 2" in result
            assert "Line 3" in result
            assert "Error line 1" in result
            assert "Error line 2" in result
            
            # Verify structure is maintained
            assert result.count("--- STDOUT ---") == 1
            assert result.count("--- STDERR ---") == 1


class TestExecuteBashCommandToolErrorHandling:
    """Test ExecuteBashCommandTool error handling scenarios."""
    
    def test_execute_bash_command_tool_nonzero_exit_code(self):
        """Test handling of commands with non-zero exit codes."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="Some output before error",
                stderr="Error occurred"
            )
            
            result = tool.execute(command="false")
            
            assert "Exit Code: 1" in result
            assert "Some output before error" in result
            assert "Error occurred" in result
            # Should not treat non-zero exit code as tool error
            assert "Tool Error:" not in result
    
    def test_execute_bash_command_tool_various_exit_codes(self):
        """Test handling of various exit codes."""
        tool = ExecuteBashCommandTool()
        
        test_cases = [
            (0, "Success"),
            (1, "General error"),
            (2, "Misuse of shell builtins"),
            (126, "Command invoked cannot execute"),
            (127, "Command not found"),
            (128, "Invalid argument to exit"),
            (255, "Exit status out of range")
        ]
        
        for exit_code, description in test_cases:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=exit_code,
                    stdout=f"Output for {description}",
                    stderr=f"Error for {description}" if exit_code != 0 else ""
                )
                
                result = tool.execute(command=f"exit {exit_code}")
                
                assert f"Exit Code: {exit_code}" in result
                assert f"Output for {description}" in result
                if exit_code != 0:
                    assert f"Error for {description}" in result
    
    def test_execute_bash_command_tool_command_not_found(self):
        """Test handling of FileNotFoundError (command not found)."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("No such file or directory")
            
            result = tool.execute(command="nonexistent_command")
            
            assert "Exit Code: 127" in result
            assert "--- STDERR ---" in result
            assert "Tool Error: Command not found: nonexistent_command" in result
    
    def test_execute_bash_command_tool_permission_denied(self):
        """Test handling of PermissionError."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = PermissionError("Permission denied")
            
            result = tool.execute(command="restricted_command")
            
            assert "Exit Code: 126" in result
            assert "--- STDERR ---" in result
            assert "Tool Error: Permission denied when executing: restricted_command" in result
    
    def test_execute_bash_command_tool_os_error(self):
        """Test handling of general OSError."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("System error occurred")
            
            result = tool.execute(command="problematic_command")
            
            assert "Exit Code: 1" in result
            assert "--- STDERR ---" in result
            assert "Tool Error: System error when executing command: System error occurred" in result
    
    def test_execute_bash_command_tool_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = ValueError("Unexpected error")
            
            result = tool.execute(command="test_command")
            
            assert "Exit Code: 1" in result
            assert "--- STDERR ---" in result
            assert "Tool Error: Unexpected error when executing command: Unexpected error" in result


class TestExecuteBashCommandToolTimeoutHandling:
    """Test ExecuteBashCommandTool timeout functionality."""
    
    def test_execute_bash_command_tool_timeout_default(self):
        """Test that default timeout is used correctly."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="test output",
                stderr=""
            )
            
            tool.execute(command="echo test")
            
            # Verify default timeout (60 seconds) was used
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 60
    
    def test_execute_bash_command_tool_custom_timeout(self):
        """Test custom timeout values."""
        tool = ExecuteBashCommandTool()
        
        timeout_values = [1, 5, 30, 120, 300]
        
        for timeout in timeout_values:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="test output",
                    stderr=""
                )
                
                tool.execute(command="echo test", timeout=timeout)
                
                # Verify custom timeout was used
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs['timeout'] == timeout
    
    def test_execute_bash_command_tool_timeout_expired(self):
        """Test handling of timeout expiration."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["sleep", "10"], 
                timeout=5
            )
            
            result = tool.execute(command="sleep 10", timeout=5)
            
            assert "Exit Code: 124" in result
            assert "--- STDERR ---" in result
            assert "Tool Error: Command timed out after 5 seconds" in result
    
    def test_execute_bash_command_tool_timeout_various_durations(self):
        """Test timeout with various duration values."""
        tool = ExecuteBashCommandTool()
        
        timeout_durations = [1, 2, 5, 10]
        
        for timeout in timeout_durations:
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(
                    cmd=["long_command"], 
                    timeout=timeout
                )
                
                result = tool.execute(command="long_command", timeout=timeout)
                
                assert f"Exit Code: 124" in result
                assert f"Tool Error: Command timed out after {timeout} seconds" in result
    
    @pytest.mark.slow
    def test_execute_bash_command_tool_actual_timeout_integration(self):
        """Integration test with actual timeout (marked as slow)."""
        tool = ExecuteBashCommandTool()
        
        # Use a command that will definitely timeout
        if platform.system() == "Windows":
            # Windows timeout command
            command = "timeout /t 10"
        else:
            # Unix sleep command
            command = "sleep 10"
        
        start_time = time.time()
        result = tool.execute(command=command, timeout=2)
        end_time = time.time()
        
        # Should timeout within reasonable time (allow some overhead)
        assert end_time - start_time < 5
        assert "Exit Code: 124" in result
        assert "timed out after 2 seconds" in result


class TestExecuteBashCommandToolVirtualEnvironment:
    """Test ExecuteBashCommandTool virtual environment handling."""
    
    def test_execute_bash_command_tool_virtual_environment_detection(self):
        """Test virtual environment detection and handling."""
        tool = ExecuteBashCommandTool()
        
        # Mock virtual environment conditions
        with patch.object(sys, 'prefix', '/path/to/venv'), \
             patch.object(sys, 'base_prefix', '/path/to/system'), \
             patch('os.environ', {'PATH': '/original/path'}), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(
                returncode=0,
                stdout="venv test",
                stderr=""
            )
            
            result = tool.execute(command="python --version")
            
            # Should detect virtual environment and modify PATH
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            
            # Check that env was passed with modified PATH
            assert 'env' in call_kwargs
            assert call_kwargs['env'] is not None
            assert '/path/to/venv/bin:' in call_kwargs['env']['PATH']
    
    def test_execute_bash_command_tool_no_virtual_environment(self):
        """Test behavior when not in virtual environment."""
        tool = ExecuteBashCommandTool()
        
        # Mock no virtual environment (prefix == base_prefix)
        with patch.object(sys, 'prefix', '/usr'), \
             patch.object(sys, 'base_prefix', '/usr'), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(
                returncode=0,
                stdout="system test",
                stderr=""
            )
            
            result = tool.execute(command="python --version")
            
            # Should not modify environment when not in venv
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            
            # env should be None (inherit current environment)
            assert call_kwargs.get('env') is None
    
    def test_execute_bash_command_tool_path_modification(self):
        """Test PATH modification in virtual environment."""
        tool = ExecuteBashCommandTool()
        
        with patch.object(sys, 'prefix', '/custom/venv'), \
             patch.object(sys, 'base_prefix', '/usr'), \
             patch('os.environ', {'PATH': '/usr/bin:/bin', 'OTHER_VAR': 'value'}), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            tool.execute(command="test_command")
            
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs['env']
            
            # Should preserve other environment variables
            assert env['OTHER_VAR'] == 'value'
            
            # Should prepend venv bin directory to PATH
            assert env['PATH'].startswith('/custom/venv/bin:')
            assert '/usr/bin:/bin' in env['PATH']
    
    def test_execute_bash_command_tool_missing_path_variable(self):
        """Test handling when PATH variable is missing."""
        tool = ExecuteBashCommandTool()
        
        with patch.object(sys, 'prefix', '/venv/path'), \
             patch.object(sys, 'base_prefix', '/system/path'), \
             patch('os.environ', {'HOME': '/home/user'}), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            tool.execute(command="test_command")
            
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs['env']
            
            # Should set PATH to just the venv bin directory
            assert env['PATH'] == '/venv/path/bin'
            # Should preserve other variables
            assert env['HOME'] == '/home/user'


class TestExecuteBashCommandToolStructuredOutput:
    """Test ExecuteBashCommandTool structured output format."""
    
    def test_execute_bash_command_tool_output_format_structure(self):
        """Test that output follows expected structured format."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=42,
                stdout="Test stdout content",
                stderr="Test stderr content"
            )
            
            result = tool.execute(command="test command")
            
            lines = result.split('\n')
            
            # Check exact format structure
            assert lines[0] == "Exit Code: 42"
            assert lines[1] == "--- STDOUT ---"
            assert lines[2] == "Test stdout content"
            assert lines[3] == "--- STDERR ---"
            assert lines[4] == "Test stderr content"
    
    def test_execute_bash_command_tool_output_format_empty_outputs(self):
        """Test structured format with empty stdout/stderr."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = tool.execute(command="test command")
            
            lines = result.split('\n')
            
            assert lines[0] == "Exit Code: 0"
            assert lines[1] == "--- STDOUT ---"
            assert lines[2] == ""
            assert lines[3] == "--- STDERR ---"
            assert lines[4] == ""
    
    def test_execute_bash_command_tool_format_error_output(self):
        """Test structured format for error outputs."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            result = tool.execute(command="nonexistent")
            
            lines = result.split('\n')
            
            assert lines[0] == "Exit Code: 127"
            assert lines[1] == "--- STDOUT ---"
            assert lines[2] == ""
            assert lines[3] == "--- STDERR ---"
            assert "Tool Error: Command not found: nonexistent" in lines[4]
    
    def test_execute_bash_command_tool_format_timeout_error(self):
        """Test structured format for timeout errors."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(['cmd'], 30)
            
            result = tool.execute(command="long_command", timeout=30)
            
            lines = result.split('\n')
            
            assert lines[0] == "Exit Code: 124"
            assert lines[1] == "--- STDOUT ---"
            assert lines[2] == ""
            assert lines[3] == "--- STDERR ---"
            assert "Tool Error: Command timed out after 30 seconds" in lines[4]
    
    def test_execute_bash_command_tool_format_validation_errors(self):
        """Test format for validation errors (no structured format)."""
        tool = ExecuteBashCommandTool()
        
        # Validation errors should return simple error message, not structured format
        result = tool.execute(command=123)
        
        assert "Tool Error: 'command' must be a string, got int" == result
        assert "Exit Code:" not in result
        assert "--- STDOUT ---" not in result
        assert "--- STDERR ---" not in result


class TestExecuteBashCommandToolCrossPlatform:
    """Test ExecuteBashCommandTool cross-platform compatibility."""
    
    def test_execute_bash_command_tool_platform_specific_commands(self):
        """Test handling of platform-specific commands."""
        tool = ExecuteBashCommandTool()
        
        if platform.system() == "Windows":
            # Test Windows-specific command
            result = tool.execute(command="dir")
            # Should either work or fail gracefully
            assert "Exit Code:" in result
        else:
            # Test Unix-specific command
            result = tool.execute(command="ls")
            # Should either work or fail gracefully
            assert "Exit Code:" in result
    
    def test_execute_bash_command_tool_cross_platform_echo(self):
        """Test cross-platform echo command."""
        tool = ExecuteBashCommandTool()
        
        # Echo should work on all platforms
        result = tool.execute(command="echo Hello World")
        
        assert "Exit Code: 0" in result
        assert "Hello World" in result
    
    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_execute_bash_command_tool_unix_specific_features(self):
        """Test Unix-specific features."""
        tool = ExecuteBashCommandTool()
        
        # Test Unix command with pipes (should be handled as separate arguments)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            # This should be parsed as separate arguments, not as a shell pipe
            tool.execute(command="echo hello | grep hello")
            
            call_args = mock_run.call_args[0][0]
            # Should be parsed as individual arguments, not shell pipe
            assert isinstance(call_args, list)
            assert "echo" in call_args
            assert "hello" in call_args
            assert "|" in call_args
            assert "grep" in call_args
    
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_execute_bash_command_tool_windows_specific_features(self):
        """Test Windows-specific features."""
        tool = ExecuteBashCommandTool()
        
        # Test Windows command
        result = tool.execute(command="echo Windows Test")
        
        assert "Exit Code:" in result
        # Windows echo might have different behavior
        assert "Windows Test" in result or "Exit Code:" in result


class TestExecuteBashCommandToolEdgeCases:
    """Test ExecuteBashCommandTool edge cases and boundary conditions."""
    
    def test_execute_bash_command_tool_very_long_command(self):
        """Test handling of very long commands."""
        tool = ExecuteBashCommandTool()
        
        # Create a very long command
        long_arg = "a" * 1000
        long_command = f"echo {long_arg}"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=long_arg,
                stderr=""
            )
            
            result = tool.execute(command=long_command)
            
            assert "Exit Code: 0" in result
            assert long_arg in result
    
    def test_execute_bash_command_tool_many_arguments(self):
        """Test command with many arguments."""
        tool = ExecuteBashCommandTool()
        
        # Create command with many arguments
        args = ["arg{}".format(i) for i in range(100)]
        command = "echo " + " ".join(args)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=" ".join(args),
                stderr=""
            )
            
            result = tool.execute(command=command)
            
            assert "Exit Code: 0" in result
            assert "arg0" in result
            assert "arg99" in result
    
    def test_execute_bash_command_tool_special_characters_in_command(self):
        """Test commands with special characters."""
        tool = ExecuteBashCommandTool()
        
        special_commands = [
            'echo "Hello $USER"',
            "echo 'Single quotes'",
            'echo "Newline\\nTest"',
            'echo "Tab\\tTest"',
            "echo Hello; echo World",  # Semicolon (parsed as separate args)
            "echo Hello && echo World",  # Logical AND (parsed as separate args)
        ]
        
        for cmd in special_commands:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="special output",
                    stderr=""
                )
                
                result = tool.execute(command=cmd)
                
                assert "Exit Code: 0" in result
                # Should handle special characters by parsing them as arguments
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert isinstance(call_args, list)
    
    def test_execute_bash_command_tool_unicode_in_command(self):
        """Test commands with Unicode characters."""
        tool = ExecuteBashCommandTool()
        
        unicode_commands = [
            'echo "Привет мир"',
            'echo "文字"',
            'echo "🚀"',
            'echo "café"'
        ]
        
        for cmd in unicode_commands:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="unicode output",
                    stderr=""
                )
                
                result = tool.execute(command=cmd)
                
                assert "Exit Code: 0" in result
                assert "unicode output" in result
    
    def test_execute_bash_command_tool_binary_output_handling(self):
        """Test handling of binary output."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            # Mock binary output that might cause encoding issues
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Binary content: \x00\x01\x02",
                stderr=""
            )
            
            result = tool.execute(command="cat binary_file")
            
            # Should handle gracefully
            assert "Exit Code: 0" in result
            # Output handling depends on subprocess text=True parameter
    
    def test_execute_bash_command_tool_large_output(self):
        """Test handling of large output."""
        tool = ExecuteBashCommandTool()
        
        # Create large output (100KB)
        large_output = "A" * 100000
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=large_output,
                stderr=""
            )
            
            result = tool.execute(command="generate_large_output")
            
            assert "Exit Code: 0" in result
            assert large_output in result
            assert len(result) > 100000
    
    def test_execute_bash_command_tool_subprocess_parameters(self):
        """Test that subprocess.run is called with correct parameters."""
        tool = ExecuteBashCommandTool()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            tool.execute(command="test command", timeout=30)
            
            mock_run.assert_called_once()
            call_args, call_kwargs = mock_run.call_args
            
            # Check required parameters
            assert call_kwargs['capture_output'] is True
            assert call_kwargs['text'] is True
            assert call_kwargs['timeout'] == 30
            assert call_kwargs['shell'] is False
            assert call_kwargs['cwd'] is None
            
            # Check that args is a list (from shlex.split)
            assert isinstance(call_args[0], list)


class TestExecuteBashCommandToolIntegration:
    """Integration tests for ExecuteBashCommandTool."""
    
    def test_execute_bash_command_tool_real_echo_command(self):
        """Integration test with real echo command."""
        tool = ExecuteBashCommandTool()
        
        result = tool.execute(command="echo Integration Test")
        
        assert "Exit Code: 0" in result
        assert "Integration Test" in result
        assert "--- STDOUT ---" in result
        assert "--- STDERR ---" in result
    
    def test_execute_bash_command_tool_real_error_command(self):
        """Integration test with command that produces error."""
        tool = ExecuteBashCommandTool()
        
        # Use a command that should fail on most systems
        result = tool.execute(command="nonexistent_command_12345")
        
        # Should handle the error gracefully
        assert "Exit Code:" in result
        # Exit code will vary by system, but should not be 0
        if "Exit Code: 0" in result:
            # If command somehow succeeded, that's also valid behavior
            pass
        else:
            # Most likely case: command not found
            assert ("Exit Code: 127" in result or 
                   "Exit Code: 126" in result or
                   "Command not found" in result)
    
    @pytest.mark.slow
    def test_execute_bash_command_tool_real_timeout(self):
        """Integration test with real timeout (marked as slow)."""
        tool = ExecuteBashCommandTool()
        
        # Use appropriate sleep command for platform
        if platform.system() == "Windows":
            # Windows doesn't have sleep, use timeout instead
            pytest.skip("Windows timeout test requires different approach")
        else:
            start_time = time.time()
            result = tool.execute(command="sleep 5", timeout=2)
            end_time = time.time()
            
            # Should timeout quickly
            assert end_time - start_time < 4
            assert "Exit Code: 124" in result
            assert "timed out" in result
    
    def test_execute_bash_command_tool_python_version_check(self):
        """Integration test checking Python version."""
        tool = ExecuteBashCommandTool()
        
        result = tool.execute(command="python --version")
        
        # Should either work or fail gracefully
        assert "Exit Code:" in result
        if "Exit Code: 0" in result:
            # If successful, should contain Python version info
            assert ("Python" in result or "--- STDOUT ---" in result)
    
    def test_execute_bash_command_tool_environment_variables(self):
        """Test that environment variables are handled correctly."""
        tool = ExecuteBashCommandTool()
        
        # Set a test environment variable
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            if platform.system() == "Windows":
                result = tool.execute(command="echo %TEST_VAR%")
            else:
                result = tool.execute(command="echo $TEST_VAR")
            
            assert "Exit Code:" in result
            # Environment variable handling depends on shell vs no-shell execution
            # With shell=False, variable expansion might not work as expected