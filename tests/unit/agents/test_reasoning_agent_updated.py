"""Unit tests for the updated ReasoningAgent class with LLM integration."""

import pytest
from unittest.mock import Mock, MagicMock

from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.core.state import State
from oniks.tools.base import Tool
from oniks.llm.client import OllamaClient, OllamaConnectionError


class TestReasoningAgentWithLLM:
    """Test ReasoningAgent with LLM integration."""
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        tool1 = Mock(spec=Tool)
        tool1.name = "read_file"
        tool1.description = "Reads the entire content of a specified file"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "write_file"
        tool2.description = "Writes content to a specified file"
        
        return [tool1, tool2]
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for testing."""
        client = Mock(spec=OllamaClient)
        client.invoke.return_value = (
            "Tool: read_file\n"
            "Arguments: {\"file_path\": \"task.txt\"}\n"
            "Reasoning: The goal requires reading a file"
        )
        return client
    
    def test_initialization_success(self, mock_tools, mock_llm_client):
        """Test successful ReasoningAgent initialization."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        
        assert agent.name == "reasoning_agent"
        assert agent.tools == mock_tools
        assert agent.llm_client == mock_llm_client
    
    def test_initialization_none_llm_client_raises_error(self, mock_tools):
        """Test initialization with None LLM client raises ValueError."""
        with pytest.raises(ValueError, match="LLM client cannot be None"):
            ReasoningAgent("agent", mock_tools, None)
    
    def test_execute_with_llm_success(self, mock_tools, mock_llm_client):
        """Test successful execution with LLM response."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        
        state = State()
        state.data["goal"] = "Read the contents of file task.txt"
        
        result_state = agent.execute(state)
        
        # Verify LLM was called
        mock_llm_client.invoke.assert_called_once()
        
        # Verify state updates
        assert "last_prompt" in result_state.data
        assert "llm_response" in result_state.data
        assert result_state.data["next_tool"] == "read_file"
        assert result_state.data["tool_args"] == {"file_path": "task.txt"}
        assert result_state.data["file_path"] == "task.txt"
    
    def test_execute_with_llm_failure_fallback(self, mock_tools, mock_llm_client):
        """Test execution falls back to basic reasoning when LLM fails."""
        # Configure LLM client to raise an exception
        mock_llm_client.invoke.side_effect = OllamaConnectionError("Connection failed")
        
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        
        state = State()
        state.data["goal"] = "Read the contents of file task.txt"
        
        result_state = agent.execute(state)
        
        # Verify LLM was attempted
        mock_llm_client.invoke.assert_called_once()
        
        # Verify fallback reasoning was used
        assert result_state.data["next_tool"] == "read_file"
        assert result_state.data["tool_args"] == {"file_path": "task.txt"}
        assert "LLM invocation failed" in str(result_state.message_history)
    
    def test_parse_llm_response_success(self, mock_tools, mock_llm_client):
        """Test successful parsing of LLM response."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        state = State()
        
        llm_response = '''
        Based on the goal, I need to use a file reading tool.
        
        Tool: read_file
        Arguments: {"file_path": "example.txt", "encoding": "utf-8"}
        Reasoning: The goal clearly indicates reading a file.
        '''
        
        agent._parse_llm_response(llm_response, state)
        
        assert state.data["next_tool"] == "read_file"
        assert state.data["tool_args"] == {"file_path": "example.txt", "encoding": "utf-8"}
        assert state.data["file_path"] == "example.txt"
        assert state.data["encoding"] == "utf-8"
    
    def test_parse_llm_response_malformed_json(self, mock_tools, mock_llm_client):
        """Test parsing LLM response with malformed JSON arguments."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        state = State()
        
        llm_response = '''
        Tool: read_file
        Arguments: {malformed json}
        '''
        
        agent._parse_llm_response(llm_response, state)
        
        assert state.data["next_tool"] == "read_file"
        assert "tool_args" not in state.data
        messages = [str(msg) for msg in state.message_history]
        assert any("Failed to parse arguments as JSON" in msg for msg in messages)
    
    def test_parse_llm_response_no_tool_pattern(self, mock_tools, mock_llm_client):
        """Test parsing LLM response without Tool pattern."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        state = State()
        
        llm_response = "This is just a regular response without the expected format."
        
        agent._parse_llm_response(llm_response, state)
        
        assert "next_tool" not in state.data
        messages = [str(msg) for msg in state.message_history]
        assert any("No tool found in LLM response" in msg for msg in messages)
    
    def test_fallback_reasoning_english(self, mock_tools, mock_llm_client):
        """Test fallback reasoning with English goal."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        state = State()
        
        agent._perform_basic_reasoning("Read file example.txt", state)
        
        assert state.data["next_tool"] == "read_file"
        assert state.data["tool_args"] == {"file_path": "task.txt"}
    
    def test_fallback_reasoning_russian(self, mock_tools, mock_llm_client):
        """Test fallback reasoning with Russian goal."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        state = State()
        
        agent._perform_basic_reasoning("Прочитать файл example.txt", state)
        
        assert state.data["next_tool"] == "read_file"
        assert state.data["tool_args"] == {"file_path": "task.txt"}
    
    def test_generate_llm_prompt(self, mock_tools, mock_llm_client):
        """Test LLM prompt generation."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        
        prompt = agent._generate_llm_prompt("Read file example.txt")
        
        assert "Goal Analysis and Tool Selection" in prompt
        assert "Read file example.txt" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "Tool: [tool_name]" in prompt
        assert "Arguments: [tool_arguments]" in prompt
    
    def test_tool_management(self, mock_tools, mock_llm_client):
        """Test tool management methods."""
        agent = ReasoningAgent("reasoning_agent", mock_tools, mock_llm_client)
        
        # Test get_available_tools
        available_tools = agent.get_available_tools()
        assert len(available_tools) == 2
        assert available_tools == mock_tools
        
        # Test add_tool
        new_tool = Mock(spec=Tool)
        new_tool.name = "new_tool"
        agent.add_tool(new_tool)
        assert len(agent.tools) == 3
        
        # Test remove_tool
        removed = agent.remove_tool("new_tool")
        assert removed is True
        assert len(agent.tools) == 2
        
        # Test remove non-existent tool
        removed = agent.remove_tool("non_existent")
        assert removed is False
        assert len(agent.tools) == 2


class TestReasoningAgentSanitization:
    """Test ReasoningAgent LLM response sanitization functionality."""
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for sanitization testing."""
        tool = Mock(spec=Tool)
        tool.name = "read_file"
        tool.description = "Reads the entire content of a specified file"
        return [tool]
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for sanitization testing."""
        return Mock(spec=OllamaClient)
    
    @pytest.fixture
    def agent(self, mock_tools, mock_llm_client):
        """Create ReasoningAgent for sanitization testing."""
        return ReasoningAgent("test_agent", mock_tools, mock_llm_client)
    
    def test_sanitize_llm_response_remove_markdown_symbols(self, agent):
        """Test removal of Markdown symbols from LLM response."""
        raw_response = (
            "# Tool: *read_file*\n"
            "## Arguments: {'file_path': '__test.txt__'}\n"
            "### Reasoning: **Bold** and _italic_ text"
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        # Markdown formatting should be removed but content preserved
        assert "*" not in sanitized  # No asterisks for bold/italic
        assert "#" not in sanitized  # No hash symbols for headers
        
        # Check that markdown emphasis is removed but identifier underscores remain
        assert "read_file" in sanitized  # Tool name preserved
        assert "file_path" in sanitized  # Parameter name preserved
        assert "test.txt" in sanitized   # __test.txt__ should become test.txt
        assert "__test.txt__" not in sanitized  # Double underscores should be removed
        assert "_italic_" not in sanitized  # Italic emphasis should be removed
        
        # Content should remain
        assert "Tool: read_file" in sanitized
        assert "Arguments:" in sanitized
        assert "Reasoning:" in sanitized
        assert "Bold" in sanitized
        assert "italic" in sanitized
    
    def test_sanitize_llm_response_strip_whitespace(self, agent):
        """Test stripping of leading and trailing whitespace from lines."""
        raw_response = (
            "   Tool: read_file   \n"
            "\t\tArguments: {'file_path': 'test.txt'}  \n"
            "  \n"
            "    Reasoning: Goal requires file reading    "
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        lines = sanitized.split('\n')
        
        # All lines should be stripped and empty lines removed
        assert lines[0] == "Tool: read_file"
        assert lines[1] == "Arguments: {\"file_path\": \"test.txt\"}"  # Also should fix quotes
        assert lines[2] == "Reasoning: Goal requires file reading"  # Empty line removed
        assert len(lines) == 3  # Should be exactly 3 lines, no empty lines
    
    def test_sanitize_llm_response_fix_json_quotes(self, agent):
        """Test fixing of single quotes to double quotes in JSON."""
        raw_response = (
            "Tool: read_file\n"
            "Arguments: {'file_path': 'test.txt', 'encoding': 'utf-8'}\n"
            "Reasoning: Fix single quotes"
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        # Single quotes in JSON should be replaced with double quotes
        assert "'file_path'" not in sanitized
        assert "'test.txt'" not in sanitized
        assert "'encoding'" not in sanitized
        assert "'utf-8'" not in sanitized
        
        assert '"file_path"' in sanitized
        assert '"test.txt"' in sanitized
        assert '"encoding"' in sanitized
        assert '"utf-8"' in sanitized
        
        # Tool name and other content should be preserved
        assert "Tool: read_file" in sanitized
        assert "Reasoning: Fix single quotes" in sanitized
    
    def test_sanitize_llm_response_fix_unquoted_keys(self, agent):
        """Test fixing of unquoted keys in JSON-like structures."""
        raw_response = (
            "Tool: read_file\n"
            "Arguments: {file_path: \"test.txt\", encoding: \"utf-8\"}\n"
            "Reasoning: Fix unquoted keys"
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        # Unquoted keys should be quoted
        assert '"file_path"' in sanitized
        assert '"encoding"' in sanitized
        assert 'file_path:' not in sanitized or sanitized.count('file_path:') == 0
    
    def test_sanitize_llm_response_remove_empty_lines(self, agent):
        """Test removal of empty lines created by cleaning."""
        raw_response = (
            "# Header\n"
            "\n"
            "Tool: read_file\n"
            "   \n"
            "Arguments: {'file_path': 'test.txt'}\n"
            "\t\n"
            "Reasoning: Remove empty lines"
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        lines = sanitized.split('\n')
        
        # No empty lines should remain
        for line in lines:
            assert line.strip() != ""
        
        # Content should be preserved
        assert "Tool: read_file" in lines
        assert any("Arguments:" in line for line in lines)
        assert any("Reasoning:" in line for line in lines)
    
    def test_sanitize_llm_response_complex_example(self, agent):
        """Test sanitization with a complex real-world example."""
        raw_response = (
            "## Analysis\n"
            "\n"
            "Based on the goal, I need to use the *read_file* tool.\n"
            "\n"
            "### Tool Selection\n"
            "   **Tool: read_file**   \n"
            "  Arguments: {'file_path': '__important_file__.txt'}  \n"
            "\n"
            "#### Reasoning\n"
            "The goal clearly states to read a file, so the _read_file_ tool is appropriate.\n"
        )
        
        sanitized = agent._sanitize_llm_response(raw_response)
        
        # Should remove all markdown formatting symbols
        assert "#" not in sanitized  # No hash symbols for headers
        assert "*" not in sanitized  # No asterisks for bold/italic
        
        # Should preserve identifier underscores but remove emphasis underscores
        assert "read_file" in sanitized  # Tool name preserved
        assert "file_path" in sanitized  # Parameter name preserved  
        assert "_read_file_" not in sanitized  # Emphasis underscores should be removed from reasoning
        
        # Should fix quotes
        assert '"file_path"' in sanitized
        assert '"important_file.txt"' in sanitized  # __important_file__.txt should become important_file.txt
        assert '__important_file__' not in sanitized  # Double underscores should be removed
        
        # Should preserve essential content
        assert "Tool: read_file" in sanitized
        assert "Arguments:" in sanitized
        assert "Reasoning" in sanitized  # Headers lose their : after # removal
        assert "Analysis" in sanitized
        assert "Tool Selection" in sanitized
        
        # Should have no empty lines
        lines = sanitized.split('\n')
        for line in lines:
            assert line.strip() != ""
    
    def test_sanitize_llm_response_non_string_input(self, agent):
        """Test sanitization with non-string input."""
        # Test with None
        sanitized = agent._sanitize_llm_response(None)
        assert sanitized == ""
        
        # Test with number
        sanitized = agent._sanitize_llm_response(123)
        assert sanitized == "123"
        
        # Test with list
        sanitized = agent._sanitize_llm_response(["Tool:", "read_file"])
        assert "Tool:" in sanitized
        assert "read_file" in sanitized
    
    def test_sanitize_llm_response_empty_string(self, agent):
        """Test sanitization with empty string."""
        sanitized = agent._sanitize_llm_response("")
        assert sanitized == ""
    
    def test_sanitize_llm_response_integration(self, mock_tools, mock_llm_client):
        """Test sanitization integration in the execute method."""
        # Configure LLM client to return response with markdown and formatting issues
        mock_llm_client.invoke.return_value = (
            "# Analysis Result\n"
            "   **Tool: *read_file***   \n"
            "  Arguments: {'file_path': 'test.txt'}  \n"
            "#### Reasoning: File reading is _required_"
        )
        
        agent = ReasoningAgent("test_agent", mock_tools, mock_llm_client)
        
        state = State()
        state.data["goal"] = "Read the contents of file test.txt"
        
        result_state = agent.execute(state)
        
        # Verify LLM was called
        mock_llm_client.invoke.assert_called_once()
        
        # Verify sanitization worked and parsing succeeded
        assert result_state.data["next_tool"] == "read_file"
        assert result_state.data["tool_args"] == {"file_path": "test.txt"}
        
        # Verify sanitization message was added
        messages = [str(msg) for msg in result_state.message_history]
        assert any("Applied sanitization to LLM response" in msg for msg in messages)