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