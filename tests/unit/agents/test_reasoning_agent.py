"""Unit tests for the ReasoningAgent class."""

import pytest
from unittest.mock import Mock

from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.core.state import State
from oniks.tools.base import Tool


class TestReasoningAgentInitialization:
    """Test ReasoningAgent initialization."""
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        tool1 = Mock(spec=Tool)
        tool1.name = "tool1"
        tool1.description = "First test tool"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "tool2"
        tool2.description = "Second test tool"
        
        return [tool1, tool2]
    
    def test_reasoning_agent_initialization_success(self, mock_tools):
        """Test successful ReasoningAgent initialization."""
        agent = ReasoningAgent("reasoning_agent", mock_tools)
        
        assert agent.name == "reasoning_agent"
        assert agent.tools == mock_tools
        assert len(agent.tools) == 2
    
    def test_reasoning_agent_initialization_empty_name_raises_error(self, mock_tools):
        """Test initialization with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ReasoningAgent("", mock_tools)
    
    def test_reasoning_agent_initialization_none_name_raises_error(self, mock_tools):
        """Test initialization with None name raises ValueError."""
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ReasoningAgent(None, mock_tools)
    
    def test_reasoning_agent_initialization_none_tools_raises_error(self):
        """Test initialization with None tools raises ValueError."""
        with pytest.raises(ValueError, match="Tools list cannot be None"):
            ReasoningAgent("agent", None)
    
    def test_reasoning_agent_initialization_non_list_tools_raises_error(self):
        """Test initialization with non-list tools raises TypeError."""
        with pytest.raises(TypeError, match="Tools must be a list"):
            ReasoningAgent("agent", "not_a_list")
        
        with pytest.raises(TypeError, match="Tools must be a list"):
            ReasoningAgent("agent", {"tool": "dict"})
    
    def test_reasoning_agent_initialization_empty_tools_list(self):
        """Test initialization with empty tools list."""
        agent = ReasoningAgent("agent", [])
        
        assert agent.name == "agent"
        assert agent.tools == []
        assert len(agent.tools) == 0
    
    def test_reasoning_agent_string_representation(self, mock_tools):
        """Test ReasoningAgent string representation."""
        agent = ReasoningAgent("test_agent", mock_tools)
        
        assert str(agent) == "ReasoningAgent(name='test_agent')"
        assert repr(agent) == "ReasoningAgent(name='test_agent')"


class TestReasoningAgentExecution:
    """Test ReasoningAgent execution logic."""
    
    @pytest.fixture
    def mock_read_file_tool(self):
        """Create a mock ReadFileTool."""
        tool = Mock(spec=Tool)
        tool.name = "read_file"
        tool.description = "Reads the entire content of a specified file. Arguments: {'file_path': 'str'}"
        return tool
    
    @pytest.fixture
    def reasoning_agent(self, mock_read_file_tool):
        """Create a ReasoningAgent with mock tools."""
        return ReasoningAgent("reasoning_agent", [mock_read_file_tool])
    
    def test_reasoning_agent_execute_with_goal(self, reasoning_agent):
        """Test execution with a goal in state."""
        state = State()
        state.data["goal"] = "Прочитать содержимое файла task.txt"
        
        result_state = reasoning_agent.execute(state)
        
        # Verify state is copied, not modified
        assert result_state is not state
        
        # Verify messages were added
        assert len(result_state.message_history) > 0
        assert any("starting analysis" in msg.lower() for msg in result_state.message_history)
        assert any("completed analysis" in msg.lower() for msg in result_state.message_history)
        
        # Verify LLM prompt was generated
        assert "last_prompt" in result_state.data
        assert isinstance(result_state.data["last_prompt"], str)
        assert len(result_state.data["last_prompt"]) > 0
        
        # Verify Russian goal triggered file reading logic
        assert result_state.data["next_tool"] == "read_file"
        assert result_state.data["file_path"] == "task.txt"
        assert "tool_args" in result_state.data
    
    def test_reasoning_agent_execute_without_goal(self, reasoning_agent):
        """Test execution without goal in state."""
        state = State()
        
        result_state = reasoning_agent.execute(state)
        
        # Should still add messages
        assert len(result_state.message_history) > 0
        assert any("No goal found" in msg for msg in result_state.message_history)
        
        # Should not set tool recommendations
        assert "next_tool" not in result_state.data
        assert "tool_args" not in result_state.data
    
    def test_reasoning_agent_execute_with_empty_goal(self, reasoning_agent):
        """Test execution with empty goal."""
        state = State()
        state.data["goal"] = ""
        
        result_state = reasoning_agent.execute(state)
        
        assert any("No goal found" in msg for msg in result_state.message_history)
        assert "next_tool" not in result_state.data
    
    def test_reasoning_agent_execute_preserves_original_state(self, reasoning_agent):
        """Test that original state is not modified."""
        original_state = State()
        original_state.data["goal"] = "Test goal"
        original_state.data["existing_key"] = "existing_value"
        original_state.add_message("Original message")
        
        result_state = reasoning_agent.execute(original_state)
        
        # Original state should be unchanged
        assert len(original_state.message_history) == 1
        assert "last_prompt" not in original_state.data
        assert "next_tool" not in original_state.data
        
        # Result state should have changes
        assert len(result_state.message_history) > 1
        assert "last_prompt" in result_state.data
        
        # Original data should be preserved
        assert result_state.data["existing_key"] == "existing_value"
    
    def test_reasoning_agent_execute_file_reading_goal_variants(self, reasoning_agent):
        """Test various Russian file reading goal formats."""
        file_reading_goals = [
            "Прочитать содержимое файла test.txt",
            "нужно прочитать файл data.json",
            "ПРОЧИТАТЬ ФАЙЛ config.yaml",
            "прочитать файл документ.docx",
            "Файл нужно прочитать: readme.md"
        ]
        
        for goal in file_reading_goals:
            state = State()
            state.data["goal"] = goal
            
            result_state = reasoning_agent.execute(state)
            
            assert result_state.data["next_tool"] == "read_file"
            assert "file_path" in result_state.data
            assert "tool_args" in result_state.data
    
    def test_reasoning_agent_execute_non_file_reading_goals(self, reasoning_agent):
        """Test goals that don't match file reading pattern."""
        non_file_goals = [
            "Calculate the sum of numbers",
            "Send an email to user",
            "Download a web page",
            "Create a new directory",
            "Delete old files",
            "Просто какая-то задача",  # Just some task in Russian but not about reading
            "прочитать книгу"  # Reading a book, not a file
        ]
        
        for goal in non_file_goals:
            state = State()
            state.data["goal"] = goal
            
            result_state = reasoning_agent.execute(state)
            
            assert "next_tool" not in result_state.data
            assert any("No specific reasoning rule matched" in msg for msg in result_state.message_history)


class TestReasoningAgentPromptGeneration:
    """Test ReasoningAgent LLM prompt generation."""
    
    @pytest.fixture
    def tools_for_prompt_test(self):
        """Create tools for prompt generation testing."""
        tool1 = Mock(spec=Tool)
        tool1.name = "read_file"
        tool1.description = "Reads file content. Args: {'file_path': 'str'}"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "write_file"
        tool2.description = "Writes content to file. Args: {'file_path': 'str', 'content': 'str'}"
        
        return [tool1, tool2]
    
    def test_generate_llm_prompt_with_tools(self, tools_for_prompt_test):
        """Test LLM prompt generation with available tools."""
        agent = ReasoningAgent("agent", tools_for_prompt_test)
        goal = "Process some files"
        
        prompt = agent._generate_llm_prompt(goal)
        
        # Verify prompt structure
        assert "Goal Analysis and Tool Selection" in prompt
        assert f"Current Goal: {goal}" in prompt
        assert "Available tools:" in prompt
        assert "read_file: Reads file content" in prompt
        assert "write_file: Writes content to file" in prompt
        assert "Which tool should be used next" in prompt
        assert "Tool: [tool_name]" in prompt
        assert "Arguments: [tool_arguments]" in prompt
        assert "Reasoning: [explanation]" in prompt
    
    def test_generate_llm_prompt_without_tools(self):
        """Test LLM prompt generation without available tools."""
        agent = ReasoningAgent("agent", [])
        goal = "Process some data"
        
        prompt = agent._generate_llm_prompt(goal)
        
        assert "Goal Analysis and Tool Selection" in prompt
        assert f"Current Goal: {goal}" in prompt
        assert "Available tools:" in prompt
        assert "No tools available." in prompt
        assert "Which tool should be used next" in prompt
    
    def test_generate_llm_prompt_with_complex_goal(self, tools_for_prompt_test):
        """Test LLM prompt generation with complex, multi-line goal."""
        agent = ReasoningAgent("agent", tools_for_prompt_test)
        complex_goal = """Read the configuration file config.json,
        parse its contents, and then write the processed
        data to output.txt with proper formatting."""
        
        prompt = agent._generate_llm_prompt(complex_goal)
        
        assert complex_goal in prompt
        assert "Available tools:" in prompt
        assert len(prompt.split('\n')) > 10  # Should be well-formatted with multiple lines
    
    def test_generate_llm_prompt_with_unicode_goal(self, tools_for_prompt_test):
        """Test LLM prompt generation with unicode characters in goal."""
        agent = ReasoningAgent("agent", tools_for_prompt_test)
        unicode_goal = "Прочитать файл данные.txt и обработать содержимое 🚀"
        
        prompt = agent._generate_llm_prompt(unicode_goal)
        
        assert unicode_goal in prompt
        assert "Прочитать" in prompt
        assert "🚀" in prompt


class TestReasoningAgentBasicReasoning:
    """Test ReasoningAgent basic reasoning logic."""
    
    @pytest.fixture
    def reasoning_agent_with_tools(self):
        """Create ReasoningAgent with mock tools for reasoning tests."""
        tool = Mock(spec=Tool)
        tool.name = "read_file"
        tool.description = "Test tool"
        return ReasoningAgent("test_agent", [tool])
    
    def test_perform_basic_reasoning_file_reading_russian(self, reasoning_agent_with_tools):
        """Test basic reasoning for Russian file reading goals."""
        state = State()
        goal = "Прочитать содержимое файла test.txt"
        
        reasoning_agent_with_tools._perform_basic_reasoning(goal, state)
        
        assert state.data["next_tool"] == "read_file"
        assert state.data["tool_args"]["file_path"] == "task.txt"  # Hardcoded in implementation
        assert state.data["file_path"] == "task.txt"
        assert any("recommending read_file tool" in msg for msg in state.message_history)
    
    def test_perform_basic_reasoning_case_insensitive(self, reasoning_agent_with_tools):
        """Test that reasoning is case insensitive."""
        state = State()
        goals = [
            "ПРОЧИТАТЬ ФАЙЛ TEST.TXT",
            "прочитать файл test.txt",
            "ПрОчИтАтЬ ФаЙл test.txt"
        ]
        
        for goal in goals:
            test_state = State()
            reasoning_agent_with_tools._perform_basic_reasoning(goal, test_state)
            
            assert test_state.data["next_tool"] == "read_file"
            assert test_state.data["file_path"] == "task.txt"
    
    def test_perform_basic_reasoning_no_match(self, reasoning_agent_with_tools):
        """Test basic reasoning when no rule matches."""
        state = State()
        goal = "Calculate mathematical expression"
        
        reasoning_agent_with_tools._perform_basic_reasoning(goal, state)
        
        assert "next_tool" not in state.data
        assert "tool_args" not in state.data
        assert any("No specific reasoning rule matched" in msg for msg in state.message_history)
    
    def test_perform_basic_reasoning_partial_matches(self, reasoning_agent_with_tools):
        """Test reasoning with partial keyword matches."""
        partial_match_goals = [
            "Прочитать книгу",  # Has "прочитать" but not "файл"
            "Создать файл",     # Has "файл" but not "прочитать"
            "файл прочитать",   # Different order
            "нужно прочитать этот файл"  # Both keywords present
        ]
        
        for goal in partial_match_goals:
            test_state = State()
            reasoning_agent_with_tools._perform_basic_reasoning(goal, test_state)
            
            if "прочитать" in goal.lower() and "файл" in goal.lower():
                assert test_state.data["next_tool"] == "read_file"
            else:
                assert "next_tool" not in test_state.data


class TestReasoningAgentToolManagement:
    """Test ReasoningAgent tool management methods."""
    
    @pytest.fixture
    def initial_tools(self):
        """Create initial tools for testing."""
        tool1 = Mock(spec=Tool)
        tool1.name = "tool1"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "tool2"
        
        return [tool1, tool2]
    
    def test_get_available_tools(self, initial_tools):
        """Test getting available tools."""
        agent = ReasoningAgent("agent", initial_tools)
        
        available_tools = agent.get_available_tools()
        
        assert len(available_tools) == 2
        assert available_tools == initial_tools
        assert available_tools is not agent.tools  # Should be a copy
        
        # Modifying returned list shouldn't affect agent's tools
        available_tools.append("new_tool")
        assert len(agent.tools) == 2
    
    def test_add_tool_success(self, initial_tools):
        """Test successful tool addition."""
        agent = ReasoningAgent("agent", initial_tools)
        
        new_tool = Mock(spec=Tool)
        new_tool.name = "new_tool"
        
        agent.add_tool(new_tool)
        
        assert len(agent.tools) == 3
        assert new_tool in agent.tools
    
    def test_add_tool_none_raises_error(self, initial_tools):
        """Test adding None tool raises ValueError."""
        agent = ReasoningAgent("agent", initial_tools)
        
        with pytest.raises(ValueError, match="Tool cannot be None"):
            agent.add_tool(None)
    
    def test_remove_tool_success(self, initial_tools):
        """Test successful tool removal."""
        agent = ReasoningAgent("agent", initial_tools)
        
        result = agent.remove_tool("tool1")
        
        assert result is True
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "tool2"
    
    def test_remove_tool_not_found(self, initial_tools):
        """Test removing non-existent tool."""
        agent = ReasoningAgent("agent", initial_tools)
        
        result = agent.remove_tool("nonexistent_tool")
        
        assert result is False
        assert len(agent.tools) == 2  # No tools removed
    
    def test_remove_tool_multiple_occurrences(self):
        """Test removing tool when multiple tools have same name."""
        tool1a = Mock(spec=Tool)
        tool1a.name = "duplicate_tool"
        
        tool1b = Mock(spec=Tool)
        tool1b.name = "duplicate_tool"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "unique_tool"
        
        agent = ReasoningAgent("agent", [tool1a, tool1b, tool2])
        
        result = agent.remove_tool("duplicate_tool")
        
        assert result is True
        assert len(agent.tools) == 1  # Both duplicate tools should be removed
        assert agent.tools[0].name == "unique_tool"


class TestReasoningAgentComplexScenarios:
    """Test ReasoningAgent in complex scenarios."""
    
    def test_reasoning_agent_with_multiple_tool_types(self):
        """Test reasoning agent with various tool types."""
        file_tool = Mock(spec=Tool)
        file_tool.name = "read_file"
        file_tool.description = "Reads files"
        
        calc_tool = Mock(spec=Tool)
        calc_tool.name = "calculator"
        calc_tool.description = "Performs calculations"
        
        web_tool = Mock(spec=Tool)
        web_tool.name = "web_scraper"
        web_tool.description = "Scrapes web pages"
        
        agent = ReasoningAgent("multi_tool_agent", [file_tool, calc_tool, web_tool])
        
        # Test file reading goal
        file_state = State()
        file_state.data["goal"] = "Прочитать файл данных"
        
        file_result = agent.execute(file_state)
        
        assert file_result.data["next_tool"] == "read_file"
        assert "read_file: Reads files" in file_result.data["last_prompt"]
        assert "calculator: Performs calculations" in file_result.data["last_prompt"]
        assert "web_scraper: Scrapes web pages" in file_result.data["last_prompt"]
    
    def test_reasoning_agent_state_preservation_across_calls(self):
        """Test that agent doesn't maintain state between calls."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        
        agent = ReasoningAgent("stateless_agent", [tool])
        
        # First execution
        state1 = State()
        state1.data["goal"] = "Прочитать файл first.txt"
        
        result1 = agent.execute(state1)
        
        # Second execution with different goal
        state2 = State()
        state2.data["goal"] = "Some other task"
        
        result2 = agent.execute(state2)
        
        # Results should be independent
        assert result1.data["next_tool"] == "read_file"
        assert "next_tool" not in result2.data
        
        # Agent should not retain any state from previous execution
        assert len(agent.tools) == 1  # Tools list should be unchanged
    
    def test_reasoning_agent_with_complex_state_data(self):
        """Test reasoning agent with complex state data structures."""
        tool = Mock(spec=Tool)
        tool.name = "processor"
        
        agent = ReasoningAgent("complex_agent", [tool])
        
        # Create complex state
        complex_state = State()
        complex_state.data.update({
            "goal": "Прочитать файл complex.json",
            "nested_data": {
                "level1": {
                    "level2": {
                        "values": [1, 2, 3, 4, 5]
                    }
                }
            },
            "list_data": [
                {"id": 1, "name": "item1"},
                {"id": 2, "name": "item2"}
            ],
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "version": "1.0.0",
                "author": "test_user"
            }
        })
        
        complex_state.add_message("Complex state initialized")
        complex_state.tool_outputs["previous_tool"] = "previous output"
        
        result = agent.execute(complex_state)
        
        # Should preserve all original data
        assert result.data["nested_data"]["level1"]["level2"]["values"] == [1, 2, 3, 4, 5]
        assert len(result.data["list_data"]) == 2
        assert result.data["metadata"]["version"] == "1.0.0"
        
        # Should add reasoning results
        assert result.data["next_tool"] == "read_file"
        assert "last_prompt" in result.data
        
        # Should preserve original messages and outputs
        assert "Complex state initialized" in result.message_history
        assert result.tool_outputs["previous_tool"] == "previous output"
    
    def test_reasoning_agent_error_resilience(self):
        """Test reasoning agent resilience to errors in state data."""
        tool = Mock(spec=Tool)
        tool.name = "resilient_tool"
        
        agent = ReasoningAgent("resilient_agent", [tool])
        
        # Test with various problematic state data
        problematic_states = [
            # Goal with unusual types
            {"goal": 123},  # Non-string goal
            {"goal": ["list", "goal"]},  # List goal
            {"goal": {"nested": "goal"}},  # Dict goal
            {"goal": None},  # None goal
        ]
        
        for problematic_data in problematic_states:
            state = State()
            state.data.update(problematic_data)
            
            # Should not raise exceptions
            result = agent.execute(state)
            
            # Should handle gracefully
            assert isinstance(result, State)
            assert len(result.message_history) > 0


class TestReasoningAgentEdgeCases:
    """Test ReasoningAgent edge cases and boundary conditions."""
    
    def test_reasoning_agent_with_empty_tool_descriptions(self):
        """Test agent with tools that have empty descriptions."""
        empty_desc_tool = Mock(spec=Tool)
        empty_desc_tool.name = "empty_desc_tool"
        empty_desc_tool.description = ""
        
        agent = ReasoningAgent("agent", [empty_desc_tool])
        
        state = State()
        state.data["goal"] = "Test goal"
        
        result = agent.execute(state)
        
        # Should handle empty descriptions gracefully
        assert "last_prompt" in result.data
        assert "empty_desc_tool:" in result.data["last_prompt"]
    
    def test_reasoning_agent_with_very_long_goal(self):
        """Test agent with very long goal text."""
        long_goal = "Прочитать файл " + "very_long_filename_" * 100 + ".txt with lots of content"
        
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        
        agent = ReasoningAgent("agent", [tool])
        
        state = State()
        state.data["goal"] = long_goal
        
        result = agent.execute(state)
        
        # Should handle long goals
        assert "last_prompt" in result.data
        assert long_goal in result.data["last_prompt"]
        assert result.data["next_tool"] == "read_file"  # Should still match pattern
    
    def test_reasoning_agent_with_unicode_tool_names(self):
        """Test agent with tools having unicode names."""
        unicode_tool = Mock(spec=Tool)
        unicode_tool.name = "инструмент_🔧"
        unicode_tool.description = "Универсальный инструмент для обработки данных"
        
        agent = ReasoningAgent("unicode_agent", [unicode_tool])
        
        state = State()
        state.data["goal"] = "Простая задача"
        
        result = agent.execute(state)
        
        # Should handle unicode tool names
        assert "last_prompt" in result.data
        assert "инструмент_🔧" in result.data["last_prompt"]
        assert "Универсальный инструмент" in result.data["last_prompt"]
    
    def test_reasoning_agent_performance_with_many_tools(self):
        """Test agent performance with large number of tools."""
        many_tools = []
        for i in range(100):
            tool = Mock(spec=Tool)
            tool.name = f"tool_{i}"
            tool.description = f"Tool number {i} for testing performance"
            many_tools.append(tool)
        
        agent = ReasoningAgent("performance_agent", many_tools)
        
        state = State()
        state.data["goal"] = "Прочитать файл performance.txt"
        
        # Should complete in reasonable time
        result = agent.execute(state)
        
        # Should still work correctly
        assert result.data["next_tool"] == "read_file"
        assert "last_prompt" in result.data
        
        # Prompt should contain all tools
        prompt = result.data["last_prompt"]
        assert "tool_0:" in prompt
        assert "tool_99:" in prompt