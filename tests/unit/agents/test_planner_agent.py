"""Unit tests for the PlannerAgent class.

This module contains comprehensive tests for the PlannerAgent implementation,
covering initialization, task decomposition, and STRICT LLM-ONLY operation.

CRITICAL: These tests verify that PlannerAgent operates in STRICT LLM-ONLY mode
with NO FALLBACKS. Any LLM failure must result in LLMUnavailableError.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from oniks.agents.planner_agent import PlannerAgent
from oniks.core.state import State
from oniks.core.exceptions import LLMUnavailableError
from oniks.llm.client import OllamaClient, OllamaConnectionError


class TestPlannerAgentInitialization(unittest.TestCase):
    """Test cases for PlannerAgent initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
    
    def test_planner_agent_initialization_success(self):
        """Test successful initialization of PlannerAgent."""
        agent = PlannerAgent("planner", self.mock_llm_client)
        
        self.assertEqual(agent.name, "planner")
        self.assertEqual(agent.llm_client, self.mock_llm_client)
    
    def test_planner_agent_initialization_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PlannerAgent("", self.mock_llm_client)
        
        self.assertIn("Node name cannot be empty", str(context.exception))
    
    def test_planner_agent_initialization_none_name_raises_error(self):
        """Test that None name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PlannerAgent(None, self.mock_llm_client)
        
        self.assertIn("Node name cannot be empty", str(context.exception))
    
    def test_planner_agent_initialization_none_llm_client_raises_error(self):
        """Test that None LLM client raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PlannerAgent("planner", None)
        
        self.assertIn("LLM client cannot be None", str(context.exception))
    
    def test_planner_agent_string_representation(self):
        """Test string representation of PlannerAgent."""
        agent = PlannerAgent("test_planner", self.mock_llm_client)
        
        expected = "PlannerAgent(name='test_planner')"
        self.assertEqual(str(agent), expected)
        self.assertEqual(repr(agent), expected)


class TestPlannerAgentStrictLLMOnlyExecution(unittest.TestCase):
    """Test cases for PlannerAgent STRICT LLM-ONLY execution logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
        self.state = State()
    
    def test_planner_agent_execute_with_valid_goal_and_successful_llm(self):
        """Test PlannerAgent execution with valid goal and successful LLM response."""
        self.state.data['goal'] = "Create file hello.txt with Hello World and display it"
        
        # Mock successful LLM response with tool calls
        mock_response = """1. write_file(file_path='hello.txt', content='Hello World')
2. execute_bash_command(command='cat hello.txt')"""
        self.mock_llm_client.invoke.return_value = mock_response
        
        result_state = self.agent.execute(self.state)
        
        # Verify LLM-generated plan was created
        self.assertIn('plan', result_state.data)
        plan = result_state.data['plan']
        self.assertIsInstance(plan, list)
        self.assertEqual(len(plan), 3)  # 2 LLM tool calls + task_complete()
        self.assertEqual(plan[0], "write_file(file_path='hello.txt', content='Hello World')")
        self.assertEqual(plan[1], "execute_bash_command(command='cat hello.txt')")
        self.assertEqual(plan[2], "task_complete()")
        
        # Verify LLM was called
        self.mock_llm_client.invoke.assert_called_once()
        
        # Verify [LLM-POWERED] messages were added
        messages = result_state.message_history
        self.assertTrue(any("STRICT LLM-ONLY task decomposition" in msg for msg in messages))
        self.assertTrue(any("[LLM-POWERED]" in msg for msg in messages))
        self.assertFalse(any("[LLM-ERROR]" in msg for msg in messages))
        self.assertFalse(any("fallback" in msg.lower() for msg in messages))
    
    def test_planner_agent_execute_without_goal_throws_error(self):
        """Test PlannerAgent execution without goal throws LLMUnavailableError."""
        # No goal in state data - MUST fail in strict mode
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("Goal validation failed", error.message)
        self.assertIsNotNone(error.correlation_id)
        self.assertIn("agent_execution_id", error.request_details)
        
        # Verify LLM was not called
        self.mock_llm_client.invoke.assert_not_called()
    
    def test_planner_agent_execute_with_empty_goal_throws_error(self):
        """Test PlannerAgent execution with empty goal throws LLMUnavailableError."""
        self.state.data['goal'] = ""  # Empty goal - MUST fail in strict mode
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("Goal validation failed", error.message)
        
        # Verify LLM was not called
        self.mock_llm_client.invoke.assert_not_called()
    
    def test_planner_agent_execute_with_whitespace_only_goal_throws_error(self):
        """Test PlannerAgent execution with whitespace-only goal throws LLMUnavailableError."""
        self.state.data['goal'] = "   \n\t   "  # Whitespace only - MUST fail
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("Goal validation failed", error.message)
    
    def test_planner_agent_execute_preserves_original_state_on_success(self):
        """Test that successful execution preserves original state object."""
        original_goal = "write_file(file_path='test.txt', content='test')"
        self.state.data['goal'] = original_goal
        self.state.add_message("Original message")
        
        # Mock successful LLM response
        self.mock_llm_client.invoke.return_value = "1. write_file(file_path='test.txt', content='test')"
        
        result_state = self.agent.execute(self.state)
        
        # Verify original state is unchanged
        self.assertEqual(self.state.data['goal'], original_goal)
        self.assertEqual(len(self.state.message_history), 1)
        self.assertNotIn('plan', self.state.data)
        
        # Verify result state is different object with LLM-generated plan
        self.assertIsNot(result_state, self.state)
        self.assertIn('plan', result_state.data)
    
    def test_planner_agent_execute_with_llm_connection_error_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError when LLM connection fails."""
        self.state.data['goal'] = "Create hello.txt with Hello ONIKS! and display it"
        
        # Mock LLM connection failure
        self.mock_llm_client.invoke.side_effect = OllamaConnectionError("Connection refused")
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM decomposition failed", error.message)
        self.assertIsInstance(error.original_error, OllamaConnectionError)
        self.assertIn("agent_execution_id", error.request_details)
        self.assertIn("goal", error.request_details)
    
    def test_planner_agent_execute_with_llm_generic_error_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError for any LLM error."""
        self.state.data['goal'] = "Valid goal"
        
        # Mock generic LLM failure
        self.mock_llm_client.invoke.side_effect = Exception("Unexpected error")
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM decomposition failed", error.message)
        self.assertIsInstance(error.original_error, Exception)
        self.assertEqual(str(error.original_error), "Unexpected error")
    
    def test_planner_agent_execute_with_empty_llm_response_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError when LLM returns empty response."""
        self.state.data['goal'] = "Valid goal"
        
        # Mock empty LLM response
        self.mock_llm_client.invoke.return_value = ""
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM response validation failed", error.message)
        self.assertIn("empty or invalid content", error.message)
    
    def test_planner_agent_execute_with_whitespace_only_llm_response_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError when LLM returns whitespace-only response."""
        self.state.data['goal'] = "Valid goal"
        
        # Mock whitespace-only LLM response
        self.mock_llm_client.invoke.return_value = "   \n\t   "
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM response validation failed", error.message)
    
    def test_planner_agent_execute_with_non_string_llm_response_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError when LLM returns non-string response."""
        self.state.data['goal'] = "Valid goal"
        
        # Mock non-string LLM response
        self.mock_llm_client.invoke.return_value = {"invalid": "response"}
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM response validation failed", error.message)
    
    def test_planner_agent_execute_with_unparseable_llm_response_throws_error(self):
        """Test PlannerAgent throws LLMUnavailableError when LLM response has no valid tool calls."""
        self.state.data['goal'] = "Valid goal"
        
        # Mock LLM response with no parseable tool calls
        self.mock_llm_client.invoke.return_value = "This is just text without any tool calls or numbered items."
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(self.state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM response parsing failed", error.message)
        self.assertIn("no valid tool calls extracted", error.message)


class TestPlannerAgentPromptGeneration(unittest.TestCase):
    """Test cases for PlannerAgent prompt generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
    
    def test_generate_decomposition_prompt(self):
        """Test decomposition prompt generation."""
        goal = "Create a file and display its content"
        
        prompt = self.agent._generate_decomposition_prompt(goal)
        
        # Verify prompt contains key sections
        self.assertIn("TASK DECOMPOSITION REQUEST", prompt)
        self.assertIn("GOAL TO DECOMPOSE", prompt)
        self.assertIn(goal, prompt)
        self.assertIn("DECOMPOSITION RULES", prompt)
        self.assertIn("OUTPUT FORMAT", prompt)
        self.assertIn("EXAMPLES", prompt)
    
    def test_generate_decomposition_prompt_with_complex_goal(self):
        """Test prompt generation with complex goal."""
        goal = "Read configuration from config.json, process the data, and save results to output.csv"
        
        prompt = self.agent._generate_decomposition_prompt(goal)
        
        self.assertIn(goal, prompt)
        self.assertIn("atomic", prompt.lower())
        self.assertIn("specific", prompt.lower())


class TestPlannerAgentResponseParsing(unittest.TestCase):
    """Test cases for PlannerAgent response parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
    
    def test_parse_decomposition_response_numbered_list(self):
        """Test parsing numbered list response."""
        response = """1. Create a file named 'test.txt' with content 'Hello'
2. Display the content of 'test.txt'
3. Delete the file 'test.txt'"""
        
        tasks = self.agent._parse_decomposition_response(response)
        
        expected = [
            "Create a file named 'test.txt' with content 'Hello'",
            "Display the content of 'test.txt'",
            "Delete the file 'test.txt'"
        ]
        self.assertEqual(tasks, expected)
    
    def test_parse_decomposition_response_bullet_list(self):
        """Test parsing bullet list response."""
        response = """- Create a new file
- Write some content
- Save the file"""
        
        tasks = self.agent._parse_decomposition_response(response)
        
        expected = [
            "Create a new file",
            "Write some content", 
            "Save the file"
        ]
        self.assertEqual(tasks, expected)
    
    def test_parse_decomposition_response_mixed_format(self):
        """Test parsing mixed format response."""
        response = """1. First task here
- Second task with bullet
3. Third numbered task
* Fourth task with asterisk"""
        
        tasks = self.agent._parse_decomposition_response(response)
        
        expected = [
            "First task here",
            "Second task with bullet",
            "Third numbered task",
            "Fourth task with asterisk"
        ]
        self.assertEqual(tasks, expected)
    
    def test_parse_decomposition_response_with_empty_lines(self):
        """Test parsing response with empty lines."""
        response = """
1. First task

2. Second task


3. Third task
"""
        
        tasks = self.agent._parse_decomposition_response(response)
        
        expected = [
            "First task",
            "Second task",
            "Third task"
        ]
        self.assertEqual(tasks, expected)
    
    def test_parse_decomposition_response_empty_or_invalid(self):
        """Test parsing empty or invalid response."""
        # Empty response
        tasks = self.agent._parse_decomposition_response("")
        self.assertEqual(tasks, [])
        
        # None response
        tasks = self.agent._parse_decomposition_response(None)
        self.assertEqual(tasks, [])
        
        # No actionable content
        tasks = self.agent._parse_decomposition_response("This is just text without tasks")
        self.assertEqual(tasks, [])




class TestPlannerAgentStrictLLMOnlyIntegration(unittest.TestCase):
    """STRICT LLM-ONLY integration test cases for PlannerAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
    
    def test_full_execution_cycle_with_successful_llm_tool_calls(self):
        """Test full execution cycle with successful LLM interaction returning tool calls."""
        state = State()
        state.data['goal'] = "Create config.ini and backup existing files"
        
        # Mock successful LLM response with valid tool calls
        mock_response = """1. write_file(file_path='config.ini', content='[default]key=value')
2. execute_bash_command(command='cp *.txt backup/')
3. list_files(path='.')"""
        self.mock_llm_client.invoke.return_value = mock_response
        
        result_state = self.agent.execute(state)
        
        # Verify complete LLM-powered execution
        self.assertIn('plan', result_state.data)
        self.assertIn('decomposition_prompt', result_state.data)
        self.assertIn('decomposition_response', result_state.data)
        
        plan = result_state.data['plan']
        self.assertEqual(len(plan), 4)  # 3 LLM tool calls + task_complete()
        self.assertEqual(plan[0], "write_file(file_path='config.ini', content='[default]key=value')")
        self.assertEqual(plan[1], "execute_bash_command(command='cp *.txt backup/')")
        self.assertEqual(plan[2], "list_files(path='.')")
        self.assertEqual(plan[3], "task_complete()")
        
        # Verify all expected [LLM-POWERED] messages
        messages = result_state.message_history
        self.assertTrue(any("STRICT LLM-ONLY task decomposition" in msg for msg in messages))
        self.assertTrue(any("Generated task decomposition prompt for LLM" in msg for msg in messages))
        self.assertTrue(any("[LLM-POWERED] Successfully received decomposition from LLM" in msg for msg in messages))
        self.assertTrue(any("[LLM-POWERED] Created tool-based plan with 4 steps" in msg for msg in messages))
        self.assertTrue(any("completed STRICT LLM-ONLY task decomposition" in msg for msg in messages))
        
        # Ensure NO fallback or error messages
        self.assertFalse(any("fallback" in msg.lower() for msg in messages))
        self.assertFalse(any("[LLM-ERROR]" in msg for msg in messages))
    
    def test_full_execution_cycle_with_llm_failure_throws_error(self):
        """Test full execution cycle with LLM failure throws LLMUnavailableError - NO FALLBACK."""
        state = State()
        state.data['goal'] = "Create hello.txt with Hello ONIKS! and display its content"
        
        # Mock LLM failure - should result in LLMUnavailableError
        self.mock_llm_client.invoke.side_effect = Exception("Network error")
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM decomposition failed", error.message)
        self.assertIn("Network error", str(error.original_error))
        self.assertIn("goal", error.request_details)
        self.assertEqual(error.request_details["goal"], "Create hello.txt with Hello ONIKS! and display its content")
    
    def test_full_execution_cycle_with_invalid_llm_response_throws_error(self):
        """Test full execution cycle with invalid LLM response throws LLMUnavailableError."""
        state = State()
        state.data['goal'] = "Valid goal for testing"
        
        # Mock invalid LLM response (no tool calls)
        self.mock_llm_client.invoke.return_value = "I cannot help with that request."
        
        with self.assertRaises(LLMUnavailableError) as context:
            self.agent.execute(state)
        
        # Verify error details
        error = context.exception
        self.assertIn("LLM response parsing failed", error.message)
        self.assertIn("no valid tool calls extracted", error.message)
        self.assertIn("raw_response", error.request_details)
    
    def test_correlation_id_consistency_across_error_scenarios(self):
        """Test that correlation IDs are consistent across different error scenarios."""
        state = State()
        state.data['goal'] = "Test goal"
        
        # Test 1: LLM connection error
        self.mock_llm_client.invoke.side_effect = OllamaConnectionError("Connection error")
        
        with self.assertRaises(LLMUnavailableError) as context1:
            self.agent.execute(state)
        
        error1 = context1.exception
        self.assertIsNotNone(error1.correlation_id)
        self.assertIn("agent_execution_id", error1.request_details)
        self.assertEqual(error1.correlation_id, error1.request_details["agent_execution_id"])
        
        # Test 2: Empty LLM response
        self.mock_llm_client.invoke.side_effect = None
        self.mock_llm_client.invoke.return_value = ""
        
        with self.assertRaises(LLMUnavailableError) as context2:
            self.agent.execute(state)
        
        error2 = context2.exception
        self.assertIsNotNone(error2.correlation_id)
        self.assertNotEqual(error1.correlation_id, error2.correlation_id)  # Different execution IDs


if __name__ == '__main__':
    unittest.main()