"""Unit tests for the PlannerAgent class.

This module contains comprehensive tests for the PlannerAgent implementation,
covering initialization, task decomposition, LLM integration, and fallback behavior.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from oniks.agents.planner_agent import PlannerAgent
from oniks.core.state import State
from oniks.llm.client import OllamaClient


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


class TestPlannerAgentExecution(unittest.TestCase):
    """Test cases for PlannerAgent execution logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
        self.state = State()
    
    def test_planner_agent_execute_with_goal(self):
        """Test PlannerAgent execution with a valid goal."""
        self.state.data['goal'] = "Create file hello.txt with Hello World and display it"
        
        # Mock LLM response
        mock_response = """1. Create a file named 'hello.txt' with the content 'Hello World'
2. Display the content of 'hello.txt' to the console"""
        self.mock_llm_client.invoke.return_value = mock_response
        
        result_state = self.agent.execute(self.state)
        
        # Verify plan was created
        self.assertIn('plan', result_state.data)
        plan = result_state.data['plan']
        self.assertIsInstance(plan, list)
        self.assertEqual(len(plan), 3)  # 2 tasks + confirmation
        self.assertEqual(plan[0], "Create a file named 'hello.txt' with the content 'Hello World'")
        self.assertEqual(plan[1], "Display the content of 'hello.txt' to the console")
        self.assertEqual(plan[2], "Confirm that all previous steps are complete")
        
        # Verify LLM was called
        self.mock_llm_client.invoke.assert_called_once()
        
        # Verify messages were added
        messages = result_state.message_history
        self.assertTrue(any("starting task decomposition" in msg for msg in messages))
        self.assertTrue(any("Created plan with 3 subtasks" in msg for msg in messages))
    
    def test_planner_agent_execute_without_goal(self):
        """Test PlannerAgent execution without goal in state."""
        # No goal in state data
        result_state = self.agent.execute(self.state)
        
        # Verify empty plan was created
        self.assertIn('plan', result_state.data)
        self.assertEqual(result_state.data['plan'], [])
        
        # Verify LLM was not called
        self.mock_llm_client.invoke.assert_not_called()
        
        # Verify appropriate message was added
        messages = result_state.message_history
        self.assertTrue(any("No goal found in state data" in msg for msg in messages))
    
    def test_planner_agent_execute_with_empty_goal(self):
        """Test PlannerAgent execution with empty goal."""
        self.state.data['goal'] = ""
        
        result_state = self.agent.execute(self.state)
        
        # Verify empty plan was created
        self.assertIn('plan', result_state.data)
        self.assertEqual(result_state.data['plan'], [])
        
        # Verify LLM was not called
        self.mock_llm_client.invoke.assert_not_called()
    
    def test_planner_agent_execute_preserves_original_state(self):
        """Test that execution preserves original state object."""
        original_goal = "Test goal"
        self.state.data['goal'] = original_goal
        self.state.add_message("Original message")
        
        # Mock LLM response
        self.mock_llm_client.invoke.return_value = "1. First task\n2. Second task"
        
        result_state = self.agent.execute(self.state)
        
        # Verify original state is unchanged
        self.assertEqual(self.state.data['goal'], original_goal)
        self.assertEqual(len(self.state.message_history), 1)
        self.assertNotIn('plan', self.state.data)
        
        # Verify result state is different object
        self.assertIsNot(result_state, self.state)
        self.assertIn('plan', result_state.data)
    
    def test_planner_agent_execute_with_llm_failure(self):
        """Test PlannerAgent execution when LLM fails."""
        self.state.data['goal'] = "Create hello.txt with Hello ONIKS! and display it"
        
        # Mock LLM failure
        self.mock_llm_client.invoke.side_effect = Exception("Connection error")
        
        result_state = self.agent.execute(self.state)
        
        # Verify fallback plan was created
        self.assertIn('plan', result_state.data)
        plan = result_state.data['plan']
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)
        
        # Verify fallback messages
        messages = result_state.message_history
        self.assertTrue(any("fallback" in msg.lower() for msg in messages))


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


class TestPlannerAgentFallbackDecomposition(unittest.TestCase):
    """Test cases for PlannerAgent fallback decomposition."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
    
    def test_perform_basic_decomposition_demo_case(self):
        """Test fallback decomposition for the demo case."""
        goal = "Create hello.txt with Hello ONIKS! and display it"
        
        tasks = self.agent._perform_basic_decomposition(goal)
        
        expected = [
            "Create a file named 'hello.txt' with the content 'Hello ONIKS!'",
            "Display the content of 'hello.txt' to the console",
            "Confirm that all previous steps are complete"
        ]
        self.assertEqual(tasks, expected)
    
    def test_perform_basic_decomposition_file_creation(self):
        """Test fallback decomposition for file creation."""
        goal = "Create a new file with some content"
        
        tasks = self.agent._perform_basic_decomposition(goal)
        
        self.assertGreater(len(tasks), 0)
        self.assertTrue(any("create" in task.lower() for task in tasks))
        self.assertEqual(tasks[-1], "Confirm that all previous steps are complete")
    
    def test_perform_basic_decomposition_file_reading(self):
        """Test fallback decomposition for file reading."""
        goal = "Read the contents of data.txt file"
        
        tasks = self.agent._perform_basic_decomposition(goal)
        
        expected = [
            "Read the specified file content",
            "Confirm that all previous steps are complete"
        ]
        self.assertEqual(tasks, expected)
    
    def test_perform_basic_decomposition_generic_goal(self):
        """Test fallback decomposition for generic goal."""
        goal = "Perform some complex operation"
        
        tasks = self.agent._perform_basic_decomposition(goal)
        
        expected = [
            "Execute the following goal: Perform some complex operation",
            "Confirm that all previous steps are complete"
        ]
        self.assertEqual(tasks, expected)
    
    def test_perform_basic_decomposition_with_none_goal(self):
        """Test fallback decomposition with None goal."""
        tasks = self.agent._perform_basic_decomposition(None)
        
        expected = [
            "Execute the following goal: None",
            "Confirm that all previous steps are complete" 
        ]
        self.assertEqual(tasks, expected)


class TestPlannerAgentIntegration(unittest.TestCase):
    """Integration test cases for PlannerAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.agent = PlannerAgent("planner", self.mock_llm_client)
    
    def test_full_execution_cycle_with_llm_success(self):
        """Test full execution cycle with successful LLM interaction."""
        state = State()
        state.data['goal'] = "Create config.ini and backup existing files"
        
        # Mock successful LLM response
        mock_response = """1. Create a file named 'config.ini' with configuration data
2. Create backup of existing files  
3. Verify all operations completed"""
        self.mock_llm_client.invoke.return_value = mock_response
        
        result_state = self.agent.execute(state)
        
        # Verify complete execution
        self.assertIn('plan', result_state.data)
        self.assertIn('decomposition_prompt', result_state.data)
        self.assertIn('decomposition_response', result_state.data)
        
        plan = result_state.data['plan']
        self.assertEqual(len(plan), 4)  # 3 tasks + confirmation
        self.assertEqual(plan[-1], "Confirm that all previous steps are complete")
        
        # Verify all expected messages
        messages = result_state.message_history
        self.assertTrue(any("starting task decomposition" in msg for msg in messages))
        self.assertTrue(any("Generated task decomposition prompt" in msg for msg in messages))
        self.assertTrue(any("Successfully received decomposition from LLM" in msg for msg in messages))
        self.assertTrue(any("Created plan with 4 subtasks" in msg for msg in messages))
        self.assertTrue(any("completed task decomposition" in msg for msg in messages))
    
    def test_full_execution_cycle_with_llm_failure(self):
        """Test full execution cycle with LLM failure and fallback."""
        state = State()
        state.data['goal'] = "Create hello.txt with Hello ONIKS! and display its content"
        
        # Mock LLM failure
        self.mock_llm_client.invoke.side_effect = Exception("Network error")
        
        result_state = self.agent.execute(state)
        
        # Verify fallback execution
        self.assertIn('plan', result_state.data)
        self.assertIn('decomposition_prompt', result_state.data)
        
        plan = result_state.data['plan']
        self.assertGreater(len(plan), 0)
        
        # Verify fallback messages
        messages = result_state.message_history
        self.assertTrue(any("Task decomposition failed" in msg for msg in messages))
        self.assertTrue(any("Falling back to basic decomposition" in msg for msg in messages))
        self.assertTrue(any("Created fallback plan" in msg for msg in messages))


if __name__ == '__main__':
    unittest.main()