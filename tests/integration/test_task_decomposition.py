"""Integration tests for the Task Decomposition Layer.

This module contains comprehensive integration tests for the Task Decomposition Layer,
testing the complete workflow from goal decomposition through task execution.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.core.checkpoint import SQLiteCheckpointSaver
from oniks.agents.planner_agent import PlannerAgent
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.tools.fs_tools import WriteFileTool
from oniks.tools.shell_tools import ExecuteBashCommandTool
from oniks.tools.core_tools import TaskCompleteTool
from oniks.llm.client import OllamaClient
from run_task_decomposition_test import TaskDecompositionGraph


class TestTaskDecompositionIntegration(unittest.TestCase):
    """Integration tests for the complete Task Decomposition Layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_llm_client = Mock(spec=OllamaClient)
        
        # Create agents
        self.planner_agent = PlannerAgent("planner_agent", self.mock_llm_client)
        self.reasoning_agent = ReasoningAgent("reasoning_agent", [], self.mock_llm_client)
        
        # Create tools
        self.write_file_tool = WriteFileTool()
        self.execute_bash_tool = ExecuteBashCommandTool()
        self.task_complete_tool = TaskCompleteTool()
        
        tools = [self.write_file_tool, self.execute_bash_tool, self.task_complete_tool]
        self.reasoning_agent.tools = tools
        
        # Create tool nodes
        self.write_file_node = ToolNode("write_file", self.write_file_tool)
        self.execute_bash_node = ToolNode("execute_bash_command", self.execute_bash_tool)
        self.task_complete_node = ToolNode("task_complete", self.task_complete_tool)
        
        # Create graph
        self.graph = TaskDecompositionGraph(
            reusable_nodes={"planner_agent", "reasoning_agent"}
        )
        
        # Add nodes to graph
        self.graph.add_node(self.planner_agent)
        self.graph.add_node(self.reasoning_agent)
        self.graph.add_node(self.write_file_node)
        self.graph.add_node(self.execute_bash_node)
        self.graph.add_node(self.task_complete_node)
        
        # Add edges
        self._setup_graph_edges()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_graph_edges(self):
        """Set up graph edges for task decomposition workflow."""
        # Planner to reasoning agent
        self.graph.add_edge(
            "planner_agent", 
            "reasoning_agent",
            condition=lambda state: True
        )
        
        # Reasoning agent to tool nodes
        self.graph.add_edge(
            "reasoning_agent", 
            "write_file",
            condition=lambda state: state.data.get('next_tool') == 'write_file'
        )
        
        self.graph.add_edge(
            "reasoning_agent", 
            "execute_bash_command",
            condition=lambda state: state.data.get('next_tool') == 'execute_bash_command'
        )
        
        self.graph.add_edge(
            "reasoning_agent", 
            "task_complete",
            condition=lambda state: state.data.get('next_tool') == 'task_complete'
        )
        
        # Tool nodes back to reasoning agent
        self.graph.add_edge(
            "write_file",
            "reasoning_agent",
            condition=lambda state: True
        )
        
        self.graph.add_edge(
            "execute_bash_command",
            "reasoning_agent",
            condition=lambda state: True
        )
    
    def test_full_task_decomposition_workflow_with_llm_success(self):
        """Test complete task decomposition workflow with successful LLM interactions."""
        # Set up initial state
        initial_state = State()
        initial_state.data['goal'] = "Create a file named 'test.txt' with content 'Hello World', then display its content"
        
        # Mock LLM responses
        decomposition_response = """1. Create a file named 'test.txt' with the content 'Hello World'
2. Display the content of 'test.txt' to the console"""
        
        reasoning_responses = [
            "Tool: write_file\nArguments: {\"file_path\": \"test.txt\", \"content\": \"Hello World\"}",
            "Tool: execute_bash_command\nArguments: {\"command\": \"cat test.txt\"}"
        ]
        
        self.mock_llm_client.invoke.side_effect = [decomposition_response] + reasoning_responses
        
        # Change to temp directory for file operations
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Execute the graph
            final_state = self.graph.execute(
                initial_state=initial_state,
                thread_id="test_integration_001",
                start_node="planner_agent",
                max_iterations=15
            )
            
            # Verify task decomposition
            self.assertIn('plan', final_state.data)
            plan = final_state.data['plan']
            self.assertEqual(len(plan), 0)  # All tasks should be removed
            
            # Verify tool executions
            self.assertIn('write_file', final_state.tool_outputs)
            self.assertIn('execute_bash_command', final_state.tool_outputs)
            self.assertIn('task_complete', final_state.tool_outputs)
            
            # Verify file was created
            test_file = Path(self.temp_dir) / "test.txt"
            self.assertTrue(test_file.exists())
            with open(test_file, 'r') as f:
                content = f.read().strip()
            self.assertEqual(content, "Hello World")
            
            # Verify file content was displayed
            bash_output = final_state.tool_outputs['execute_bash_command']
            self.assertIn("Hello World", bash_output)
            
            # Verify task completion
            task_output = final_state.tool_outputs['task_complete']
            self.assertEqual(task_output, "Task finished successfully.")
            
        finally:
            os.chdir(original_cwd)
    
    def test_full_task_decomposition_workflow_with_llm_fallback(self):
        """Test complete workflow with LLM fallback to basic reasoning."""
        # Set up initial state with demo case goal
        initial_state = State()
        initial_state.data['goal'] = "Create hello.txt with Hello ONIKS! and display its content to the console"
        
        # Mock LLM failures
        self.mock_llm_client.invoke.side_effect = Exception("Connection error")
        
        # Change to temp directory for file operations
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Execute the graph
            final_state = self.graph.execute(
                initial_state=initial_state,
                thread_id="test_fallback_001",
                start_node="planner_agent",
                max_iterations=15
            )
            
            # Verify fallback decomposition worked
            self.assertIn('plan', final_state.data)
            plan = final_state.data['plan']
            self.assertEqual(len(plan), 0)  # All tasks should be removed
            
            # Verify tool executions
            self.assertIn('write_file', final_state.tool_outputs)
            self.assertIn('execute_bash_command', final_state.tool_outputs)
            self.assertIn('task_complete', final_state.tool_outputs)
            
            # Verify file was created
            hello_file = Path(self.temp_dir) / "hello.txt"
            self.assertTrue(hello_file.exists())
            with open(hello_file, 'r') as f:
                content = f.read().strip()
            self.assertEqual(content, "Hello ONIKS!")
            
            # Verify file content was displayed
            bash_output = final_state.tool_outputs['execute_bash_command']
            self.assertIn("Hello ONIKS!", bash_output)
            
        finally:
            os.chdir(original_cwd)
    
    def test_task_removal_mechanism(self):
        """Test that tasks are properly removed from plan after execution."""
        # Set up initial state
        initial_state = State()
        initial_state.data['goal'] = "Create simple.txt with Simple Content"
        
        # Mock simplified LLM response
        self.mock_llm_client.invoke.side_effect = [
            "1. Create a file named 'simple.txt' with the content 'Simple Content'",
            "Tool: write_file\nArguments: {\"file_path\": \"simple.txt\", \"content\": \"Simple Content\"}"
        ]
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Execute the graph
            final_state = self.graph.execute(
                initial_state=initial_state,
                thread_id="test_removal_001",
                start_node="planner_agent",
                max_iterations=10
            )
            
            # Verify plan is empty (all tasks removed)
            plan = final_state.data.get('plan', [])
            self.assertEqual(len(plan), 0)
            
            # Verify task removal messages in history
            messages = final_state.message_history
            removal_messages = [msg for msg in messages if "Removed completed task from plan" in msg]
            self.assertGreater(len(removal_messages), 0)
            
        finally:
            os.chdir(original_cwd)
    
    def test_deterministic_completion_detection(self):
        """Test that final confirmation task triggers task_complete."""
        # Set up state with only confirmation task
        test_state = State()
        test_state.data['plan'] = ["Confirm that all previous steps are complete"]
        
        # Mock LLM client (shouldn't be called for confirmation task)
        self.mock_llm_client.invoke.return_value = "Should not be used"
        
        # Execute reasoning agent
        result_state = self.reasoning_agent.execute(test_state)
        
        # Verify task_complete was selected
        self.assertEqual(result_state.data.get('next_tool'), 'task_complete')
        self.assertEqual(result_state.data.get('tool_args'), {})
        
        # Verify LLM was not called
        self.mock_llm_client.invoke.assert_not_called()
    
    def test_graph_reusability_of_agents(self):
        """Test that planner and reasoning agents can be re-executed multiple times."""
        # Set up initial state
        initial_state = State()
        initial_state.data['goal'] = "Create two files: first.txt and second.txt"
        
        # Mock LLM responses for multiple executions
        self.mock_llm_client.invoke.side_effect = [
            "1. Create a file named 'first.txt'\n2. Create a file named 'second.txt'",
            "Tool: write_file\nArguments: {\"file_path\": \"first.txt\", \"content\": \"First\"}",
            "Tool: write_file\nArguments: {\"file_path\": \"second.txt\", \"content\": \"Second\"}"
        ]
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Execute the graph
            final_state = self.graph.execute(
                initial_state=initial_state,
                thread_id="test_reusability_001",
                start_node="planner_agent",
                max_iterations=20
            )
            
            # Verify both reasoning agent executions occurred
            messages = final_state.message_history
            reasoning_executions = [msg for msg in messages if "Reasoning agent reasoning_agent starting analysis" in msg]
            self.assertGreaterEqual(len(reasoning_executions), 2)
            
            # Verify planner was executed once
            planner_executions = [msg for msg in messages if "Planner agent planner_agent starting task decomposition" in msg]
            self.assertEqual(len(planner_executions), 1)
            
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling_in_task_decomposition(self):
        """Test error handling when tool execution fails."""
        # Set up initial state
        initial_state = State()
        initial_state.data['goal'] = "Create a file in invalid location"
        
        # Mock LLM response that will cause tool error
        self.mock_llm_client.invoke.side_effect = [
            "1. Create a file in invalid location",
            "Tool: write_file\nArguments: {\"file_path\": \"/invalid/path/file.txt\", \"content\": \"Content\"}"
        ]
        
        # Execute the graph
        final_state = self.graph.execute(
            initial_state=initial_state,
            thread_id="test_error_001",
            start_node="planner_agent",
            max_iterations=10
        )
        
        # Verify error was handled
        self.assertIn('write_file', final_state.tool_outputs)
        tool_output = final_state.tool_outputs['write_file']
        self.assertIn("Error", tool_output)
        
        # Verify plan was still processed (error doesn't break the flow)
        plan = final_state.data.get('plan', [])
        # Plan should be empty if task was removed even with error,
        # or have remaining tasks if error prevented removal
        self.assertIsInstance(plan, list)
    
    def test_checkpoint_integration_with_task_decomposition(self):
        """Test checkpoint saving and loading with task decomposition."""
        # Create temporary checkpoint file
        checkpoint_file = Path(self.temp_dir) / "test_checkpoints.db"
        checkpointer = SQLiteCheckpointSaver(str(checkpoint_file))
        
        # Create graph with checkpointer
        graph_with_checkpoint = TaskDecompositionGraph(
            checkpointer=checkpointer,
            reusable_nodes={"planner_agent", "reasoning_agent"}
        )
        
        # Add nodes and edges (same as before)
        graph_with_checkpoint.add_node(self.planner_agent)
        graph_with_checkpoint.add_node(self.reasoning_agent)
        graph_with_checkpoint.add_node(self.write_file_node)
        graph_with_checkpoint.add_node(self.task_complete_node)
        
        graph_with_checkpoint.add_edge("planner_agent", "reasoning_agent", condition=lambda state: True)
        graph_with_checkpoint.add_edge("reasoning_agent", "write_file", condition=lambda state: state.data.get('next_tool') == 'write_file')
        graph_with_checkpoint.add_edge("reasoning_agent", "task_complete", condition=lambda state: state.data.get('next_tool') == 'task_complete')
        graph_with_checkpoint.add_edge("write_file", "reasoning_agent", condition=lambda state: True)
        
        # Set up initial state
        initial_state = State()
        initial_state.data['goal'] = "Simple checkpoint test"
        
        # Mock LLM responses
        self.mock_llm_client.invoke.side_effect = [
            "1. Simple task",
            "Tool: write_file\nArguments: {\"file_path\": \"checkpoint_test.txt\", \"content\": \"Test\"}"
        ]
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Execute the graph
            final_state = graph_with_checkpoint.execute(
                initial_state=initial_state,
                thread_id="checkpoint_test_001",
                start_node="planner_agent",
                max_iterations=10
            )
            
            # Verify checkpoint file was created
            self.assertTrue(checkpoint_file.exists())
            
            # Verify checkpoint messages in history
            messages = final_state.message_history
            checkpoint_messages = [msg for msg in messages if "checkpoint" in msg.lower()]
            self.assertGreater(len(checkpoint_messages), 0)
            
            # Test loading checkpoint
            loaded_state = graph_with_checkpoint.load_checkpoint("checkpoint_test_001")
            self.assertIsNotNone(loaded_state)
            self.assertIn('goal', loaded_state.data)
            
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()