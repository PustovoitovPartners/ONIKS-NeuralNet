"""Integration tests for the complete ONIKS NeuralNet system.

This module contains integration tests that verify the end-to-end functionality
of the framework by testing the interaction between all components:
Graph, Nodes, Agents, Tools, State, and CheckpointSaver.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.core.checkpoint import SQLiteCheckpointSaver
from oniks.tools.file_tools import ReadFileTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.agents.base import BaseAgent
from oniks.llm.client import OllamaClient


class TestFullSystemIntegration:
    """Test complete system integration scenarios."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for checkpointing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        yield db_path
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for integration testing."""
        client = Mock(spec=OllamaClient)
        client.invoke.return_value = (
            "Tool: read_file\n"
            "Arguments: {\"file_path\": \"task.txt\"}\n"
            "Reasoning: The goal requires reading a file"
        )
        return client
        
        # Cleanup
        path = Path(db_path)
        if path.exists():
            path.unlink()
    
    @pytest.fixture
    def test_file_content(self):
        """Create a temporary test file with content."""
        content = "This is a test file for integration testing.\nLine 2: More content here.\nLine 3: Final line.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        yield temp_path, content
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_full_system_workflow_like_run_reasoning_test(self, temp_db_path, test_file_content, mock_llm_client):
        """Test complete workflow similar to run_reasoning_test.py."""
        test_file_path, expected_content = test_file_content
        
        # 1. Initialize graph with checkpoint saver
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        graph = Graph(checkpointer=checkpointer)
        
        # 2. Create initial state with goal
        initial_state = State()
        initial_state.data['goal'] = 'Прочитать содержимое файла task.txt'
        initial_state.add_message("Integration test started with file reading goal")
        
        # 3. Create tools
        read_file_tool = ReadFileTool()
        
        # 4. Create agents and nodes
        reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        file_reader_node = ToolNode("file_reader", read_file_tool)
        
        # 5. Build graph structure
        graph.add_node(reasoning_agent)
        graph.add_node(file_reader_node)
        
        # Add edges with conditions
        graph.add_edge(
            "reasoning_agent", 
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        # Modify the agent to use our test file instead of hardcoded task.txt
        original_reasoning = reasoning_agent._perform_basic_reasoning
        
        def modified_reasoning(goal, state):
            original_reasoning(goal, state)
            if state.data.get('next_tool') == 'read_file':
                state.data['file_path'] = test_file_path  # Use our test file
                state.data['tool_args'] = {'file_path': test_file_path}
        
        reasoning_agent._perform_basic_reasoning = modified_reasoning
        
        # 6. Execute the graph
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="integration_test_thread_001",
            start_node="reasoning_agent",
            max_iterations=10
        )
        
        # 7. Verify results
        
        # Check that both nodes were executed
        node_execution_messages = [msg for msg in final_state.message_history if "Executing node:" in msg]
        assert len(node_execution_messages) == 2
        assert any("reasoning_agent" in msg for msg in node_execution_messages)
        assert any("file_reader" in msg for msg in node_execution_messages)
        
        # Check that reasoning agent worked correctly
        assert "last_prompt" in final_state.data
        assert final_state.data["next_tool"] == "read_file"
        assert final_state.data["file_path"] == test_file_path
        
        # Check that file was read successfully
        assert "read_file" in final_state.tool_outputs
        assert final_state.tool_outputs["read_file"] == expected_content
        
        # Check that tool execution was recorded
        tool_execution_messages = [msg for msg in final_state.message_history if "Executing tool:" in msg]
        assert len(tool_execution_messages) == 1
        assert "read_file" in tool_execution_messages[0]
        
        # Check checkpoint operations
        checkpoint_messages = [msg for msg in final_state.message_history if "checkpoint" in msg.lower()]
        assert len(checkpoint_messages) > 0  # Should have checkpoint save messages
        
        # Verify checkpointer saved states
        saved_checkpoints = checkpointer.list_checkpoints()
        assert "integration_test_thread_001" in saved_checkpoints
        
        # Load and verify saved state
        loaded_state = checkpointer.load("integration_test_thread_001")
        assert loaded_state is not None
        assert loaded_state.data["goal"] == initial_state.data["goal"]
    
    def test_system_with_multiple_reasoning_cycles(self, temp_db_path, test_file_content, mock_llm_client):
        """Test system with multiple reasoning and tool execution cycles."""
        test_file_path, expected_content = test_file_content
        
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        graph = Graph(checkpointer=checkpointer)
        
        # Create initial state
        initial_state = State()
        initial_state.data['goal'] = 'Прочитать содержимое файла и проанализировать'
        initial_state.data['cycle_count'] = 0
        
        # Create tools and agents
        read_file_tool = ReadFileTool()
        reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        file_reader_node = ToolNode("file_reader", read_file_tool)
        
        # Add nodes to graph
        graph.add_node(reasoning_agent)
        graph.add_node(file_reader_node)
        
        # Create analyzer node for second phase
        class AnalyzerAgent(BaseAgent):
            def execute(self, state):
                result_state = state.model_copy(deep=True)
                result_state.add_message("Analyzing file content")
                
                content = result_state.tool_outputs.get('read_file', '')
                line_count = len(content.split('\n')) if content else 0
                result_state.data['analysis'] = {
                    'line_count': line_count,
                    'character_count': len(content),
                    'has_content': bool(content.strip())
                }
                result_state.data['cycle_count'] = 2
                result_state.add_message("Analysis completed")
                return result_state
        
        analyzer_agent = AnalyzerAgent("analyzer_agent")
        graph.add_node(analyzer_agent)
        
        # Custom reasoning logic for file reading only
        original_reasoning = reasoning_agent._perform_basic_reasoning
        
        def file_reading_reasoning(goal, state):
            if "прочитать" in goal.lower() and "файл" in goal.lower():
                state.data['next_tool'] = 'read_file'
                state.data['file_path'] = test_file_path
                state.data['tool_args'] = {'file_path': test_file_path}
                state.data['cycle_count'] = 1
                state.add_message("First cycle: Reading file")
        
        reasoning_agent._perform_basic_reasoning = file_reading_reasoning
        
        # Add edges for multi-step execution
        graph.add_edge(
            "reasoning_agent", 
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        graph.add_edge(
            "file_reader",
            "analyzer_agent"
        )
        
        # Execute graph
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="multi_cycle_test",
            start_node="reasoning_agent",
            max_iterations=10
        )
        
        # Verify multi-cycle execution
        assert final_state.data['cycle_count'] == 2
        assert 'analysis' in final_state.data
        assert final_state.data['analysis']['has_content'] is True
        assert final_state.data['analysis']['line_count'] > 0
        
        # Verify both phases occurred
        phase_messages = [msg for msg in final_state.message_history if "cycle:" in msg.lower() or "Analyzing" in msg]
        assert len(phase_messages) >= 2
        assert any("First cycle" in msg for msg in final_state.message_history)
        assert any("Analyzing file content" in msg for msg in final_state.message_history)
        
        # Verify file was read
        assert "read_file" in final_state.tool_outputs
        assert final_state.tool_outputs["read_file"] == expected_content
    
    def test_system_error_handling_integration(self, temp_db_path, mock_llm_client):
        """Test system-wide error handling and recovery."""
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        graph = Graph(checkpointer=checkpointer)
        
        # Create state with goal pointing to non-existent file
        initial_state = State()
        initial_state.data['goal'] = 'Прочитать содержимое файла non_existent.txt'
        
        # Create tools and agents
        read_file_tool = ReadFileTool()
        reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        file_reader_node = ToolNode("file_reader", read_file_tool)
        
        graph.add_node(reasoning_agent)
        graph.add_node(file_reader_node)
        
        # Modify reasoning to point to non-existent file
        original_reasoning = reasoning_agent._perform_basic_reasoning
        
        def error_reasoning(goal, state):
            if "прочитать" in goal.lower() and "файл" in goal.lower():
                state.data['next_tool'] = 'read_file'
                state.data['file_path'] = '/this/file/does/not/exist.txt'
                state.data['tool_args'] = {'file_path': '/this/file/does/not/exist.txt'}
                state.add_message("Attempting to read non-existent file")
        
        reasoning_agent._perform_basic_reasoning = error_reasoning
        
        graph.add_edge(
            "reasoning_agent", 
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        # Execute graph - should handle error gracefully
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="error_handling_test",
            start_node="reasoning_agent",
            max_iterations=10
        )
        
        # Verify error was handled gracefully
        assert "read_file" in final_state.tool_outputs
        tool_output = final_state.tool_outputs["read_file"]
        assert "Error:" in tool_output
        assert "not found" in tool_output
        
        # Verify system didn't crash and completed execution
        assert "Saving final checkpoint for thread: error_handling_test" in final_state.message_history[-1]
        
        # Verify checkpoint was still saved despite error
        saved_state = checkpointer.load("error_handling_test")
        assert saved_state is not None
    
    def test_system_with_complex_state_evolution(self, temp_db_path, test_file_content, mock_llm_client):
        """Test system with complex state evolution through multiple components."""
        test_file_path, expected_content = test_file_content
        
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        graph = Graph(checkpointer=checkpointer)
        
        # Create rich initial state
        initial_state = State()
        initial_state.data.update({
            'goal': 'Прочитать и обработать файл данных',
            'processing_config': {
                'max_lines': 100,
                'encoding': 'utf-8',
                'analysis_type': 'basic'
            },
            'metadata': {
                'user_id': 'test_user',
                'session_id': 'session_123',
                'timestamp': '2023-01-01T00:00:00Z'
            }
        })
        initial_state.add_message("Complex processing task started")
        
        # Create tools and agents
        read_file_tool = ReadFileTool()
        reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        file_reader_node = ToolNode("file_reader", read_file_tool)
        
        graph.add_node(reasoning_agent)
        graph.add_node(file_reader_node)
        
        # Enhanced reasoning that considers configuration
        original_reasoning = reasoning_agent._perform_basic_reasoning
        
        def config_aware_reasoning(goal, state):
            if "прочитать" in goal.lower() and "файл" in goal.lower():
                config = state.data.get('processing_config', {})
                encoding = config.get('encoding', 'utf-8')
                
                state.data['next_tool'] = 'read_file'
                state.data['file_path'] = test_file_path
                state.data['tool_args'] = {'file_path': test_file_path}
                
                # Add processing context
                state.data['processing_context'] = {
                    'encoding_used': encoding,
                    'config_applied': True,
                    'file_processed': test_file_path
                }
                
                state.add_message(f"File reading configured with encoding: {encoding}")
        
        reasoning_agent._perform_basic_reasoning = config_aware_reasoning
        
        # Custom tool node that adds post-processing
        class EnhancedToolNode(ToolNode):
            def execute(self, state):
                result_state = super().execute(state)
                
                # Add post-processing analysis
                if self.tool.name == "read_file" and "read_file" in result_state.tool_outputs:
                    content = result_state.tool_outputs["read_file"]
                    if not content.startswith("Error:"):
                        lines = content.split('\n')
                        analysis = {
                            'total_lines': len(lines),
                            'non_empty_lines': len([line for line in lines if line.strip()]),
                            'total_characters': len(content),
                            'first_line_preview': lines[0][:50] if lines else ""
                        }
                        result_state.data['file_analysis'] = analysis
                        result_state.add_message(f"File analysis completed: {analysis['total_lines']} lines")
                
                return result_state
        
        enhanced_file_node = EnhancedToolNode("enhanced_file_reader", read_file_tool)
        graph.nodes["file_reader"] = enhanced_file_node  # Replace the node
        
        graph.add_edge(
            "reasoning_agent", 
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        # Execute graph
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="complex_state_test",
            start_node="reasoning_agent",
            max_iterations=10
        )
        
        # Verify complex state evolution
        
        # Original data should be preserved
        assert final_state.data['processing_config']['max_lines'] == 100
        assert final_state.data['metadata']['user_id'] == 'test_user'
        
        # Processing context should be added
        assert 'processing_context' in final_state.data
        assert final_state.data['processing_context']['config_applied'] is True
        assert final_state.data['processing_context']['encoding_used'] == 'utf-8'
        
        # File should be read and analyzed
        assert "read_file" in final_state.tool_outputs
        assert final_state.tool_outputs["read_file"] == expected_content
        
        # Analysis should be added
        assert 'file_analysis' in final_state.data
        analysis = final_state.data['file_analysis']
        assert analysis['total_lines'] > 0
        assert analysis['total_characters'] > 0
        assert len(analysis['first_line_preview']) > 0
        
        # Verify message evolution
        messages = final_state.message_history
        assert "Complex processing task started" in messages
        assert any("encoding: utf-8" in msg for msg in messages)
        assert any("File analysis completed" in msg for msg in messages)
    
    def test_checkpoint_recovery_integration(self, temp_db_path, test_file_content, mock_llm_client):
        """Test checkpoint recovery in a full system scenario."""
        test_file_path, expected_content = test_file_content
        
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        
        # Phase 1: Start execution and save checkpoint
        graph1 = Graph(checkpointer=checkpointer)
        
        initial_state = State()
        initial_state.data['goal'] = 'Прочитать файл для восстановления'
        initial_state.data['phase'] = 'initial'
        
        read_file_tool = ReadFileTool()
        reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        
        graph1.add_node(reasoning_agent)
        
        # Modify reasoning to simulate partial execution
        def partial_reasoning(goal, state):
            state.data['phase'] = 'reasoning_completed'
            state.data['next_tool'] = 'read_file'
            state.data['file_path'] = test_file_path
            state.add_message("Reasoning phase completed - ready for tool execution")
            # Don't proceed to tool execution in this phase
        
        reasoning_agent._perform_basic_reasoning = partial_reasoning
        
        # Execute only reasoning phase
        partial_state = graph1.execute(
            initial_state=initial_state,
            thread_id="checkpoint_recovery_test",
            start_node="reasoning_agent",
            max_iterations=5
        )
        
        # Verify partial execution
        assert partial_state.data['phase'] == 'reasoning_completed'
        assert partial_state.data['next_tool'] == 'read_file'
        assert "read_file" not in partial_state.tool_outputs  # Tool not executed yet
        
        # Phase 2: Create new graph and recover from checkpoint
        graph2 = Graph(checkpointer=checkpointer)
        
        # Load saved state
        recovered_state = graph2.load_checkpoint("checkpoint_recovery_test")
        assert recovered_state is not None
        assert recovered_state.data['phase'] == 'reasoning_completed'
        assert recovered_state.data['goal'] == initial_state.data['goal']
        
        # Continue execution with tool node
        file_reader_node = ToolNode("file_reader", read_file_tool)
        reasoning_agent2 = ReasoningAgent("reasoning_agent", [read_file_tool], mock_llm_client)
        
        graph2.add_node(reasoning_agent2)
        graph2.add_node(file_reader_node)
        
        # Modify reasoning for continuation
        def continuation_reasoning(goal, state):
            if state.data.get('phase') == 'reasoning_completed':
                state.data['phase'] = 'continuing_from_checkpoint'
                state.add_message("Continuing execution from checkpoint")
                # Keep existing tool recommendation
            else:
                state.add_message("No continuation needed")
        
        reasoning_agent2._perform_basic_reasoning = continuation_reasoning
        
        graph2.add_edge(
            "reasoning_agent", 
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        # Continue execution from recovered state
        final_state = graph2.execute(
            initial_state=recovered_state,
            thread_id="checkpoint_recovery_test_continued",
            start_node="reasoning_agent",
            max_iterations=10
        )
        
        # Verify complete execution after recovery
        assert final_state.data['phase'] == 'continuing_from_checkpoint'
        assert "read_file" in final_state.tool_outputs
        assert final_state.tool_outputs["read_file"] == expected_content
        
        # Verify message continuity
        messages = final_state.message_history
        assert "Reasoning phase completed - ready for tool execution" in messages
        assert "Continuing execution from checkpoint" in messages
        assert any("Executing tool: read_file" in msg for msg in messages)
    
    def test_system_scalability_with_multiple_agents_and_tools(self, temp_db_path, test_file_content, mock_llm_client):
        """Test system scalability with multiple agents and tools."""
        test_file_path, expected_content = test_file_content
        
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        graph = Graph(checkpointer=checkpointer)
        
        # Create multiple tools
        read_file_tool = ReadFileTool()
        
        # Create multiple agents with different capabilities
        file_agent = ReasoningAgent("file_agent", [read_file_tool], mock_llm_client)
        coordinator_agent = ReasoningAgent("coordinator_agent", [], mock_llm_client)
        
        # Create multiple tool nodes
        file_reader_node = ToolNode("file_reader", read_file_tool)
        
        # Add all nodes to graph
        graph.add_node(coordinator_agent)
        graph.add_node(file_agent)
        graph.add_node(file_reader_node)
        
        # Create complex state
        initial_state = State()
        initial_state.data.update({
            'goal': 'Координировать чтение файла через агентов',
            'workflow_stage': 'coordination',
            'agents_involved': ['coordinator_agent', 'file_agent'],
            'files_to_process': [test_file_path]
        })
        
        # Enhanced reasoning for coordinator
        def coordinator_reasoning(goal, state):
            stage = state.data.get('workflow_stage', '')
            
            if stage == 'coordination':
                state.data['workflow_stage'] = 'delegate_to_file_agent'
                state.data['delegate_to'] = 'file_agent'
                state.add_message("Coordinator: Delegating to file agent")
            else:
                state.add_message("Coordinator: No action needed")
        
        # Enhanced reasoning for file agent
        def file_agent_reasoning(goal, state):
            stage = state.data.get('workflow_stage', '')
            
            if stage == 'delegate_to_file_agent' or "файл" in goal.lower():
                state.data['workflow_stage'] = 'file_processing'
                state.data['next_tool'] = 'read_file'
                state.data['file_path'] = test_file_path
                state.data['tool_args'] = {'file_path': test_file_path}
                state.add_message("File agent: Processing file request")
            else:
                state.add_message("File agent: No file processing needed")
        
        coordinator_agent._perform_basic_reasoning = coordinator_reasoning
        file_agent._perform_basic_reasoning = file_agent_reasoning
        
        # Add edges for complex workflow
        graph.add_edge(
            "coordinator_agent",
            "file_agent",
            condition=lambda state: state.data.get('delegate_to') == 'file_agent'
        )
        
        graph.add_edge(
            "file_agent",
            "file_reader",
            condition=lambda state: state.data.get('next_tool') == 'read_file'
        )
        
        # Execute complex workflow
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="scalability_test",
            start_node="coordinator_agent",
            max_iterations=15
        )
        
        # Verify complex workflow execution
        assert final_state.data['workflow_stage'] == 'file_processing'
        
        # Verify all agents participated
        agent_messages = [msg for msg in final_state.message_history if ("coordinator:" in msg.lower() or "file agent:" in msg.lower())]
        assert len(agent_messages) >= 2
        assert any("Coordinator:" in msg for msg in agent_messages)
        assert any("File agent:" in msg for msg in agent_messages)
        
        # Verify file was processed
        assert "read_file" in final_state.tool_outputs
        assert final_state.tool_outputs["read_file"] == expected_content
        
        # Verify execution flow
        execution_messages = [msg for msg in final_state.message_history if "Executing node:" in msg]
        node_names = [msg.split(":")[-1].strip() for msg in execution_messages]
        assert "coordinator_agent" in node_names
        assert "file_agent" in node_names
        assert "file_reader" in node_names
        
        # Verify checkpoint saved complex state
        saved_state = checkpointer.load("scalability_test")
        assert saved_state is not None
        assert saved_state.data['agents_involved'] == ['coordinator_agent', 'file_agent']