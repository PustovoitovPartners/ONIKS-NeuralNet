"""Unit tests for the Graph, Node, ToolNode, and Edge classes."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from oniks.core.graph import Graph, Node, ToolNode, Edge
from oniks.core.state import State
from oniks.tools.base import Tool


class TestNode:
    """Test the abstract Node class."""
    
    def test_node_initialization_valid_name(self):
        """Test Node initialization with valid name."""
        # Create a concrete implementation for testing
        class ConcreteNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        node = ConcreteNode("test_node")
        
        assert node.name == "test_node"
    
    def test_node_initialization_empty_name_raises_error(self):
        """Test Node initialization with empty name raises ValueError."""
        class ConcreteNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ConcreteNode("")
    
    def test_node_initialization_none_name_raises_error(self):
        """Test Node initialization with None name raises ValueError."""
        class ConcreteNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ConcreteNode(None)
    
    def test_node_string_representation(self):
        """Test Node string representation."""
        class ConcreteNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        node = ConcreteNode("test_node")
        
        assert str(node) == "ConcreteNode(name='test_node')"
        assert repr(node) == "ConcreteNode(name='test_node')"
    
    def test_node_abstract_execute_method(self):
        """Test that Node.execute is abstract and raises NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            Node("test")


class TestToolNode:
    """Test the ToolNode class."""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.execute = Mock(return_value="tool_output")
        return tool
    
    def test_tool_node_initialization_valid(self, mock_tool):
        """Test ToolNode initialization with valid parameters."""
        node = ToolNode("tool_node", mock_tool)
        
        assert node.name == "tool_node"
        assert node.tool == mock_tool
    
    def test_tool_node_initialization_empty_name_raises_error(self, mock_tool):
        """Test ToolNode initialization with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ToolNode("", mock_tool)
    
    def test_tool_node_initialization_none_tool_raises_error(self):
        """Test ToolNode initialization with None tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool cannot be None"):
            ToolNode("tool_node", None)
    
    def test_tool_node_execute_success(self, mock_tool):
        """Test successful tool execution."""
        node = ToolNode("tool_node", mock_tool)
        state = State()
        state.data["arg1"] = "value1"
        state.data["arg2"] = "value2"
        
        result_state = node.execute(state)
        
        # Verify tool was called with state data
        mock_tool.execute.assert_called_once_with(arg1="value1", arg2="value2")
        
        # Verify result state
        assert result_state is not state  # Should be a copy
        assert result_state.tool_outputs["test_tool"] == "tool_output"
        assert len(result_state.message_history) >= 2  # Should have execution messages
        assert any("Executing tool: test_tool" in msg for msg in result_state.message_history)
        assert any("executed successfully" in msg for msg in result_state.message_history)
    
    def test_tool_node_execute_with_exception(self, mock_tool):
        """Test tool execution when tool raises exception."""
        mock_tool.execute.side_effect = Exception("Tool error")
        node = ToolNode("tool_node", mock_tool)
        state = State()
        
        result_state = node.execute(state)
        
        # Verify error handling
        assert "Error executing tool test_tool: Tool error" in result_state.tool_outputs["test_tool"]
        assert any("Error executing tool" in msg for msg in result_state.message_history)
    
    def test_tool_node_execute_preserves_original_state(self, mock_tool):
        """Test that original state is not modified during execution."""
        node = ToolNode("tool_node", mock_tool)
        original_state = State()
        original_state.data["test"] = "value"
        original_state.add_message("original message")
        
        result_state = node.execute(original_state)
        
        # Original state should be unchanged
        assert len(original_state.message_history) == 1
        assert "test_tool" not in original_state.tool_outputs
        
        # Result state should have changes
        assert len(result_state.message_history) > 1
        assert "test_tool" in result_state.tool_outputs
    
    def test_tool_node_execute_empty_state_data(self, mock_tool):
        """Test tool execution with empty state data."""
        node = ToolNode("tool_node", mock_tool)
        state = State()
        
        result_state = node.execute(state)
        
        # Tool should be called with no arguments
        mock_tool.execute.assert_called_once_with()
        assert result_state.tool_outputs["test_tool"] == "tool_output"


class TestEdge:
    """Test the Edge class."""
    
    def test_edge_initialization_without_condition(self):
        """Test Edge initialization without condition."""
        edge = Edge(start_node="node1", end_node="node2")
        
        assert edge.start_node == "node1"
        assert edge.end_node == "node2"
        assert edge.condition is None
    
    def test_edge_initialization_with_condition(self):
        """Test Edge initialization with condition."""
        condition = lambda state: state.data.get("test") == "value"
        edge = Edge(start_node="node1", end_node="node2", condition=condition)
        
        assert edge.start_node == "node1"
        assert edge.end_node == "node2"
        assert edge.condition == condition
    
    def test_edge_should_transition_no_condition(self):
        """Test should_transition with no condition always returns True."""
        edge = Edge(start_node="node1", end_node="node2")
        state = State()
        
        assert edge.should_transition(state) is True
    
    def test_edge_should_transition_condition_true(self):
        """Test should_transition when condition returns True."""
        condition = lambda state: state.data.get("test") == "value"
        edge = Edge(start_node="node1", end_node="node2", condition=condition)
        
        state = State()
        state.data["test"] = "value"
        
        assert edge.should_transition(state) is True
    
    def test_edge_should_transition_condition_false(self):
        """Test should_transition when condition returns False."""
        condition = lambda state: state.data.get("test") == "value"
        edge = Edge(start_node="node1", end_node="node2", condition=condition)
        
        state = State()
        state.data["test"] = "other_value"
        
        assert edge.should_transition(state) is False
    
    def test_edge_should_transition_condition_exception(self):
        """Test should_transition when condition raises exception."""
        def bad_condition(state):
            raise Exception("Condition error")
        
        edge = Edge(start_node="node1", end_node="node2", condition=bad_condition)
        state = State()
        
        # Exception should bubble up
        with pytest.raises(Exception, match="Condition error"):
            edge.should_transition(state)
    
    def test_edge_string_representation(self):
        """Test Edge string representation."""
        edge_unconditional = Edge(start_node="node1", end_node="node2")
        edge_conditional = Edge(
            start_node="node1", 
            end_node="node2", 
            condition=lambda s: True
        )
        
        assert str(edge_unconditional) == "Edge(node1 -> node2, unconditional)"
        assert str(edge_conditional) == "Edge(node1 -> node2, conditional)"


class TestGraph:
    """Test the Graph class."""
    
    @pytest.fixture
    def sample_node(self):
        """Create a sample node for testing."""
        class SampleNode(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message(f"Executed {self.name}")
                return result_state
        
        return SampleNode("sample_node")
    
    @pytest.fixture
    def mock_checkpointer(self):
        """Create a mock checkpointer."""
        checkpointer = Mock()
        checkpointer.save = Mock()
        checkpointer.load = Mock(return_value=None)
        return checkpointer
    
    def test_graph_initialization_no_checkpointer(self):
        """Test Graph initialization without checkpointer."""
        graph = Graph()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.checkpointer is None
    
    def test_graph_initialization_with_checkpointer(self, mock_checkpointer):
        """Test Graph initialization with checkpointer."""
        graph = Graph(checkpointer=mock_checkpointer)
        
        assert graph.checkpointer == mock_checkpointer
    
    def test_graph_add_node_success(self, sample_node):
        """Test successful node addition."""
        graph = Graph()
        
        graph.add_node(sample_node)
        
        assert "sample_node" in graph.nodes
        assert graph.nodes["sample_node"] == sample_node
        assert "sample_node" in graph.edges
        assert len(graph.edges["sample_node"]) == 0
        assert graph.get_node_count() == 1
    
    def test_graph_add_node_duplicate_name_raises_error(self, sample_node):
        """Test adding node with duplicate name raises ValueError."""
        graph = Graph()
        
        class AnotherNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        duplicate_node = AnotherNode("sample_node")  # Same name as sample_node
        
        graph.add_node(sample_node)
        
        with pytest.raises(ValueError, match="Node with name 'sample_node' already exists"):
            graph.add_node(duplicate_node)
    
    def test_graph_add_edge_success(self, sample_node):
        """Test successful edge addition."""
        graph = Graph()
        
        class SecondNode(Node):
            def execute(self, state: State) -> State:
                return state
        
        second_node = SecondNode("second_node")
        
        graph.add_node(sample_node)
        graph.add_node(second_node)
        
        condition = lambda state: True
        graph.add_edge("sample_node", "second_node", condition)
        
        assert len(graph.edges["sample_node"]) == 1
        edge = graph.edges["sample_node"][0]
        assert edge.start_node == "sample_node"
        assert edge.end_node == "second_node"
        assert edge.condition == condition
        assert graph.get_edge_count() == 1
    
    def test_graph_add_edge_start_node_missing_raises_error(self, sample_node):
        """Test adding edge with missing start node raises ValueError."""
        graph = Graph()
        graph.add_node(sample_node)
        
        with pytest.raises(ValueError, match="Start node 'missing_node' does not exist"):
            graph.add_edge("missing_node", "sample_node")
    
    def test_graph_add_edge_end_node_missing_raises_error(self, sample_node):
        """Test adding edge with missing end node raises ValueError."""
        graph = Graph()
        graph.add_node(sample_node)
        
        with pytest.raises(ValueError, match="End node 'missing_node' does not exist"):
            graph.add_edge("sample_node", "missing_node")
    
    def test_graph_get_next_nodes_no_edges(self, sample_node):
        """Test get_next_nodes when node has no outgoing edges."""
        graph = Graph()
        graph.add_node(sample_node)
        state = State()
        
        next_nodes = graph.get_next_nodes("sample_node", state)
        
        assert next_nodes == []
    
    def test_graph_get_next_nodes_with_edges(self, sample_node):
        """Test get_next_nodes with edges."""
        graph = Graph()
        
        class Node2(Node):
            def execute(self, state: State) -> State:
                return state
        
        class Node3(Node):
            def execute(self, state: State) -> State:
                return state
        
        node2 = Node2("node2")
        node3 = Node3("node3")
        
        graph.add_node(sample_node)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Add conditional edges
        graph.add_edge("sample_node", "node2", lambda s: s.data.get("go_to_2"))
        graph.add_edge("sample_node", "node3", lambda s: s.data.get("go_to_3"))
        
        # Test when conditions are met
        state = State()
        state.data["go_to_2"] = True
        state.data["go_to_3"] = True
        
        next_nodes = graph.get_next_nodes("sample_node", state)
        
        assert set(next_nodes) == {"node2", "node3"}
    
    def test_graph_get_next_nodes_missing_node_raises_error(self):
        """Test get_next_nodes with missing node raises ValueError."""
        graph = Graph()
        state = State()
        
        with pytest.raises(ValueError, match="Node 'missing_node' does not exist"):
            graph.get_next_nodes("missing_node", state)
    
    def test_graph_execute_empty_graph_raises_error(self):
        """Test executing empty graph raises ValueError."""
        graph = Graph()
        state = State()
        
        with pytest.raises(ValueError, match="Cannot execute empty graph"):
            graph.execute(state, "thread_id")
    
    def test_graph_execute_empty_thread_id_raises_error(self, sample_node):
        """Test executing with empty thread_id raises ValueError."""
        graph = Graph()
        graph.add_node(sample_node)
        state = State()
        
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            graph.execute(state, "")
    
    def test_graph_execute_invalid_start_node_raises_error(self, sample_node):
        """Test executing with invalid start_node raises ValueError."""
        graph = Graph()
        graph.add_node(sample_node)
        state = State()
        
        with pytest.raises(ValueError, match="Start node 'invalid_node' does not exist"):
            graph.execute(state, "thread_id", start_node="invalid_node")
    
    def test_graph_execute_single_node_success(self, sample_node):
        """Test successful execution of single node."""
        graph = Graph()
        graph.add_node(sample_node)
        
        initial_state = State()
        initial_state.data["test"] = "value"
        
        result_state = graph.execute(initial_state, "test_thread")
        
        assert result_state is not initial_state
        assert "Executed sample_node" in result_state.message_history
        assert any("Starting graph execution" in msg for msg in result_state.message_history)
        assert "Graph execution completed" in result_state.message_history[-1]
    
    def test_graph_execute_with_checkpointer(self, sample_node, mock_checkpointer):
        """Test graph execution with checkpointer."""
        graph = Graph(checkpointer=mock_checkpointer)
        graph.add_node(sample_node)
        
        initial_state = State()
        
        result_state = graph.execute(initial_state, "test_thread")
        
        # Verify checkpointer was called
        assert mock_checkpointer.save.call_count >= 3  # Initial, before node, after node, final
        save_calls = mock_checkpointer.save.call_args_list
        
        # Check thread_id is passed correctly
        for call in save_calls:
            assert call[0][0] == "test_thread"
    
    def test_graph_execute_max_iterations_exceeded_raises_error(self):
        """Test graph execution with max_iterations exceeded raises RuntimeError."""
        graph = Graph()
        
        # Create multiple nodes to create a chain longer than max_iterations
        class ChainNode(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                current_count = result_state.data.get("chain_count", 0)
                result_state.data["chain_count"] = current_count + 1
                return result_state
        
        # Create a chain of 10 nodes, each linking to the next
        for i in range(10):
            node = ChainNode(f"node_{i}")
            graph.add_node(node)
            
            if i > 0:
                # Link previous node to current node
                graph.add_edge(f"node_{i-1}", f"node_{i}")
        
        initial_state = State()
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            graph.execute(initial_state, "test_thread", start_node="node_0", max_iterations=5)
    
    def test_graph_execute_multiple_nodes_with_conditions(self):
        """Test graph execution with multiple nodes and conditions."""
        graph = Graph()
        
        class Node1(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Executed Node1")
                result_state.data["step"] = 1
                return result_state
        
        class Node2(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Executed Node2")
                result_state.data["step"] = 2
                return result_state
        
        node1 = Node1("node1")
        node2 = Node2("node2")
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edge from node1 to node2
        graph.add_edge("node1", "node2", lambda s: s.data.get("step") == 1)
        
        initial_state = State()
        
        result_state = graph.execute(initial_state, "test_thread")
        
        assert "Executed Node1" in result_state.message_history
        assert "Executed Node2" in result_state.message_history
        assert result_state.data["step"] == 2
    
    def test_graph_load_checkpoint_no_checkpointer(self, sample_node):
        """Test load_checkpoint when no checkpointer is configured."""
        graph = Graph()
        
        result = graph.load_checkpoint("test_thread")
        
        assert result is None
    
    def test_graph_load_checkpoint_empty_thread_id_raises_error(self, mock_checkpointer):
        """Test load_checkpoint with empty thread_id raises ValueError."""
        graph = Graph(checkpointer=mock_checkpointer)
        
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            graph.load_checkpoint("")
    
    def test_graph_load_checkpoint_success(self, mock_checkpointer):
        """Test successful checkpoint loading."""
        expected_state = State()
        expected_state.data["test"] = "value"
        mock_checkpointer.load.return_value = expected_state
        
        graph = Graph(checkpointer=mock_checkpointer)
        
        result = graph.load_checkpoint("test_thread")
        
        assert result == expected_state
        mock_checkpointer.load.assert_called_once_with("test_thread")
    
    def test_graph_utility_methods(self, sample_node):
        """Test Graph utility methods."""
        graph = Graph()
        
        # Test empty graph
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0
        assert not graph.has_node("sample_node")
        assert graph.get_node_names() == []
        
        # Add node and test
        graph.add_node(sample_node)
        
        class Node2(Node):
            def execute(self, state: State) -> State:
                return state
        
        node2 = Node2("node2")
        graph.add_node(node2)
        graph.add_edge("sample_node", "node2")
        
        assert graph.get_node_count() == 2
        assert graph.get_edge_count() == 1
        assert graph.has_node("sample_node")
        assert graph.has_node("node2")
        assert not graph.has_node("missing_node")
        assert set(graph.get_node_names()) == {"sample_node", "node2"}
    
    def test_graph_clear(self, sample_node):
        """Test Graph clear method."""
        graph = Graph()
        graph.add_node(sample_node)
        
        class Node2(Node):
            def execute(self, state: State) -> State:
                return state
        
        node2 = Node2("node2")
        graph.add_node(node2)
        graph.add_edge("sample_node", "node2")
        
        assert graph.get_node_count() == 2
        assert graph.get_edge_count() == 1
        
        graph.clear()
        
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_graph_string_representation(self, sample_node):
        """Test Graph string representation."""
        graph = Graph()
        
        assert str(graph) == "Graph(nodes=0, edges=0)"
        assert repr(graph) == "Graph(nodes=0, edges=0)"
        
        graph.add_node(sample_node)
        
        class Node2(Node):
            def execute(self, state: State) -> State:
                return state
        
        node2 = Node2("node2")
        graph.add_node(node2)
        graph.add_edge("sample_node", "node2")
        
        assert str(graph) == "Graph(nodes=2, edges=1)"
        assert repr(graph) == "Graph(nodes=2, edges=1)"


class TestGraphExecutionFlow:
    """Test complex graph execution flows."""
    
    def test_graph_execution_with_branching(self):
        """Test graph execution with conditional branching."""
        graph = Graph()
        
        class DecisionNode(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Making decision")
                # Set branch based on input data
                if result_state.data.get("choice") == "A":
                    result_state.data["go_A"] = True
                else:
                    result_state.data["go_B"] = True
                return result_state
        
        class NodeA(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Executed path A")
                result_state.data["result"] = "A"
                return result_state
        
        class NodeB(Node):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Executed path B")
                result_state.data["result"] = "B"
                return result_state
        
        decision = DecisionNode("decision")
        node_a = NodeA("node_a")
        node_b = NodeB("node_b")
        
        graph.add_node(decision)
        graph.add_node(node_a)
        graph.add_node(node_b)
        
        graph.add_edge("decision", "node_a", lambda s: s.data.get("go_A"))
        graph.add_edge("decision", "node_b", lambda s: s.data.get("go_B"))
        
        # Test path A
        state_a = State()
        state_a.data["choice"] = "A"
        
        result_a = graph.execute(state_a, "thread_a", start_node="decision")
        
        assert "Making decision" in result_a.message_history
        assert "Executed path A" in result_a.message_history
        assert "Executed path B" not in result_a.message_history
        assert result_a.data["result"] == "A"
        
        # Test path B
        state_b = State()
        state_b.data["choice"] = "B"
        
        result_b = graph.execute(state_b, "thread_b", start_node="decision")
        
        assert "Making decision" in result_b.message_history
        assert "Executed path B" in result_b.message_history
        assert "Executed path A" not in result_b.message_history
        assert result_b.data["result"] == "B"
    
    def test_graph_execution_with_sequential_processing(self):
        """Test graph execution with sequential processing pattern."""
        graph = Graph()
        
        class ProcessorNode(Node):
            def __init__(self, name: str, step_number: int):
                super().__init__(name)
                self.step_number = step_number
            
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                processed_steps = result_state.data.get("processed_steps", [])
                processed_steps.append(self.step_number)
                result_state.data["processed_steps"] = processed_steps
                result_state.add_message(f"Processed step {self.step_number}")
                
                # Set next step if there are more steps
                if self.step_number < 3:
                    result_state.data["next_step"] = self.step_number + 1
                
                return result_state
        
        # Create sequential processing nodes
        for i in range(1, 4):
            processor = ProcessorNode(f"processor_{i}", i)
            graph.add_node(processor)
            
            if i > 1:
                # Link previous processor to current
                graph.add_edge(
                    f"processor_{i-1}", 
                    f"processor_{i}",
                    lambda s, target_step=i: s.data.get("next_step") == target_step
                )
        
        initial_state = State()
        
        result_state = graph.execute(initial_state, "sequential_thread", start_node="processor_1")
        
        assert result_state.data["processed_steps"] == [1, 2, 3]
        # Should have 3 processing messages
        process_messages = [msg for msg in result_state.message_history if "Processed step" in msg]
        assert len(process_messages) == 3
        assert "Processed step 1" in process_messages[0]
        assert "Processed step 2" in process_messages[1]
        assert "Processed step 3" in process_messages[2]