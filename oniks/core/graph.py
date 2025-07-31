"""Graph execution framework for the ONIKS NeuralNet project.

This module provides the core classes for building and executing computational graphs
with nodes, edges, and conditional transitions based on state.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING
from pydantic import BaseModel, Field

from oniks.core.state import State

if TYPE_CHECKING:
    from oniks.core.checkpoint import CheckpointSaver
    from oniks.tools.base import Tool


class Node(ABC):
    """Abstract base class representing a single execution step in the graph.
    
    Each node represents a discrete operation that can be performed on the graph state.
    Nodes must implement the execute method to define their specific behavior.
    
    Attributes:
        name: Unique identifier for the node.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize a new node with the given name.
        
        Args:
            name: Unique identifier for this node.
            
        Raises:
            ValueError: If name is empty or None.
        """
        if not name:
            raise ValueError("Node name cannot be empty")
        self.name = name
    
    @abstractmethod
    def execute(self, state: State) -> State:
        """Execute the node's logic on the given state.
        
        This method must be implemented by all concrete node classes to define
        their specific behavior when executed in the graph.
        
        Args:
            state: The current state of the graph execution.
            
        Returns:
            The modified state after executing this node's logic.
            
        Raises:
            NotImplementedError: If not implemented by concrete subclass.
        """
        pass
    
    def __str__(self) -> str:
        """Return string representation of the node.
        
        Returns:
            String representation showing the node's name and type.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the node.
        
        Returns:
            Detailed string representation for debugging.
        """
        return self.__str__()


class ToolNode(Node):
    """Node that executes a tool during graph execution.
    
    A ToolNode wraps a Tool instance and integrates it into the graph execution
    framework. When executed, it extracts arguments from the state data,
    calls the tool's execute method, and stores the result in the state's
    tool_outputs dictionary.
    
    The node expects the required arguments for the tool to be present in
    state.data. The tool's output is stored in state.tool_outputs using
    the tool's name as the key.
    
    Attributes:
        tool: The Tool instance to execute.
    
    Example:
        >>> from oniks.tools.file_tools import ReadFileTool
        >>> tool = ReadFileTool()
        >>> node = ToolNode("file_reader", tool)
        >>> state = State()
        >>> state.data["file_path"] = "/path/to/file.txt"
        >>> result_state = node.execute(state)
        >>> print(result_state.tool_outputs["read_file"])
        File content here...
    """
    
    def __init__(self, name: str, tool: "Tool") -> None:
        """Initialize a ToolNode with a tool instance.
        
        Args:
            name: Unique identifier for this node.
            tool: The Tool instance to execute when this node runs.
            
        Raises:
            ValueError: If name is empty or tool is None.
        """
        super().__init__(name)
        if tool is None:
            raise ValueError("Tool cannot be None")
        self.tool = tool
    
    def execute(self, state: State) -> State:
        """Execute the tool with arguments from state data.
        
        This method extracts all key-value pairs from state.data and passes them
        as keyword arguments to the tool's execute method. The tool's output
        is then stored in state.tool_outputs using the tool's name as the key.
        
        Args:
            state: The current state containing tool arguments in the data field.
            
        Returns:
            The modified state with tool output added to tool_outputs.
        """
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Add message about tool execution
        result_state.add_message(f"Executing tool: {self.tool.name}")
        
        try:
            # Extract arguments from state data and execute the tool
            tool_output = self.tool.execute(**result_state.data)
            
            # Store the tool output in the state
            result_state.tool_outputs[self.tool.name] = tool_output
            
            result_state.add_message(f"Tool {self.tool.name} executed successfully")
            
        except Exception as e:
            # Handle any exceptions during tool execution
            error_message = f"Error executing tool {self.tool.name}: {str(e)}"
            result_state.tool_outputs[self.tool.name] = error_message
            result_state.add_message(error_message)
        
        return result_state


class Edge(BaseModel):
    """Represents a conditional connection between two nodes in the graph.
    
    An edge defines a potential transition from one node to another, with an
    optional condition that determines whether the transition should be taken
    based on the current state.
    
    Attributes:
        start_node: Name of the source node.
        end_node: Name of the destination node.
        condition: Optional function that determines if transition should occur.
    """
    
    start_node: str = Field(description="Name of the source node")
    end_node: str = Field(description="Name of the destination node")
    condition: Optional[Callable[[State], bool]] = Field(
        default=None,
        description="Optional condition function for the edge transition"
    )
    
    class Config:
        """Pydantic configuration for Edge class."""
        arbitrary_types_allowed = True
    
    def should_transition(self, state: State) -> bool:
        """Determine if the transition should occur based on the current state.
        
        Args:
            state: The current state to evaluate against the condition.
            
        Returns:
            True if the transition should occur, False otherwise.
            If no condition is set, always returns True.
        """
        if self.condition is None:
            return True
        return self.condition(state)
    
    def __str__(self) -> str:
        """Return string representation of the edge.
        
        Returns:
            String representation showing the edge connection.
        """
        condition_info = "conditional" if self.condition else "unconditional"
        return f"Edge({self.start_node} -> {self.end_node}, {condition_info})"


class Graph:
    """Represents a computational graph with nodes and conditional edges.
    
    The Graph class manages a collection of nodes connected by edges, providing
    functionality to execute the graph from a starting point until completion
    or until a stopping condition is met. Optionally supports checkpointing
    for state persistence during execution.
    
    Attributes:
        nodes: Dictionary mapping node names to Node instances.
        edges: Dictionary mapping node names to lists of outgoing edges.
        checkpointer: Optional checkpoint saver for state persistence.
    """
    
    def __init__(self, checkpointer: Optional["CheckpointSaver"] = None) -> None:
        """Initialize an empty graph with optional checkpointing support.
        
        Args:
            checkpointer: Optional CheckpointSaver instance for state persistence.
                         If provided, the graph will save state before executing each node.
        """
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self.checkpointer = checkpointer
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        Args:
            node: The Node instance to add to the graph.
            
        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists")
        
        self.nodes[node.name] = node
        if node.name not in self.edges:
            self.edges[node.name] = []
    
    def add_edge(
        self, 
        start_node: str, 
        end_node: str, 
        condition: Optional[Callable[[State], bool]] = None
    ) -> None:
        """Add an edge between two nodes in the graph.
        
        Args:
            start_node: Name of the source node.
            end_node: Name of the destination node.
            condition: Optional function that determines if transition should occur.
                      If None, the transition will always occur.
                      
        Raises:
            ValueError: If either node doesn't exist in the graph.
        """
        if start_node not in self.nodes:
            raise ValueError(f"Start node '{start_node}' does not exist in graph")
        if end_node not in self.nodes:
            raise ValueError(f"End node '{end_node}' does not exist in graph")
        
        edge = Edge(start_node=start_node, end_node=end_node, condition=condition)
        self.edges[start_node].append(edge)
    
    def get_next_nodes(self, current_node: str, state: State) -> List[str]:
        """Get list of nodes that should be executed next based on edge conditions.
        
        Args:
            current_node: Name of the current node.
            state: Current state to evaluate edge conditions against.
            
        Returns:
            List of node names that should be executed next.
            
        Raises:
            ValueError: If the current node doesn't exist in the graph.
        """
        if current_node not in self.nodes:
            raise ValueError(f"Node '{current_node}' does not exist in graph")
        
        next_nodes = []
        for edge in self.edges[current_node]:
            if edge.should_transition(state):
                next_nodes.append(edge.end_node)
        
        return next_nodes
    
    def execute(
        self, 
        initial_state: State, 
        thread_id: str,
        start_node: str = None,
        max_iterations: int = 1000
    ) -> State:
        """Execute the graph starting from the specified node.
        
        Args:
            initial_state: The initial state to begin execution with.
            thread_id: Unique identifier for this execution thread/task.
            start_node: Name of the starting node. If None, uses the first added node.
            max_iterations: Maximum number of iterations to prevent infinite loops.
            
        Returns:
            The final state after graph execution completes.
            
        Raises:
            ValueError: If start_node is specified but doesn't exist, or if graph is empty,
                       or if thread_id is empty.
            RuntimeError: If maximum iterations exceeded (potential infinite loop).
        """
        if not self.nodes:
            raise ValueError("Cannot execute empty graph")
        
        if not thread_id:
            raise ValueError("Thread ID cannot be empty or None")
        
        # Determine starting node
        if start_node is None:
            start_node = next(iter(self.nodes))
        elif start_node not in self.nodes:
            raise ValueError(f"Start node '{start_node}' does not exist in graph")
        
        current_state = initial_state.model_copy(deep=True)
        current_nodes = [start_node]
        visited_nodes: Set[str] = set()
        iterations = 0
        
        current_state.add_message(f"Starting graph execution from node: {start_node}")
        
        # Save initial state if checkpointer is available
        if self.checkpointer:
            current_state.add_message(f"Saving initial checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        while current_nodes and iterations < max_iterations:
            iterations += 1
            next_nodes = []
            
            for node_name in current_nodes:
                if node_name in visited_nodes:
                    continue
                
                visited_nodes.add(node_name)
                node = self.nodes[node_name]
                
                current_state.add_message(f"Executing node: {node_name}")
                
                # Save checkpoint before executing node if checkpointer is available
                if self.checkpointer:
                    current_state.add_message(f"Saving checkpoint before executing node: {node_name}")
                    self.checkpointer.save(thread_id, current_state)
                
                # Execute current node
                current_state = node.execute(current_state)
                
                # Save checkpoint after executing node if checkpointer is available
                if self.checkpointer:
                    current_state.add_message(f"Saving checkpoint after executing node: {node_name}")
                    self.checkpointer.save(thread_id, current_state)
                
                # Get next nodes based on edge conditions
                next_candidates = self.get_next_nodes(node_name, current_state)
                next_nodes.extend(next_candidates)
            
            # Remove duplicates while preserving order
            current_nodes = list(dict.fromkeys(next_nodes))
        
        if iterations >= max_iterations:
            raise RuntimeError(
                f"Graph execution exceeded maximum iterations ({max_iterations}). "
                "Possible infinite loop detected."
            )
        
        current_state.add_message("Graph execution completed")
        
        # Save final state if checkpointer is available
        if self.checkpointer:
            current_state.add_message(f"Saving final checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        return current_state
    
    def load_checkpoint(self, thread_id: str) -> Optional[State]:
        """Load a checkpoint for a specific thread.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            
        Returns:
            The saved State object for the thread, or None if no checkpoint exists
            or no checkpointer is configured.
            
        Raises:
            ValueError: If thread_id is empty or None.
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty or None")
        
        if not self.checkpointer:
            return None
        
        return self.checkpointer.load(thread_id)
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in the graph.
        
        Returns:
            Number of nodes in the graph.
        """
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph.
        
        Returns:
            Total number of edges in the graph.
        """
        return sum(len(edge_list) for edge_list in self.edges.values())
    
    def has_node(self, node_name: str) -> bool:
        """Check if a node exists in the graph.
        
        Args:
            node_name: Name of the node to check.
            
        Returns:
            True if the node exists, False otherwise.
        """
        return node_name in self.nodes
    
    def get_node_names(self) -> List[str]:
        """Get list of all node names in the graph.
        
        Returns:
            List of node names in the order they were added.
        """
        return list(self.nodes.keys())
    
    def clear(self) -> None:
        """Remove all nodes and edges from the graph.
        
        This method completely clears the graph, returning it to an empty state.
        """
        self.nodes.clear()
        self.edges.clear()
    
    def __str__(self) -> str:
        """Return string representation of the graph.
        
        Returns:
            String representation showing nodes and edges count.
        """
        return f"Graph(nodes={self.get_node_count()}, edges={self.get_edge_count()})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the graph.
        
        Returns:
            Detailed string representation for debugging.
        """
        return self.__str__()