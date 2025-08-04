"""
Simplified execution runner that provides clean user experience.
Wraps the complex ONIKS execution with user-friendly progress tracking.
"""

import time
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from ..core.graph import Graph, ToolNode
from ..core.state import State
from ..core.checkpoint import SQLiteCheckpointSaver
from ..agents.planner_agent import PlannerAgent
from ..agents.reasoning_agent import ReasoningAgent
from ..llm.client import OllamaClient
from ..tools.fs_tools import WriteFileTool, CreateDirectoryTool, ListFilesTool
from ..tools.shell_tools import ExecuteBashCommandTool
from ..tools.core_tools import TaskCompleteTool
from ..tools.file_tools import FileSearchReplaceTool


class TaskExecutor:
    """Simplified executor that hides complexity from users."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.setup_logging()
        
    def setup_logging(self):
        """Suppress verbose logging during execution."""
        # Create a custom logger that doesn't output to console
        self.logger = logging.getLogger('oniks.ui.runner')
        self.logger.setLevel(logging.ERROR)
        
        # Suppress all other ONIKS logging during execution
        for logger_name in ['oniks.agents', 'oniks.core', 'oniks.llm', 'oniks.tools']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
    
    def execute_task(self, goal: str) -> Dict[str, Any]:
        """
        Execute a user task with clean error handling and progress tracking.
        
        Args:
            goal: User's goal description
            
        Returns:
            Dictionary with execution results and metadata
        """
        start_time = time.time()
        
        try:
            # Initialize system
            self._update_progress("Initializing AI system...")
            
            llm_client = OllamaClient(timeout=600)  # Shorter timeout for users
            
            # Create tools
            tools = {
                'list_files': ListFilesTool(),
                'write_file': WriteFileTool(),
                'create_directory': CreateDirectoryTool(),
                'execute_bash_command': ExecuteBashCommandTool(),
                'task_complete': TaskCompleteTool(),
                'file_search_replace': FileSearchReplaceTool()
            }
            
            # Create agents
            self._update_progress("Setting up AI agents...")
            
            planner_agent = PlannerAgent(
                name="planner_agent",
                llm_client=llm_client,
                available_tools=list(tools.values())
            )
            
            reasoning_agent = ReasoningAgent(
                name="reasoning_agent", 
                llm_client=llm_client,
                available_tools=list(tools.values())
            )
            
            # Setup execution environment
            self._update_progress("Creating execution plan...")
            
            checkpoint_file = f"temp_session_{int(time.time())}.db"
            checkpoint_saver = SQLiteCheckpointSaver(checkpoint_file)
            
            # Create initial state
            initial_state = State()
            initial_state.data['goal'] = goal
            
            # Build execution graph
            graph = Graph(checkpoint_saver=checkpoint_saver)
            
            # Add nodes
            planner_node = graph.add_node(planner_agent)
            reasoning_node = graph.add_node(reasoning_agent)
            
            tool_nodes = {}
            for name, tool in tools.items():
                tool_nodes[name] = graph.add_node(ToolNode(tool))
            
            # Build edges
            graph.add_edge(planner_node, reasoning_node)
            for tool_node in tool_nodes.values():
                graph.add_edge(reasoning_node, tool_node)
                graph.add_edge(tool_node, reasoning_node)
            
            # Set termination condition
            graph.set_termination_condition(
                lambda state: state.data.get('next_tool') == 'task_complete'
            )
            
            # Execute with progress tracking
            self._update_progress("Executing plan...")
            
            # Use a custom execution loop to track progress
            result_state = self._execute_with_progress(graph, initial_state)
            
            # Clean up
            try:
                Path(checkpoint_file).unlink(missing_ok=True)
            except:
                pass
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'state': result_state,
                'execution_time': execution_time,
                'created_files': self._extract_created_files(result_state),
                'executed_commands': self._extract_executed_commands(result_state)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Clean up on error
            try:
                Path(checkpoint_file).unlink(missing_ok=True)
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _execute_with_progress(self, graph: Graph, initial_state: State) -> State:
        """Execute graph with progress callbacks."""
        current_state = initial_state
        current_node = graph.start_node
        iteration = 0
        max_iterations = 50
        
        step_descriptions = {
            'planner_agent': "Creating task plan...",
            'reasoning_agent': "Analyzing next step...",
            'write_file': "Creating files...", 
            'create_directory': "Creating directories...",
            'execute_bash_command': "Running commands...",
            'file_search_replace': "Modifying files...",
            'list_files': "Checking files...",
            'task_complete': "Finishing up..."
        }
        
        while current_node and iteration < max_iterations:
            iteration += 1
            
            # Update progress based on node type
            node_name = getattr(current_node, 'name', type(current_node).__name__)
            if hasattr(current_node, 'tool'):
                tool_name = type(current_node.tool).__name__.lower().replace('tool', '')
                description = step_descriptions.get(tool_name, f"Working on {tool_name}...")
            else:
                description = step_descriptions.get(node_name, f"Processing {node_name}...")
            
            self._update_progress(description)
            
            # Execute node
            current_state = current_node.execute(current_state)
            
            # Check termination
            if graph.termination_condition and graph.termination_condition(current_state):
                break
            
            # Get next node
            next_candidates = graph.get_next_nodes(current_node, current_state)
            if not next_candidates:
                break
            
            current_node = next_candidates[0]
        
        return current_state
    
    def _update_progress(self, message: str):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def _extract_created_files(self, state: State) -> list:
        """Extract list of files created during execution."""
        created_files = []
        
        if 'file_path' in state.data:
            file_path = state.data['file_path']
            if Path(file_path).exists():
                created_files.append(file_path)
        
        return created_files
    
    def _extract_executed_commands(self, state: State) -> list:
        """Extract list of commands executed."""
        commands = []
        
        if 'command' in state.data:
            commands.append(state.data['command'])
        
        return commands