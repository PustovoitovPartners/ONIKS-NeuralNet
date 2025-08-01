#!/usr/bin/env python3
"""Multi-step demonstration script for the hierarchical PlannerAgent + ReasoningAgent system.

This script demonstrates a comprehensive multi-step workflow with hierarchical task management
where the PlannerAgent decomposes complex goals into atomic subtasks, and the ReasoningAgent
executes each subtask by coordinating with multiple tool nodes.

The demonstration workflow:
1. PlannerAgent analyzes the goal and creates a structured plan with atomic subtasks
2. ReasoningAgent takes the first subtask from the plan and selects appropriate tools
3. Tool nodes execute the selected operations (file creation, command execution, etc.)
4. Returns to ReasoningAgent for the next subtask in the plan
5. Process continues until all subtasks are completed
6. Final confirmation task triggers task_complete tool
7. Graph terminates when all planned tasks are completed

Usage:
    python run_reasoning_test.py
"""

import sys
import shutil
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.core.checkpoint import SQLiteCheckpointSaver
from typing import Set, List, Dict, Optional
from oniks.tools.file_tools import ReadFileTool
from oniks.tools.fs_tools import ListFilesTool, WriteFileTool, CreateDirectoryTool
from oniks.tools.shell_tools import ExecuteBashCommandTool
from oniks.tools.core_tools import TaskCompleteTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.agents.planner_agent import PlannerAgent
from oniks.llm.client import OllamaClient, OllamaConnectionError


class MultiStepGraph(Graph):
    """Extended Graph class that allows re-execution of reasoning nodes for multi-step workflows.
    
    This class modifies the standard graph execution to allow specific nodes (like reasoning agents)
    to be re-executed multiple times, enabling multi-step workflows where the reasoning agent
    coordinates multiple tool executions.
    """
    
    def __init__(self, checkpointer: Optional = None, reusable_nodes: Optional[Set[str]] = None):
        """Initialize a MultiStepGraph.
        
        Args:
            checkpointer: Optional checkpoint saver for state persistence.
            reusable_nodes: Set of node names that can be re-executed multiple times.
        """
        super().__init__(checkpointer)
        self.reusable_nodes = reusable_nodes or set()
    
    def execute(
        self, 
        initial_state: State, 
        thread_id: str,
        start_node: str = None,
        max_iterations: int = 1000
    ) -> State:
        """Execute the graph with support for re-executable nodes.
        
        This method extends the base graph execution to allow certain nodes
        to be re-executed multiple times, enabling multi-step workflows.
        
        Args:
            initial_state: The initial state to begin execution with.
            thread_id: Unique identifier for this execution thread/task.
            start_node: Name of the starting node.
            max_iterations: Maximum number of iterations to prevent infinite loops.
            
        Returns:
            The final state after graph execution completes.
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
        
        current_state.add_message(f"Starting multi-step graph execution from node: {start_node}")
        
        # Save initial state if checkpointer is available
        if self.checkpointer:
            current_state.add_message(f"Saving initial checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        while current_nodes and iterations < max_iterations:
            iterations += 1
            next_nodes = []
            
            for node_name in current_nodes:
                # Allow re-execution of reusable nodes (like reasoning agents)
                if node_name in visited_nodes and node_name not in self.reusable_nodes:
                    continue
                
                visited_nodes.add(node_name)
                node = self.nodes[node_name]
                
                current_state.add_message(f"Executing node: {node_name} (iteration: {iterations})")
                
                # Save state before execution
                if self.checkpointer:
                    current_state.add_message(f"Saving checkpoint before executing node: {node_name}")
                    self.checkpointer.save(thread_id, current_state)
                
                # Execute current node
                current_state = node.execute(current_state)
                
                # Save state after execution
                if self.checkpointer:
                    current_state.add_message(f"Saving checkpoint after executing node: {node_name}")
                    self.checkpointer.save(thread_id, current_state)
                
                # Get next nodes based on edge conditions
                next_candidates = self.get_next_nodes(node_name, current_state)
                next_nodes.extend(next_candidates)
            
            # Remove duplicates while preserving order
            current_nodes = list(dict.fromkeys(next_nodes))
            
            # Reset visited status for reusable nodes if they're going to be executed again
            for node_name in current_nodes:
                if node_name in self.reusable_nodes and node_name in visited_nodes:
                    # Allow re-execution by removing from visited set
                    visited_nodes.discard(node_name)
        
        if iterations >= max_iterations:
            raise RuntimeError(
                f"Multi-step graph execution exceeded maximum iterations ({max_iterations}). "
                "Possible infinite loop detected."
            )
        
        current_state.add_message("Multi-step graph execution completed")
        
        # Save final state
        if self.checkpointer:
            current_state.add_message(f"Saving final checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        return current_state


def cleanup_demo_files() -> None:
    """Clean up any existing demo files and directories before starting."""
    demo_file_path = project_root / "hello.txt"
    output_dir_path = project_root / "output"
    
    if demo_file_path.exists():
        demo_file_path.unlink()
        print(f"Cleaned up existing demo file: {demo_file_path}")
    
    if output_dir_path.exists():
        # Remove the entire output directory and its contents
        shutil.rmtree(output_dir_path)
        print(f"Cleaned up existing output directory: {output_dir_path}")


def main() -> None:
    """Main multi-step demonstration function."""
    print("=== ONIKS Hierarchical PlannerAgent + ReasoningAgent Demonstration ===\n")
    
    # Clean up any existing demo files
    cleanup_demo_files()
    
    # Initialize multi-step graph with checkpoint saver
    print("1. Initializing multi-step graph and checkpoint saver...")
    checkpointer = SQLiteCheckpointSaver("demo_checkpoints.db")
    # Allow both planner and reasoning agents to be re-executed for multi-step workflows
    graph = MultiStepGraph(checkpointer=checkpointer, reusable_nodes={"planner_agent", "reasoning_agent"})
    
    # Create initial state with complex goal for hierarchical planning
    print("2. Creating initial state with complex goal for hierarchical planning...")
    initial_state = State()
    initial_state.data['goal'] = "Create a directory named 'output', and inside it, create a file 'log.txt' with the text 'System test OK'"
    initial_state.add_message("Demo started with complex goal that will be decomposed into atomic subtasks")
    
    print(f"   Goal: {initial_state.data['goal']}")
    
    # Create all required tools
    print("3. Creating tools...")
    list_files_tool = ListFilesTool()
    write_file_tool = WriteFileTool()
    create_directory_tool = CreateDirectoryTool()
    execute_bash_tool = ExecuteBashCommandTool()
    task_complete_tool = TaskCompleteTool()
    
    tools = [list_files_tool, write_file_tool, create_directory_tool, execute_bash_tool, task_complete_tool]
    
    for tool in tools:
        print(f"   Created tool: {tool.name}")
        print(f"   Tool description: {tool.description}")
    
    # Create LLM client
    print("4. Creating LLM client...")
    try:
        llm_client = OllamaClient()
        # Check if the default model is available
        if llm_client.check_model_availability():
            print("   LLM client created successfully with model 'llama3:8b'")
        else:
            print("   Warning: Model 'llama3:8b' not found, but client created")
            print("   Available models:", llm_client.list_available_models())
    except OllamaConnectionError as e:
        print(f"   Warning: {e}")
        print("   Note: Ensure Ollama is running locally with 'ollama serve'")
        print("   And that you have pulled the llama3:8b model with 'ollama pull llama3:8b'")
        llm_client = OllamaClient()  # Create client anyway for demonstration
    
    # Create agents and nodes
    print("5. Creating agents and nodes...")
    planner_agent = PlannerAgent("planner_agent", llm_client, tools)
    reasoning_agent = ReasoningAgent("reasoning_agent", tools, llm_client)
    
    # Create tool nodes for each tool
    list_files_node = ToolNode("list_files", list_files_tool)
    write_file_node = ToolNode("write_file", write_file_tool)
    create_directory_node = ToolNode("create_directory", create_directory_tool)
    execute_bash_node = ToolNode("execute_bash_command", execute_bash_tool)
    task_complete_node = ToolNode("task_complete", task_complete_tool)
    
    tool_nodes = [list_files_node, write_file_node, create_directory_node, execute_bash_node, task_complete_node]
    
    print(f"   Created planner agent: {planner_agent.name}")
    print(f"   Created reasoning agent: {reasoning_agent.name}")
    for node in tool_nodes:
        print(f"   Created tool node: {node.name}")
    
    # Add nodes to graph
    print("6. Building comprehensive graph structure...")
    graph.add_node(planner_agent)
    graph.add_node(reasoning_agent)
    for node in tool_nodes:
        graph.add_node(node)
    
    # Add edge from planner agent to reasoning agent (after plan is created)
    graph.add_edge(
        "planner_agent",
        "reasoning_agent", 
        condition=lambda state: 'plan' in state.data and state.data['plan']
    )
    
    # Add edges from reasoning agent to each tool node
    graph.add_edge(
        "reasoning_agent", 
        "list_files",
        condition=lambda state: state.data.get('next_tool') == 'list_files'
    )
    
    graph.add_edge(
        "reasoning_agent", 
        "write_file",
        condition=lambda state: state.data.get('next_tool') == 'write_file'
    )
    
    graph.add_edge(
        "reasoning_agent", 
        "create_directory",
        condition=lambda state: state.data.get('next_tool') == 'create_directory'
    )
    
    graph.add_edge(
        "reasoning_agent", 
        "execute_bash_command",
        condition=lambda state: state.data.get('next_tool') == 'execute_bash_command'
    )
    
    graph.add_edge(
        "reasoning_agent", 
        "task_complete",
        condition=lambda state: state.data.get('next_tool') == 'task_complete'
    )
    
    # Add edges from all tool nodes back to reasoning agent for next step analysis
    # Each tool node can transition back to reasoning agent (except task_complete which is terminal)
    # But first remove the completed task from the plan
    def create_plan_progression_condition():
        """Create a condition that removes completed task and continues to next task."""
        def condition_with_plan_progression(state):
            # Remove the completed task from the plan
            plan = state.data.get('plan', [])
            if plan and isinstance(plan, list) and len(plan) > 0:
                completed_task = plan.pop(0)
                state.data['plan'] = plan
                state.add_message(f"Removed completed task from plan: {completed_task}")
                state.add_message(f"Remaining tasks in plan: {len(plan)}")
                
                # Continue to next task if plan is not empty
                return len(plan) > 0
            else:
                state.add_message("No tasks to remove from plan")
                return False
        return condition_with_plan_progression
    
    graph.add_edge(
        "list_files",
        "reasoning_agent",
        condition=create_plan_progression_condition()
    )
    
    graph.add_edge(
        "write_file",
        "reasoning_agent",
        condition=create_plan_progression_condition()
    )
    
    graph.add_edge(
        "create_directory",
        "reasoning_agent",
        condition=create_plan_progression_condition()
    )
    
    graph.add_edge(
        "execute_bash_command",
        "reasoning_agent",
        condition=create_plan_progression_condition()
    )
    
    # Note: task_complete node has no outgoing edges - it's a terminal node
    
    print(f"   Graph nodes: {graph.get_node_count()}")
    print(f"   Graph edges: {graph.get_edge_count()}")
    print("   Configured graph to terminate when task_complete tool is selected")
    
    # Execute the graph
    print("7. Executing graph...")
    print("   Starting from planner_agent node...\n")
    
    try:
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="demo_thread_001",
            start_node="planner_agent",
            max_iterations=10
        )
        
        print("8. Graph execution completed successfully!\n")
        
        # Display results
        print("=== EXECUTION RESULTS ===")
        
        print("\n📋 Message History:")
        for i, message in enumerate(final_state.message_history, 1):
            print(f"   {i:2d}. {message}")
        
        print("\n🧠 Generated LLM Prompt:")
        if 'last_prompt' in final_state.data:
            prompt_lines = final_state.data['last_prompt'].split('\n')
            for line in prompt_lines:
                print(f"   {line}")
        else:
            print("   No prompt generated")
        
        print("\n🤖 LLM Response:")
        if 'llm_response' in final_state.data:
            response_lines = final_state.data['llm_response'].split('\n')
            for line in response_lines:
                print(f"   {line}")
        else:
            print("   No LLM response (may have used fallback reasoning)")
        
        print("\n🔧 Tool Execution Results:")
        if final_state.tool_outputs:
            for tool_name, output in final_state.tool_outputs.items():
                print(f"   Tool: {tool_name}")
                print(f"   Output: {output[:200]}{'...' if len(str(output)) > 200 else ''}")
        else:
            print("   No tool outputs")
        
        print("\n📊 Final State Data:")
        for key, value in final_state.data.items():
            if key not in ['last_prompt', 'llm_response']:  # Skip the long texts
                print(f"   {key}: {value}")
        
        # Final verification check - ensure task completion
        print("\n🔍 Final Verification Check:")
        error_found = False
        task_completed = 'task_complete' in final_state.tool_outputs
        
        # Check for errors in tool outputs
        if final_state.tool_outputs:
            for tool_name, output in final_state.tool_outputs.items():
                if isinstance(output, str) and "Error:" in output:
                    error_found = True
                    print(f"   ❌ Error detected in {tool_name}: {output}")
            
            if not error_found:
                print("   ✅ No errors detected in tool outputs")
        
        # Check if the task was completed successfully
        if task_completed:
            print("   ✅ Task completion tool executed successfully")
            
            # Verify the output directory was created
            output_dir = project_root / "output"
            if output_dir.exists() and output_dir.is_dir():
                print("   ✅ Directory 'output' created successfully")
                
                # Verify the log.txt file was created inside the output directory
                log_file = output_dir / "log.txt"
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content == "System test OK":
                        print("   ✅ File 'output/log.txt' created with correct content")
                    else:
                        print(f"   ❌ File 'output/log.txt' has incorrect content: {content}")
                        error_found = True
                else:
                    print("   ❌ File 'output/log.txt' was not created")
                    error_found = True
            else:
                print("   ❌ Directory 'output' was not created")
                error_found = True
        else:
            print("   ❌ Task completion tool was not executed")
            error_found = True
        
        if not final_state.tool_outputs:
            print("   ⚠️  No tool outputs to verify")
            error_found = True
        
        if error_found:
            # Display prominent failure message
            print("\n" + "="*50)
            print("*" * 50)
            print("**                                              **")
            print("**            DEMONSTRATION FAILED             **")
            print("**                                              **")
            print("*" * 50)
            print("="*50)
            print("\n❌ Errors were found in tool execution outputs.")
            print("The demonstration did not complete successfully.")
            sys.exit(1)
        
        print("\n✅ Demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during graph execution: {e}")
        print("\n" + "="*50)
        print("*" * 50)
        print("**                                              **")
        print("**            DEMONSTRATION FAILED             **")
        print("**                                              **")
        print("*" * 50)
        print("="*50)
        sys.exit(1)
    
    # Clean up
    print("\n9. Cleaning up...")
    checkpoint_file = project_root / "demo_checkpoints.db"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("   Removed checkpoint database")
    
    # Keep the demo files for user inspection
    output_dir = project_root / "output"
    if output_dir.exists():
        print(f"   Demo directory preserved for inspection: {output_dir}")
        log_file = output_dir / "log.txt"
        if log_file.exists():
            print(f"   Demo file preserved for inspection: {log_file}")


if __name__ == "__main__":
    main()