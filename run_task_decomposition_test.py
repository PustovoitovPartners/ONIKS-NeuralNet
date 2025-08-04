#!/usr/bin/env python3
"""Task Decomposition Layer demonstration script for the ONIKS NeuralNet framework.

This script demonstrates the new Task Decomposition Layer approach where:
1. PlannerAgent decomposes complex goals into atomic subtasks
2. ReasoningAgent works with single subtasks from plan[0]
3. Deterministic code manages task removal and completion
4. LLM is used for creative decomposition and simple tool selection

The demonstration workflow:
1. PlannerAgent analyzes the goal and creates a structured task plan
2. ReasoningAgent processes the first task from plan[0]
3. Tool executes the selected action
4. System removes completed task from plan[0]
5. Loop continues until plan is empty and task_complete is reached

Usage:
    python run_task_decomposition_test.py
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.core.checkpoint import SQLiteCheckpointSaver
from typing import Set, List, Dict, Optional
from oniks.tools.file_tools import ReadFileTool
from oniks.tools.fs_tools import ListFilesTool, WriteFileTool
from oniks.tools.shell_tools import ExecuteBashCommandTool
from oniks.tools.core_tools import TaskCompleteTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.agents.planner_agent import PlannerAgent
from oniks.llm.client import OllamaClient, OllamaConnectionError


class TaskDecompositionGraph(Graph):
    """Extended Graph class that implements the Task Decomposition Layer.
    
    This class modifies the standard graph execution to manage task decomposition:
    1. Allows planner and reasoning agents to be re-executed
    2. Automatically removes completed tasks from the plan
    3. Provides deterministic state management for reliable execution
    """
    
    def __init__(self, checkpointer: Optional = None, reusable_nodes: Optional[Set[str]] = None):
        """Initialize a TaskDecompositionGraph.
        
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
        """Execute the graph with task decomposition layer support.
        
        This method extends the base graph execution to support:
        1. Task decomposition and plan management  
        2. Automatic task removal after successful tool execution
        3. Re-execution of planner and reasoning agents
        4. Deterministic completion detection
        
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
        
        current_state.add_message(f"Starting task decomposition graph execution from node: {start_node}")
        
        # Save initial state if checkpointer is available
        if self.checkpointer:
            current_state.add_message(f"Saving initial checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        while current_nodes and iterations < max_iterations:
            iterations += 1
            next_nodes = []
            
            for node_name in current_nodes:
                # Allow re-execution of reusable nodes (planner and reasoning agents)
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
                
                # CRITICAL: Task removal logic after successful tool execution
                if self._is_tool_node(node_name) and self._was_tool_successful(node_name, current_state):
                    self._remove_completed_task(current_state)
                
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
                f"Task decomposition graph execution exceeded maximum iterations ({max_iterations}). "
                "Possible infinite loop detected."
            )
        
        current_state.add_message("Task decomposition graph execution completed")
        
        # Save final state
        if self.checkpointer:
            current_state.add_message(f"Saving final checkpoint for thread: {thread_id}")
            self.checkpointer.save(thread_id, current_state)
        
        return current_state
    
    def _is_tool_node(self, node_name: str) -> bool:
        """Check if a node is a tool node (not an agent)."""
        return (node_name != "planner_agent" and 
                node_name != "reasoning_agent")
    
    def _was_tool_successful(self, node_name: str, state: State) -> bool:
        """Check if a tool execution was successful (no errors)."""
        if node_name not in state.tool_outputs:
            return False
        
        output = state.tool_outputs[node_name]
        if isinstance(output, str) and output.startswith("Error:"):
            return False
        
        return True
    
    def _remove_completed_task(self, state: State) -> None:
        """Remove the first task from the plan after successful tool execution."""
        plan = state.data.get('plan', [])
        
        if plan and isinstance(plan, list) and len(plan) > 0:
            completed_task = plan.pop(0)
            state.data['plan'] = plan
            state.add_message(f"Removed completed task from plan: {completed_task}")
            state.add_message(f"Remaining tasks in plan: {len(plan)}")
        else:
            state.add_message("No tasks to remove from plan")


def cleanup_demo_files() -> None:
    """Clean up any existing demo files before starting."""
    demo_file_path = project_root / "hello.txt"
    
    if demo_file_path.exists():
        demo_file_path.unlink()
        print(f"Cleaned up existing demo file: {demo_file_path}")


def main() -> None:
    """Main task decomposition demonstration function."""
    print("=== ONIKS Task Decomposition Layer Demonstration ===\n")
    
    # Clean up any existing demo files
    cleanup_demo_files()
    
    # Initialize task decomposition graph with checkpoint saver
    print("1. Initializing task decomposition graph and checkpoint saver...")
    checkpointer = SQLiteCheckpointSaver("task_decomposition_checkpoints.db")
    # Allow both planner and reasoning agents to be re-executed
    graph = TaskDecompositionGraph(
        checkpointer=checkpointer, 
        reusable_nodes={"planner_agent", "reasoning_agent"}
    )
    
    # Create initial state with complex goal
    print("2. Creating initial state with complex goal...")
    initial_state = State()
    initial_state.data['goal'] = "Create a file named 'hello.txt' with the content 'Hello ONIKS!', then display its content to the console."
    initial_state.add_message("Task decomposition demo started with complex goal")
    
    print(f"   Goal: {initial_state.data['goal']}")
    
    # Create all required tools
    print("3. Creating tools...")
    list_files_tool = ListFilesTool()
    write_file_tool = WriteFileTool()
    execute_bash_tool = ExecuteBashCommandTool()
    task_complete_tool = TaskCompleteTool()
    
    tools = [list_files_tool, write_file_tool, execute_bash_tool, task_complete_tool]
    
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
    planner_agent = PlannerAgent("planner_agent", llm_client)
    reasoning_agent = ReasoningAgent("reasoning_agent", tools, llm_client)
    
    # Create tool nodes for each tool
    list_files_node = ToolNode("list_files", list_files_tool)
    write_file_node = ToolNode("write_file", write_file_tool)
    execute_bash_node = ToolNode("execute_bash_command", execute_bash_tool)
    task_complete_node = ToolNode("task_complete", task_complete_tool)
    
    tool_nodes = [list_files_node, write_file_node, execute_bash_node, task_complete_node]
    
    print(f"   Created planner agent: {planner_agent.name}")
    print(f"   Created reasoning agent: {reasoning_agent.name}")
    for node in tool_nodes:
        print(f"   Created tool node: {node.name}")
    
    # Add nodes to graph
    print("6. Building task decomposition graph structure...")
    graph.add_node(planner_agent)
    graph.add_node(reasoning_agent)
    for node in tool_nodes:
        graph.add_node(node)
    
    # Add edge from planner to reasoning agent (planner runs first)
    graph.add_edge(
        "planner_agent", 
        "reasoning_agent",
        condition=lambda state: True  # Always proceed to reasoning after planning
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
        "execute_bash_command",
        condition=lambda state: state.data.get('next_tool') == 'execute_bash_command'
    )
    
    graph.add_edge(
        "reasoning_agent", 
        "task_complete",
        condition=lambda state: state.data.get('next_tool') == 'task_complete'
    )
    
    # Add edges from all tool nodes back to reasoning agent for next task analysis
    # (except task_complete which is terminal)
    graph.add_edge(
        "list_files",
        "reasoning_agent",
        condition=lambda state: True  # Always return to reasoning agent
    )
    
    graph.add_edge(
        "write_file",
        "reasoning_agent",
        condition=lambda state: True  # Always return to reasoning agent
    )
    
    graph.add_edge(
        "execute_bash_command",
        "reasoning_agent",
        condition=lambda state: True  # Always return to reasoning agent
    )
    
    # Note: task_complete node has no outgoing edges - it's a terminal node
    
    print(f"   Graph starts with planner_agent for task decomposition")
    print(f"   Graph nodes: {graph.get_node_count()}")
    print(f"   Graph edges: {graph.get_edge_count()}")
    print("   Configured deterministic task removal after tool execution")
    
    # Execute the graph
    print("7. Executing task decomposition graph...")
    print("   Starting from planner_agent node...\n")
    
    try:
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="task_decomposition_demo_001",
            start_node="planner_agent",
            max_iterations=20
        )
        
        print("8. Task decomposition graph execution completed successfully!\n")
        
        # Display results
        print("=== EXECUTION RESULTS ===")
        
        print("\n📋 Message History:")
        for i, message in enumerate(final_state.message_history, 1):
            print(f"   {i:2d}. {message}")
        
        print("\n🧠 Task Decomposition Results:")
        if 'plan' in final_state.data:
            original_plan = final_state.data.get('plan', [])
            print(f"   Final plan state: {len(original_plan)} remaining tasks")
            if original_plan:
                for i, task in enumerate(original_plan, 1):
                    print(f"      {i}. {task}")
            else:
                print("   ✅ All tasks completed successfully")
        
        if 'decomposition_prompt' in final_state.data:
            print("\n📝 Generated Decomposition Prompt (first 500 chars):")
            prompt = final_state.data['decomposition_prompt']
            print(f"   {prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        
        if 'decomposition_response' in final_state.data:
            print("\n🤖 LLM Decomposition Response:")
            response = final_state.data['decomposition_response']
            print(f"   {response}")
        
        print("\n🔧 Tool Execution Results:")
        if final_state.tool_outputs:
            for tool_name, output in final_state.tool_outputs.items():
                print(f"   Tool: {tool_name}")
                print(f"   Output: {output[:200]}{'...' if len(str(output)) > 200 else ''}")
        else:
            print("   No tool outputs")
        
        print("\n📊 Final State Data:")
        for key, value in final_state.data.items():
            if key not in ['decomposition_prompt', 'decomposition_response', 'last_prompt', 'llm_response']:
                print(f"   {key}: {value}")
        
        # Final verification check
        print("\n🔍 Task Decomposition Verification:")
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
            
            # Verify the hello.txt file was created
            hello_file = project_root / "hello.txt"
            if hello_file.exists():
                with open(hello_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content == "Hello ONIKS!":
                    print("   ✅ File hello.txt created with correct content")
                else:
                    print(f"   ❌ File hello.txt has incorrect content: {content}")
                    error_found = True
            else:
                print("   ❌ File hello.txt was not created")
                error_found = True
            
            # Check if file content was displayed
            bash_output_found = False
            if final_state.tool_outputs:
                for tool_name, output in final_state.tool_outputs.items():
                    if tool_name == 'execute_bash_command' and isinstance(output, str):
                        if 'Hello ONIKS!' in output:
                            print("   ✅ File content displayed successfully via bash command")
                            bash_output_found = True
                            break
            
            if not bash_output_found:
                print("   ❌ File content was not displayed via bash command")
                error_found = True
        else:
            print("   ❌ Task completion tool was not executed")
            error_found = True
        
        # Check plan completion
        remaining_tasks = len(final_state.data.get('plan', []))
        if remaining_tasks == 0:
            print("   ✅ All tasks removed from plan successfully")
        else:
            print(f"   ❌ {remaining_tasks} tasks still remaining in plan")
            error_found = True
        
        if not final_state.tool_outputs:
            print("   ⚠️  No tool outputs to verify")
            error_found = True
        
        if error_found:
            # Display prominent failure message
            print("\n" + "="*50)
            print("*" * 50)
            print("**                                              **")
            print("**       TASK DECOMPOSITION DEMO FAILED        **")
            print("**                                              **")
            print("*" * 50)
            print("="*50)
            print("\n❌ Errors were found in task decomposition execution.")
            print("The demonstration did not complete successfully.")
            sys.exit(1)
        
        print("\n✅ Task Decomposition Layer demonstration completed successfully!")
        print("\n🎯 Key Achievements:")
        print("   • Complex goal successfully decomposed into atomic subtasks")
        print("   • Deterministic task management implemented")
        print("   • LLM used for creative decomposition and simple tool selection")
        print("   • Reliable execution without state comparison complexity")
        
    except Exception as e:
        print(f"❌ Error during task decomposition graph execution: {e}")
        print("\n" + "="*50)
        print("*" * 50)
        print("**                                              **")
        print("**       TASK DECOMPOSITION DEMO FAILED        **")
        print("**                                              **")
        print("*" * 50)
        print("="*50)
        sys.exit(1)
    
    # Clean up
    print("\n9. Cleaning up...")
    checkpoint_file = project_root / "task_decomposition_checkpoints.db"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("   Removed checkpoint database")
    
    # Keep the demo file for user inspection
    hello_file = project_root / "hello.txt"
    if hello_file.exists():
        print(f"   Demo file preserved for inspection: {hello_file}")


if __name__ == "__main__":
    main()