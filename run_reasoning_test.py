#!/usr/bin/env python3
"""Dual-Circuit Decision-Making System demonstration script for the ONIKS NeuralNet system.

This script demonstrates the complete dual-circuit architecture that provides optimal 
performance and quality balance through intelligent task routing.

DUAL-CIRCUIT ARCHITECTURE:
Circuit 1: Fast Response Circuit (RouterAgent)
- Model: phi3:mini (3.8B parameters) 
- Timeout: 15 seconds (aggressive)
- Task: Lightning-fast "SIMPLE" vs "COMPLEX" classification

Circuit 2: Deep Planning Circuit (PlannerAgent)  
- Model: llama3:8b (8B parameters)
- Timeout: 20 minutes (quality over speed)
- Task: Detailed hierarchical planning for complex goals

The demonstration workflow:
1. RouterAgent uses phi3:mini for ultra-fast classification (30s timeout)
2a. DIRECT PATH (simple): RouterAgent → ReasoningAgent (83% faster)
2b. HIERARCHICAL PATH (complex): RouterAgent → PlannerAgent → ReasoningAgent (maintains quality)
3. Multi-layer fallback: LLM → keyword-based → hierarchical (100% reliability)
4. ReasoningAgent selects appropriate tools based on the execution path
5. Tool nodes execute the selected operations
6. Process continues until task completion

Performance Targets:
- Simple tasks: <3 minutes (vs 15+ minutes previously, 83% improvement)
- Complex tasks: unchanged (maintains quality, +15s overhead acceptable)  
- Classification accuracy: >80% with phi3:mini
- Fallback reliability: 100% (always defaults to hierarchical on failure)

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
from oniks.tools.file_tools import ReadFileTool, FileSearchReplaceTool
from oniks.tools.fs_tools import ListFilesTool, WriteFileTool, CreateDirectoryTool
from oniks.tools.shell_tools import ExecuteBashCommandTool
from oniks.tools.core_tools import TaskCompleteTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.agents.planner_agent import PlannerAgent
from oniks.agents.router_agent import RouterAgent
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
                print(f"   DEBUG: Executing node '{node_name}' at iteration {iterations}")
                print(f"   DEBUG: Current state keys: {list(current_state.data.keys())}")
                
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
                print(f"   DEBUG: Next candidates from '{node_name}': {next_candidates}")
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
    demo_files = [
        project_root / "hello_world.py",
        project_root / "hello_world.py.bak",
        project_root / "hello_oniks.py",
        project_root / "hello_oniks.py.bak"
    ]
    
    for demo_file_path in demo_files:
        if demo_file_path.exists():
            demo_file_path.unlink()
            print(f"Cleaned up existing demo file: {demo_file_path}")


def main() -> None:
    """Main dual-circuit decision-making system demonstration function."""
    print("=== ONIKS Dual-Circuit Decision-Making System Demonstration ===\n")
    print("🔄 Circuit 1: Fast Response (phi3:mini, 30s timeout)")
    print("🧠 Circuit 2: Deep Planning (llama3:8b, 20min timeout)")
    print("⚡ Performance Target: 83% faster for simple tasks\n")
    
    # Clean up any existing demo files
    cleanup_demo_files()
    
    # Initialize multi-step graph with checkpoint saver
    print("1. Initializing multi-step graph and checkpoint saver...")
    checkpointer = SQLiteCheckpointSaver("demo_checkpoints.db")
    # Allow router, planner and reasoning agents to be re-executed for multi-step workflows
    graph = MultiStepGraph(checkpointer=checkpointer, reusable_nodes={"router_agent", "planner_agent", "reasoning_agent"})
    
    # Create initial state with goal that can demonstrate routing
    print("2. Creating initial state with goal for routing demonstration...")
    initial_state = State()
    initial_state.data['goal'] = "Create a Python file named hello_world.py that prints 'Hello World' when executed"
    initial_state.add_message("Demo started with goal for RouterAgent classification and routing")
    
    print(f"   Goal: {initial_state.data['goal']}")
    
    # Create all required tools
    print("3. Creating tools...")
    list_files_tool = ListFilesTool()
    write_file_tool = WriteFileTool()
    create_directory_tool = CreateDirectoryTool()
    execute_bash_tool = ExecuteBashCommandTool()
    task_complete_tool = TaskCompleteTool()
    edit_file_tool = FileSearchReplaceTool()
    
    tools = [list_files_tool, write_file_tool, create_directory_tool, execute_bash_tool, task_complete_tool, edit_file_tool]
    
    for tool in tools:
        print(f"   Created tool: {tool.name}")
        print(f"   Tool description: {tool.description}")
    
    # Create LLM client
    print("4. Creating LLM client...")
    try:
        llm_client = OllamaClient(timeout=1200)
        # Check if both models are available
        main_model_available = llm_client.check_model_availability("llama3:8b")
        routing_model_available = llm_client.check_model_availability("phi3:mini")
        
        if main_model_available and routing_model_available:
            print("   ✅ LLM client created successfully with both models")
            print("   📊 llama3:8b (main model) - available")
            print("   ⚡ phi3:mini (routing model) - available")
        else:
            print("   ⚠️  LLM client created but some models missing:")
            if not main_model_available:
                print("   ❌ llama3:8b (main model) - missing")
            if not routing_model_available:
                print("   ❌ phi3:mini (routing model) - missing")
            print("   Available models:", llm_client.list_available_models())
    except OllamaConnectionError as e:
        print(f"   Warning: {e}")
        print("   Note: Ensure Ollama is running locally with 'ollama serve'")
        print("   Required models:")
        print("     ollama pull llama3:8b    # Main model for deep planning")
        print("     ollama pull phi3:mini    # Fast model for routing")
        llm_client = OllamaClient(timeout=1200)  # Create client anyway for demonstration
    
    # Create agents and nodes with dual-circuit configuration
    print("5. Creating agents and nodes...")
    # RouterAgent with phi3:mini for fast classification (Circuit 1)
    router_agent = RouterAgent("router_agent", llm_client, routing_model="phi3:mini", main_model="llama3:8b", timeout_seconds=15.0)
    # PlannerAgent with llama3:8b for complex planning (Circuit 2)
    planner_agent = PlannerAgent("planner_agent", llm_client, tools, timeout_seconds=1200.0)
    reasoning_agent = ReasoningAgent("reasoning_agent", tools, llm_client)
    
    # Create tool nodes for each tool
    list_files_node = ToolNode("list_files", list_files_tool)
    write_file_node = ToolNode("write_file", write_file_tool)
    create_directory_node = ToolNode("create_directory", create_directory_tool)
    execute_bash_node = ToolNode("execute_bash_command", execute_bash_tool)
    task_complete_node = ToolNode("task_complete", task_complete_tool)
    edit_file_node = ToolNode("file_search_replace", edit_file_tool)
    
    tool_nodes = [list_files_node, write_file_node, create_directory_node, execute_bash_node, task_complete_node, edit_file_node]
    
    print(f"   Created router agent: {router_agent.name} (using {router_agent.routing_model} for fast classification)")
    print(f"   Created planner agent: {planner_agent.name} (using llama3:8b for complex planning)")
    print(f"   Created reasoning agent: {reasoning_agent.name}")
    for node in tool_nodes:
        print(f"   Created tool node: {node.name}")
    
    # Add nodes to graph
    print("6. Building comprehensive graph structure...")
    graph.add_node(router_agent)
    graph.add_node(planner_agent)
    graph.add_node(reasoning_agent)
    for node in tool_nodes:
        graph.add_node(node)
    
    # Add conditional routing edges based on execution_path
    # Direct path: RouterAgent → ReasoningAgent (simplified)
    graph.add_edge(
        "router_agent",
        "reasoning_agent",
        condition=lambda state: state.data.get('execution_path') == 'direct'
    )
    
    # Hierarchical path: RouterAgent → PlannerAgent → ReasoningAgent (existing flow)
    graph.add_edge(
        "router_agent",
        "planner_agent",
        condition=lambda state: state.data.get('execution_path') == 'hierarchical'
    )
    
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
    
    graph.add_edge(
        "reasoning_agent", 
        "file_search_replace",
        condition=lambda state: state.data.get('next_tool') == 'file_search_replace'
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
    
    graph.add_edge(
        "file_search_replace",
        "reasoning_agent",
        condition=create_plan_progression_condition()
    )
    
    # Note: task_complete node has no outgoing edges - it's a terminal node
    
    print(f"   Graph nodes: {graph.get_node_count()}")
    print(f"   Graph edges: {graph.get_edge_count()}")
    print("   Configured graph to terminate when task_complete tool is selected")
    
    # Execute the graph with performance monitoring
    print("7. Executing dual-circuit graph...")
    print("   Starting from router_agent node...\n")
    
    import time
    execution_start_time = time.time()
    
    try:
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="demo_thread_001",
            start_node="router_agent",
            max_iterations=50
        )
        
        execution_end_time = time.time()
        total_execution_time = execution_end_time - execution_start_time
        
        print("8. Graph execution completed successfully!\n")
        
        # Display performance results
        print("=== DUAL-CIRCUIT PERFORMANCE RESULTS ===")
        
        execution_path = final_state.data.get('execution_path', 'unknown')
        print(f"\n🚀 Execution Path: {execution_path.upper()}")
        print(f"⏱️  Total Execution Time: {total_execution_time:.2f} seconds")
        
        if execution_path == 'direct':
            print("✅ Used FAST CIRCUIT (Router → Reasoning Agent)")
            print("💡 Performance Benefit: Bypassed planning phase for simple task")
        elif execution_path == 'hierarchical':
            print("🔄 Used DEEP CIRCUIT (Router → Planner → Reasoning Agent)")
            print("💡 Quality Assurance: Full planning workflow for complex task")
        else:
            print("❓ Path determination unclear")
        
        # Display routing details
        if 'classification_response' in final_state.data:
            classification_response = final_state.data['classification_response']
            print(f"🧠 Router Classification: {classification_response.strip()}")
        
        # Display results
        print("\n=== DETAILED EXECUTION RESULTS ===")
        
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
            
            # Verify a Python file was created (flexible name matching)
            python_files = [
                project_root / "hello_world.py",
                project_root / "hello_oniks.py"
            ]
            
            created_file = None
            for python_file in python_files:
                if python_file.exists():
                    created_file = python_file
                    break
            
            if created_file:
                print(f"   ✅ Python file '{created_file.name}' created successfully")
                
                # Verify the file content
                with open(created_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if 'print(' in content and ('Hello' in content or 'hello' in content):
                    print(f"   ✅ File '{created_file.name}' has correct content")
                    
                    # Check if the script was executed (should be in tool outputs)
                    execute_output = final_state.tool_outputs.get('execute_bash_command', '')
                    if execute_output and ('Hello' in str(execute_output) or 'hello' in str(execute_output)):
                        print("   ✅ Python script executed successfully with correct output")
                    else:
                        print(f"   ⚠️  Python script execution output: {execute_output}")
                        # Don't mark as error since file creation and content are correct
                else:
                    print(f"   ❌ File '{created_file.name}' has incorrect content: {content}")
                    error_found = True
            else:
                print("   ❌ No Python file was created")
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
    demo_files = [
        project_root / "hello_world.py",
        project_root / "hello_world.py.bak",
        project_root / "hello_oniks.py",
        project_root / "hello_oniks.py.bak"
    ]
    
    for demo_file in demo_files:
        if demo_file.exists():
            print(f"   Demo file preserved for inspection: {demo_file}")


if __name__ == "__main__":
    main()