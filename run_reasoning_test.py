#!/usr/bin/env python3
"""Demonstration script for testing the ReasoningAgent system.

This script demonstrates the integration of ReasoningAgent with the graph execution
framework, showing how intelligent agents can analyze goals, select tools, and
coordinate with tool nodes to achieve objectives.

The script creates a simple workflow:
1. ReasoningAgent analyzes a goal and determines which tool to use
2. ToolNode executes the selected tool with provided arguments
3. The process can continue with additional reasoning steps

Usage:
    python run_reasoning_test.py
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.core.checkpoint import SQLiteCheckpointSaver
from oniks.tools.file_tools import ReadFileTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.llm.client import OllamaClient, OllamaConnectionError


def create_test_file() -> None:
    """Create a test file for the demonstration."""
    test_file_path = project_root / "task.txt"
    
    if not test_file_path.exists():
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("This is a test file for the ReasoningAgent demonstration.\n")
            f.write("The agent successfully identified the need to read this file!\n")
            f.write("Content: Task completed successfully.\n")
        print(f"Created test file: {test_file_path}")
    else:
        print(f"Test file already exists: {test_file_path}")


def main() -> None:
    """Main demonstration function."""
    print("=== ONIKS ReasoningAgent Demonstration ===\n")
    
    # Create test file for demonstration
    create_test_file()
    
    # Initialize graph with checkpoint saver
    print("1. Initializing graph and checkpoint saver...")
    checkpointer = SQLiteCheckpointSaver("demo_checkpoints.db")
    graph = Graph(checkpointer=checkpointer)
    
    # Create initial state with goal
    print("2. Creating initial state with goal...")
    initial_state = State()
    initial_state.data['goal'] = 'Read the contents of file task.txt'
    initial_state.add_message("Demo started with file reading goal")
    
    print(f"   Goal: {initial_state.data['goal']}")
    
    # Create tools
    print("3. Creating tools...")
    read_file_tool = ReadFileTool()
    print(f"   Created tool: {read_file_tool.name}")
    print(f"   Tool description: {read_file_tool.description}")
    
    # Create LLM client
    print("4. Creating LLM client...")
    try:
        llm_client = OllamaClient()
        # Check if the default model is available
        if llm_client.check_model_availability():
            print("   LLM client created successfully with model 'tinyllama'")
        else:
            print("   Warning: Model 'tinyllama' not found, but client created")
            print("   Available models:", llm_client.list_available_models())
    except OllamaConnectionError as e:
        print(f"   Warning: {e}")
        print("   Note: Ensure Ollama is running locally with 'ollama serve'")
        print("   And that you have pulled the tinyllama model with 'ollama pull tinyllama'")
        llm_client = OllamaClient()  # Create client anyway for demonstration
    
    # Create agents and nodes
    print("5. Creating agents and nodes...")
    reasoning_agent = ReasoningAgent("reasoning_agent", [read_file_tool], llm_client)
    file_reader_node = ToolNode("file_reader", read_file_tool)
    
    print(f"   Created reasoning agent: {reasoning_agent.name}")
    print(f"   Created tool node: {file_reader_node.name}")
    
    # Add nodes to graph
    print("6. Building graph structure...")
    graph.add_node(reasoning_agent)
    graph.add_node(file_reader_node)
    
    # Add edges with conditions
    # From reasoning agent to file reader when next_tool is 'read_file'
    graph.add_edge(
        "reasoning_agent", 
        "file_reader",
        condition=lambda state: state.data.get('next_tool') == 'read_file'
    )
    
    # From file reader back to reasoning agent for potential next step
    graph.add_edge(
        "file_reader",
        "reasoning_agent",
        condition=lambda state: state.data.get('continue_reasoning', False)
    )
    
    print(f"   Graph nodes: {graph.get_node_count()}")
    print(f"   Graph edges: {graph.get_edge_count()}")
    
    # Execute the graph
    print("7. Executing graph...")
    print("   Starting from reasoning_agent node...\n")
    
    try:
        final_state = graph.execute(
            initial_state=initial_state,
            thread_id="demo_thread_001",
            start_node="reasoning_agent",
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
                print(f"   Output: {output}")
        else:
            print("   No tool outputs")
        
        print("\n📊 Final State Data:")
        for key, value in final_state.data.items():
            if key not in ['last_prompt', 'llm_response']:  # Skip the long texts
                print(f"   {key}: {value}")
        
        # Final verification check - ensure no errors in tool outputs
        print("\n🔍 Final Verification Check:")
        error_found = False
        if final_state.tool_outputs:
            for tool_name, output in final_state.tool_outputs.items():
                if isinstance(output, str) and "Error:" in output:
                    error_found = True
                    print(f"   ❌ Error detected in {tool_name}: {output}")
                    break
            
            if not error_found:
                print("   ✅ No errors detected in tool outputs")
        else:
            print("   ⚠️  No tool outputs to verify")
        
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


if __name__ == "__main__":
    main()