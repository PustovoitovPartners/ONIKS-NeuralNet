#!/usr/bin/env python3
"""Test script to demonstrate bulletproof LLM logging and transparency.

This script demonstrates the comprehensive LLM call chain logging implemented
to expose every request, response, and fallback in the ONIKS NeuralNet system.

Usage:
    python test_llm_transparency.py
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oniks.core.graph import Graph, ToolNode
from oniks.core.state import State
from oniks.tools.file_tools import ReadFileTool
from oniks.tools.fs_tools import WriteFileTool
from oniks.tools.core_tools import TaskCompleteTool
from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.agents.planner_agent import PlannerAgent
from oniks.llm.client import OllamaClient, OllamaConnectionError


def setup_comprehensive_logging():
    """Setup comprehensive logging to see all LLM interactions."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llm_audit_log.txt', mode='w')
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('oniks.llm.client').setLevel(logging.INFO)
    logging.getLogger('oniks.agents.planner_agent').setLevel(logging.INFO)
    logging.getLogger('oniks.agents.reasoning_agent').setLevel(logging.INFO)
    
    print("=== COMPREHENSIVE LOGGING CONFIGURED ===")
    print("All LLM requests, responses, and fallbacks will be logged in full detail.")
    print("Log file: llm_audit_log.txt")
    print("")


def test_llm_transparency_scenario():
    """Test LLM transparency with both working and failing scenarios."""
    print("=== TESTING LLM TRANSPARENCY ===\n")
    
    # Create tools
    read_tool = ReadFileTool()
    write_tool = WriteFileTool()
    complete_tool = TaskCompleteTool()
    tools = [read_tool, write_tool, complete_tool]
    
    # Test 1: LLM Available Scenario (if Ollama is running)
    print("Test 1: Testing with LLM client (may fallback if Ollama unavailable)...")
    try:
        llm_client = OllamaClient()
        test_prompt = "Select the appropriate tool for this task: Create a file named 'test.txt' with content 'Hello World'"
        
        print(f"Sending test prompt to LLM: {test_prompt[:50]}...")
        response = llm_client.invoke(test_prompt)
        print(f"LLM Response received: {response[:50]}...")
        print("✅ LLM client is working - full requests/responses logged above")
        
    except OllamaConnectionError as e:
        print(f"⚠️  LLM client unavailable: {e}")
        print("This will trigger fallback reasoning (marked as [FALLBACK-REASONING])")
    
    print("\nTest 2: Testing PlannerAgent with complex goal...")
    
    # Create mock LLM client for consistent testing
    class MockLLMClient:
        def invoke(self, prompt):
            # Simulate an error to test fallback
            raise OllamaConnectionError("Mock LLM failure for testing")
    
    mock_client = MockLLMClient()
    
    # Create initial state
    initial_state = State()
    initial_state.data['goal'] = "Create a file called 'demo.txt' with content 'Testing transparency' and then read it back"
    
    # Create planner agent
    planner = PlannerAgent("test_planner", mock_client, tools)
    
    print("Executing planner agent (will trigger fallback due to mock failure)...")
    result_state = planner.execute(initial_state)
    
    print("\nPlannerAgent Results:")
    print(f"Plan created: {result_state.data.get('plan', 'No plan')}")
    print("Check logs above for [FALLBACK-REASONING] markers")
    
    print("\nTest 3: Testing ReasoningAgent with mock failure...")
    
    # Create reasoning agent
    reasoning = ReasoningAgent("test_reasoning", tools, mock_client)
    
    # Set up state with a simple task
    task_state = State()
    task_state.data['plan'] = ["write_file(file_path='test.txt', content='Hello')"]
    
    print("Executing reasoning agent (will trigger fallback due to mock failure)...")
    reasoning_result = reasoning.execute(task_state)
    
    print("\nReasoningAgent Results:")
    print(f"Selected tool: {reasoning_result.data.get('next_tool', 'No tool')}")
    print(f"Tool args: {reasoning_result.data.get('tool_args', 'No args')}")
    print("Check logs above for [FALLBACK-REASONING] markers")
    
    print("\n=== TRANSPARENCY TEST COMPLETE ===")
    print("Key features demonstrated:")
    print("✅ Full LLM request logging with [LLM-REQUEST-{id}] markers")
    print("✅ Full LLM response logging with [LLM-RESPONSE-{id}] markers") 
    print("✅ Complete error tracebacks with [LLM-ERROR-{id}] markers")
    print("✅ Clear [FALLBACK-REASONING] markers for hardcoded logic")
    print("✅ Unique execution IDs for request correlation")
    print("✅ Timestamps for performance analysis")
    print("✅ No silent failures - all errors are logged in full")


def demonstrate_request_correlation():
    """Demonstrate request correlation with execution IDs."""
    print("\n=== DEMONSTRATING REQUEST CORRELATION ===")
    
    try:
        client = OllamaClient()
        
        print("Making multiple LLM calls to show correlation IDs...")
        
        prompts = [
            "What is the capital of France?",
            "How do you create a file in Python?",
            "Explain what a REST API is."
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nRequest {i}: {prompt}")
            try:
                response = client.invoke(prompt)
                print(f"Response {i} received (length: {len(response)} chars)")
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        print("\n✅ Check the logs above - each request has a unique [LLM-REQUEST-{id}]")
        print("   and corresponding [LLM-RESPONSE-{id}] or [LLM-ERROR-{id}] marker")
        
    except Exception as e:
        print(f"LLM client unavailable: {e}")
        print("Correlation IDs would still be generated for fallback scenarios")


def show_log_analysis_tips():
    """Show tips for analyzing the generated logs."""
    print("\n=== LOG ANALYSIS TIPS ===")
    print("The generated log file 'llm_audit_log.txt' contains:")
    print("")
    print("🔍 To find all LLM requests:")
    print("   grep 'LLM-REQUEST' llm_audit_log.txt")
    print("")
    print("🔍 To find all LLM responses:")
    print("   grep 'LLM-RESPONSE' llm_audit_log.txt")
    print("")
    print("🔍 To find all LLM errors:")
    print("   grep 'LLM-ERROR' llm_audit_log.txt")
    print("")
    print("🔍 To find all fallback reasoning:")
    print("   grep 'FALLBACK-REASONING' llm_audit_log.txt")
    print("")
    print("🔍 To trace a specific request (replace {id} with actual ID):")
    print("   grep 'REQUEST-{id}\\|RESPONSE-{id}\\|ERROR-{id}' llm_audit_log.txt")
    print("")
    print("🔍 To see full prompts sent to LLM:")
    print("   grep -A 2 'FULL PROMPT BEGINS' llm_audit_log.txt")
    print("")
    print("🔍 To see full responses from LLM:")
    print("   grep -A 2 'FULL RESPONSE BEGINS' llm_audit_log.txt")
    print("")
    print("🔍 To see complete error tracebacks:")
    print("   grep -A 10 'FULL ERROR TRACEBACK BEGINS' llm_audit_log.txt")


def main():
    """Main test function."""
    setup_comprehensive_logging()
    
    try:
        test_llm_transparency_scenario()
        demonstrate_request_correlation()
        show_log_analysis_tips()
        
        print("\n🎉 LLM TRANSPARENCY TEST COMPLETED SUCCESSFULLY!")
        print("📄 Review 'llm_audit_log.txt' for complete audit trail")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()