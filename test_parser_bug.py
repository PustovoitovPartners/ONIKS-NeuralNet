#!/usr/bin/env python3
"""Test script to identify parser bugs in ReasoningAgent."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.core.state import State
from oniks.tools.base import Tool
from unittest.mock import Mock

def create_test_agent():
    """Create a test agent with mock tools."""
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool for parsing"
    
    # Mock LLM client
    llm_client = Mock()
    llm_client.invoke = Mock(return_value="Mock response")
    
    return ReasoningAgent("test_agent", [tool], llm_client)

def test_function_call_parsing():
    """Test various function call argument parsing scenarios."""
    agent = create_test_agent()
    
    test_cases = [
        # String arguments
        ("test_tool(file_path='hello.txt')", "str", {'file_path': 'hello.txt'}),
        ('test_tool(file_path="hello.txt")', "str", {'file_path': 'hello.txt'}),
        
        # Integer arguments
        ("test_tool(count=42)", "int", {'count': 42}),
        ("test_tool(port=8080)", "int", {'port': 8080}),
        
        # Boolean arguments
        ("test_tool(enabled=True)", "bool", {'enabled': True}),
        ("test_tool(debug=False)", "bool", {'debug': False}),
        
        # List arguments
        ("test_tool(files=['a.txt', 'b.txt', 'c.txt'])", "list[str]", {'files': ['a.txt', 'b.txt', 'c.txt']}),
        ("test_tool(numbers=[1, 2, 3, 4, 5])", "list[int]", {'numbers': [1, 2, 3, 4, 5]}),
        ("test_tool(flags=[True, False, True])", "list[bool]", {'flags': [True, False, True]}),
        
        # Dictionary arguments
        ("test_tool(config={'host': 'localhost', 'port': 8080})", "dict", {'config': {'host': 'localhost', 'port': 8080}}),
        ("test_tool(metadata={'version': '1.0', 'debug': True})", "dict", {'metadata': {'version': '1.0', 'debug': True}}),
        
        # Mixed arguments
        ("test_tool(name='test', count=5, enabled=True)", "mixed", {'name': 'test', 'count': 5, 'enabled': True}),
        ("test_tool(files=['a.txt'], config={'mode': 'strict'})", "mixed", {'files': ['a.txt'], 'config': {'mode': 'strict'}}),
        
        # Edge cases
        ("test_tool(empty_list=[])", "empty_list", {'empty_list': []}),
        ("test_tool(empty_dict={})", "empty_dict", {'empty_dict': {}}),
        ("test_tool(none_value=None)", "none", {'none_value': None}),
    ]
    
    print("Testing function call argument parsing:")
    print("=" * 60)
    
    failed_tests = []
    passed_tests = []
    
    for function_call, test_type, expected_args in test_cases:
        print(f"\nTesting {test_type}: {function_call}")
        
        state = State()
        success = agent._parse_function_call(function_call, state)
        
        if success:
            actual_args = state.data.get('tool_args', {})
            if actual_args == expected_args:
                print(f"  ✓ PASS: {actual_args}")
                passed_tests.append((test_type, function_call))
            else:
                print(f"  ✗ FAIL: Expected {expected_args}, got {actual_args}")
                failed_tests.append((test_type, function_call, expected_args, actual_args))
        else:
            print(f"  ✗ FAIL: Parsing failed completely")
            failed_tests.append((test_type, function_call, expected_args, "PARSING_FAILED"))
    
    print("\n" + "=" * 60)
    print(f"Results: {len(passed_tests)} passed, {len(failed_tests)} failed")
    
    if failed_tests:
        print("\nFailed tests:")
        for test_type, function_call, expected, actual in failed_tests:
            print(f"  - {test_type}: {function_call}")
            print(f"    Expected: {expected}")
            print(f"    Actual: {actual}")
    
    return len(failed_tests) == 0

def test_llm_response_parsing():
    """Test LLM response argument parsing scenarios."""
    agent = create_test_agent()
    
    test_cases = [
        # String arguments
        ('Arguments: {"file_path": "hello.txt"}', "str", {'file_path': 'hello.txt'}),
        
        # Integer arguments
        ('Arguments: {"count": 42}', "int", {'count': 42}),
        
        # Boolean arguments
        ('Arguments: {"enabled": true}', "bool", {'enabled': True}),
        ('Arguments: {"debug": false}', "bool", {'debug': False}),
        
        # List arguments
        ('Arguments: {"files": ["a.txt", "b.txt"]}', "list[str]", {'files': ['a.txt', 'b.txt']}),
        ('Arguments: {"numbers": [1, 2, 3]}', "list[int]", {'numbers': [1, 2, 3]}),
        
        # Dictionary arguments
        ('Arguments: {"config": {"host": "localhost", "port": 8080}}', "dict", {'config': {'host': 'localhost', 'port': 8080}}),
        
        # Python literal format (using ast.literal_eval)
        ("Arguments: ['file1.txt', 'file2.txt']", "list[str]", ['file1.txt', 'file2.txt']),
        ("Arguments: {'key': 'value', 'number': 42}", "dict", {'key': 'value', 'number': 42}),
        ("Arguments: (True, False, True)", "tuple[bool]", (True, False, True)),
    ]
    
    print("\n\nTesting LLM response argument parsing:")
    print("=" * 60)
    
    failed_tests = []
    passed_tests = []
    
    for args_text, test_type, expected_result in test_cases:
        print(f"\nTesting {test_type}: {args_text}")
        
        state = State()
        parsed = agent._parse_arguments_multi_stage(args_text.replace("Arguments: ", ""), state)
        
        # For non-dict results, normalize them
        if isinstance(expected_result, dict):
            expected = expected_result
        else:
            expected = agent._normalize_arguments(expected_result, state)
        
        if parsed == expected:
            print(f"  ✓ PASS: {parsed}")
            passed_tests.append((test_type, args_text))
        else:
            print(f"  ✗ FAIL: Expected {expected}, got {parsed}")
            failed_tests.append((test_type, args_text, expected, parsed))
    
    print("\n" + "=" * 60)
    print(f"Results: {len(passed_tests)} passed, {len(failed_tests)} failed")
    
    if failed_tests:
        print("\nFailed tests:")
        for test_type, args_text, expected, actual in failed_tests:
            print(f"  - {test_type}: {args_text}")
            print(f"    Expected: {expected}")
            print(f"    Actual: {actual}")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    print("ONIKS NeuralNet - ExecutorAgent Parser Bug Analysis")
    print("=" * 60)
    
    success1 = test_function_call_parsing()
    success2 = test_llm_response_parsing()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("All tests passed! Parser is working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Parser needs to be fixed.")
        sys.exit(1)