"""Unit tests for ReasoningAgent parser functionality.

This module tests the critical parsing functions in ReasoningAgent that handle
function call arguments and LLM response parsing, ensuring they correctly
handle all Python data types including lists, dicts, booleans, and more.
"""

import pytest
from unittest.mock import Mock

from oniks.agents.reasoning_agent import ReasoningAgent
from oniks.core.state import State
from oniks.tools.base import Tool


class TestReasoningAgentFunctionCallParsing:
    """Test ReasoningAgent function call argument parsing."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a ReasoningAgent for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "Test tool for parsing"
        
        llm_client = Mock()
        llm_client.invoke = Mock(return_value="Mock response")
        
        return ReasoningAgent("test_agent", [tool], llm_client)
    
    def test_parse_function_call_string_arguments(self, reasoning_agent):
        """Test parsing function calls with string arguments."""
        test_cases = [
            ("test_tool(file_path='hello.txt')", {'file_path': 'hello.txt'}),
            ('test_tool(file_path="hello.txt")', {'file_path': 'hello.txt'}),
            ("test_tool(name='test file.txt')", {'name': 'test file.txt'}),
            ('test_tool(path="/path/to/file")', {'path': '/path/to/file'}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['next_tool'] == 'test_tool'
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_integer_arguments(self, reasoning_agent):
        """Test parsing function calls with integer arguments."""
        test_cases = [
            ("test_tool(count=42)", {'count': 42}),
            ("test_tool(port=8080)", {'port': 8080}),
            ("test_tool(negative=-123)", {'negative': -123}),
            ("test_tool(zero=0)", {'zero': 0}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_boolean_arguments(self, reasoning_agent):
        """Test parsing function calls with boolean arguments."""
        test_cases = [
            ("test_tool(enabled=True)", {'enabled': True}),
            ("test_tool(debug=False)", {'debug': False}),
            ("test_tool(active=True, verbose=False)", {'active': True, 'verbose': False}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_list_arguments(self, reasoning_agent):
        """Test parsing function calls with list arguments."""
        test_cases = [
            ("test_tool(files=['a.txt', 'b.txt', 'c.txt'])", 
             {'files': ['a.txt', 'b.txt', 'c.txt']}),
            ("test_tool(numbers=[1, 2, 3, 4, 5])", 
             {'numbers': [1, 2, 3, 4, 5]}),
            ("test_tool(flags=[True, False, True])", 
             {'flags': [True, False, True]}),
            ("test_tool(mixed=[1, 'two', True, None])", 
             {'mixed': [1, 'two', True, None]}),
            ("test_tool(empty_list=[])", 
             {'empty_list': []}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_dict_arguments(self, reasoning_agent):
        """Test parsing function calls with dictionary arguments."""
        test_cases = [
            ("test_tool(config={'host': 'localhost', 'port': 8080})", 
             {'config': {'host': 'localhost', 'port': 8080}}),
            ("test_tool(metadata={'version': '1.0', 'debug': True})", 
             {'metadata': {'version': '1.0', 'debug': True}}),
            ("test_tool(nested={'level1': {'level2': 'value'}})", 
             {'nested': {'level1': {'level2': 'value'}}}),
            ("test_tool(empty_dict={})", 
             {'empty_dict': {}}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_none_arguments(self, reasoning_agent):
        """Test parsing function calls with None arguments."""
        test_cases = [
            ("test_tool(value=None)", {'value': None}),
            ("test_tool(optional=None, required='test')", {'optional': None, 'required': 'test'}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_mixed_arguments(self, reasoning_agent):
        """Test parsing function calls with mixed argument types."""
        test_cases = [
            ("test_tool(name='test', count=5, enabled=True)", 
             {'name': 'test', 'count': 5, 'enabled': True}),
            ("test_tool(files=['a.txt'], config={'mode': 'strict'})", 
             {'files': ['a.txt'], 'config': {'mode': 'strict'}}),
            ("test_tool(data=[1, 2], meta={'id': 42}, active=False)", 
             {'data': [1, 2], 'meta': {'id': 42}, 'active': False}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_function_call_no_arguments(self, reasoning_agent):
        """Test parsing function calls with no arguments."""
        state = State()
        success = reasoning_agent._parse_function_call("test_tool()", state)
        
        assert success
        assert state.data['next_tool'] == 'test_tool'
        assert state.data['tool_args'] == {}
    
    def test_parse_function_call_invalid_format(self, reasoning_agent):
        """Test parsing function calls with invalid format."""
        invalid_calls = [
            "not_a_function_call",
            "test_tool[invalid_brackets]",
            "test_tool(unclosed_paren",
            "test_tool)missing_open_paren(",
            "",
            None,
        ]
        
        for invalid_call in invalid_calls:
            state = State()
            success = reasoning_agent._parse_function_call(invalid_call, state)
            
            assert not success, f"Should have failed to parse: {invalid_call}"


class TestReasoningAgentLLMResponseParsing:
    """Test ReasoningAgent LLM response argument parsing."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a ReasoningAgent for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "Test tool for parsing"
        
        llm_client = Mock()
        llm_client.invoke = Mock(return_value="Mock response")
        
        return ReasoningAgent("test_agent", [tool], llm_client)
    
    def test_parse_json_arguments(self, reasoning_agent):
        """Test parsing JSON format arguments from LLM responses."""
        test_cases = [
            ('{"file_path": "hello.txt"}', {'file_path': 'hello.txt'}),
            ('{"count": 42}', {'count': 42}),
            ('{"enabled": true}', {'enabled': True}),
            ('{"debug": false}', {'debug': False}),
            ('{"files": ["a.txt", "b.txt"]}', {'files': ['a.txt', 'b.txt']}),
            ('{"numbers": [1, 2, 3]}', {'numbers': [1, 2, 3]}),
            ('{"config": {"host": "localhost", "port": 8080}}', 
             {'config': {'host': 'localhost', 'port': 8080}}),
            ('{"empty_list": []}', {'empty_list': []}),
            ('{"empty_dict": {}}', {'empty_dict': {}}),
        ]
        
        for args_str, expected_result in test_cases:
            state = State()
            parsed = reasoning_agent._parse_arguments_multi_stage(args_str, state)
            
            assert parsed == expected_result, f"Failed to parse JSON: {args_str}"
    
    def test_parse_python_literal_arguments(self, reasoning_agent):
        """Test parsing Python literal format arguments."""
        test_cases = [
            ("['file1.txt', 'file2.txt']", ['file1.txt', 'file2.txt']),
            ("{'key': 'value', 'number': 42}", {'key': 'value', 'number': 42}),
            ("(True, False, True)", (True, False, True)),
            ("[1, 2, 3, 4, 5]", [1, 2, 3, 4, 5]),
            ("{'nested': {'level': 1}}", {'nested': {'level': 1}}),
        ]
        
        for args_str, expected_raw in test_cases:
            state = State()
            parsed = reasoning_agent._parse_arguments_multi_stage(args_str, state)
            
            # For non-dict results, they get normalized by the agent
            if isinstance(expected_raw, dict):
                expected = expected_raw
            else:
                expected = reasoning_agent._normalize_arguments(expected_raw, state)
            
            assert parsed == expected, f"Failed to parse Python literal: {args_str}"
    
    def test_parse_malformed_arguments(self, reasoning_agent):
        """Test parsing malformed arguments with fallback strategies."""
        test_cases = [
            ("file_path='hello.txt'", {'file_path': 'hello.txt'}),  # Missing braces
            ("{file_path: 'hello.txt'}", {'file_path': 'hello.txt'}),  # Missing quotes on key
            ("'single_file.txt'", {'file_path': 'single_file.txt'}),  # Single string value
        ]
        
        for args_str, expected_result in test_cases:
            state = State()
            parsed = reasoning_agent._parse_arguments_multi_stage(args_str, state)
            
            assert parsed == expected_result, f"Failed to parse malformed: {args_str}"
    
    def test_parse_empty_arguments(self, reasoning_agent):
        """Test parsing empty or whitespace arguments."""
        test_cases = [
            ("{}", {}),  # Empty JSON object should return empty dict
            ("", {'value': ''}),  # Empty string becomes value dict (fallback behavior)
            ("   ", {'value': '   '}),  # Whitespace string preserved as-is
            ("\n\t", {'value': '\n\t'}),  # Whitespace with newlines preserved as-is
        ]
        
        for args_str, expected in test_cases:
            state = State()
            parsed = reasoning_agent._parse_arguments_multi_stage(args_str, state)
            
            assert parsed == expected, f"Failed to parse empty case '{args_str}': expected {expected}, got {parsed}"


class TestReasoningAgentParserEdgeCases:
    """Test edge cases and error conditions in argument parsing."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a ReasoningAgent for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "Test tool for parsing"
        
        llm_client = Mock()
        llm_client.invoke = Mock(return_value="Mock response")
        
        return ReasoningAgent("test_agent", [tool], llm_client)
    
    def test_parse_deeply_nested_structures(self, reasoning_agent):
        """Test parsing deeply nested data structures."""
        deep_structure = {
            'level1': {
                'level2': {
                    'level3': {
                        'data': [1, 2, {'nested_list': [True, False]}]
                    }
                }
            }
        }
        
        function_call = f"test_tool(config={deep_structure})"
        state = State()
        success = reasoning_agent._parse_function_call(function_call, state)
        
        assert success
        assert state.data['tool_args'] == {'config': deep_structure}
    
    def test_parse_special_characters_in_strings(self, reasoning_agent):
        """Test parsing strings with special characters."""
        test_cases = [
            ("test_tool(path='/path/with/slashes')", {'path': '/path/with/slashes'}),
            ("test_tool(regex=r'\\d+\\.\\w+')", {'regex': r'\d+\.\w+'}),
            ("test_tool(unicode='Привет мир! 🚀')", {'unicode': 'Привет мир! 🚀'}),
            ("test_tool(json_str='{\"key\": \"value\"}')", {'json_str': '{"key": "value"}'}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parse_large_data_structures(self, reasoning_agent):
        """Test parsing large data structures."""
        large_list = list(range(100))
        large_dict = {f'key_{i}': f'value_{i}' for i in range(50)}
        
        test_cases = [
            (f"test_tool(numbers={large_list})", {'numbers': large_list}),
            (f"test_tool(mapping={large_dict})", {'mapping': large_dict}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse large structure: {function_call}"
            assert state.data['tool_args'] == expected_args
    
    def test_parser_error_resilience(self, reasoning_agent):
        """Test that parser handles errors gracefully without crashing."""
        problematic_inputs = [
            "test_tool(invalid=invalid_syntax)",  # Invalid Python syntax
            "test_tool(func=lambda x: x)",  # Non-literal expression
            "test_tool(import=__import__('os'))",  # Potentially dangerous expression
            "test_tool(eval=eval('1+1'))",  # Another dangerous expression
        ]
        
        for problematic_input in problematic_inputs:
            state = State()
            # Should not raise exceptions, but may return False or fallback values
            try:
                success = reasoning_agent._parse_function_call(problematic_input, state)
                # Either successfully parsed with fallback or failed gracefully
                assert success in [True, False]
            except Exception as e:
                pytest.fail(f"Parser should handle errors gracefully, but raised: {e}")
    
    def test_normalize_arguments_various_types(self, reasoning_agent):
        """Test argument normalization for various input types."""
        state = State()
        
        test_cases = [
            # Dict should remain unchanged
            ({'key': 'value'}, {'key': 'value'}),
            
            # FIXED: Lists should be preserved as-is (no longer converted to dicts)
            (['data.json'], ['data.json']),
            ([1, 2, 3], [1, 2, 3]),
            (['file1.txt', 'file2.txt'], ['file1.txt', 'file2.txt']),
            
            # FIXED: Primitive types should be preserved as-is
            (42, 42),
            (True, True),
            (False, False),
            (None, None),
            (3.14, 3.14),
            
            # Single item tuples should still become dict with guessed key (legacy behavior)
            (('file.txt',), {'file_path': 'file.txt'}),
            (('simple_value',), {'value': 'simple_value'}),
            
            # Two item tuple should become key-value dict
            (('key', 'value'), {'key': 'value'}),
            
            # Multi-element tuples should be preserved as-is (FIXED)
            ((1, 2, 3, 4), (1, 2, 3, 4)),
            
            # String should become dict with guessed key
            ('config.yaml', {'file_path': 'config.yaml'}),
            ('simple_string', {'value': 'simple_string'}),
        ]
        
        for input_value, expected_output in test_cases:
            result = reasoning_agent._normalize_arguments(input_value, state)
            assert result == expected_output, f"Failed to normalize {input_value}"


class TestReasoningAgentParserCriticalBugFixes:
    """Test cases specifically for the critical bug fixes in list/dict/bool parsing."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a ReasoningAgent for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "Test tool for parsing"
        
        llm_client = Mock()
        llm_client.invoke = Mock(return_value="Mock response")
        
        return ReasoningAgent("test_agent", [tool], llm_client)
    
    def test_list_preservation_in_llm_response(self, reasoning_agent):
        """Test that lists from LLM responses are preserved as lists, not converted to indexed dicts."""
        test_cases = [
            # Simple string list
            ('Tool: test_tool\nArguments: ["file1.txt", "file2.txt", "file3.txt"]', 
             ['file1.txt', 'file2.txt', 'file3.txt']),
            
            # Mixed type list  
            ('Tool: test_tool\nArguments: [1, "text", true, null]',
             [1, 'text', True, None]),
            
            # Nested list
            ('Tool: test_tool\nArguments: [["a", "b"], ["c", "d"]]',
             [['a', 'b'], ['c', 'd']]),
            
            # Empty list
            ('Tool: test_tool\nArguments: []',
             []),
        ]
        
        for llm_response, expected_list in test_cases:
            state = State()
            reasoning_agent._parse_llm_response(llm_response, state)
            
            assert state.data['tool_args'] == expected_list, f"Failed to preserve list: {llm_response}"
            assert isinstance(state.data['tool_args'], list), "Arguments should be a list"
    
    def test_boolean_preservation_in_llm_response(self, reasoning_agent):
        """Test that boolean values are preserved as booleans, not converted to dicts."""
        test_cases = [
            ('Tool: test_tool\nArguments: true', True),
            ('Tool: test_tool\nArguments: false', False),
        ]
        
        for llm_response, expected_bool in test_cases:
            state = State()
            reasoning_agent._parse_llm_response(llm_response, state)
            
            assert state.data['tool_args'] == expected_bool, f"Failed to preserve boolean: {llm_response}"
            assert isinstance(state.data['tool_args'], bool), "Arguments should be a boolean"
    
    def test_dict_preservation_in_llm_response(self, reasoning_agent):
        """Test that dictionaries are preserved correctly in LLM responses."""
        test_cases = [
            # Simple dict
            ('Tool: test_tool\nArguments: {"key": "value"}',
             {'key': 'value'}),
            
            # Dict with mixed types
            ('Tool: test_tool\nArguments: {"string": "text", "number": 42, "bool": true, "null": null}',
             {'string': 'text', 'number': 42, 'bool': True, 'null': None}),
            
            # Nested dict with list
            ('Tool: test_tool\nArguments: {"files": ["a.txt", "b.txt"], "config": {"debug": true}}',
             {'files': ['a.txt', 'b.txt'], 'config': {'debug': True}}),
        ]
        
        for llm_response, expected_dict in test_cases:
            state = State()
            reasoning_agent._parse_llm_response(llm_response, state)
            
            assert state.data['tool_args'] == expected_dict, f"Failed to preserve dict: {llm_response}"
            assert isinstance(state.data['tool_args'], dict), "Arguments should be a dict"
    
    def test_primitive_types_preservation(self, reasoning_agent):
        """Test that primitive types (int, float, None) are preserved correctly."""
        test_cases = [
            ('Tool: test_tool\nArguments: 42', 42),
            ('Tool: test_tool\nArguments: 3.14', 3.14),
            ('Tool: test_tool\nArguments: null', None),
            ('Tool: test_tool\nArguments: -123', -123),
        ]
        
        for llm_response, expected_value in test_cases:
            state = State()
            reasoning_agent._parse_llm_response(llm_response, state)
            
            assert state.data['tool_args'] == expected_value, f"Failed to preserve value: {llm_response}"
            assert type(state.data['tool_args']) == type(expected_value), f"Type mismatch for: {llm_response}"
    
    def test_function_call_list_preservation(self, reasoning_agent):
        """Test that lists in function calls are preserved in tool_args."""
        test_cases = [
            ("test_tool(files=['input.txt', 'output.txt'])",
             {'files': ['input.txt', 'output.txt']}),
            
            ("test_tool(numbers=[1, 2, 3, 4, 5])",
             {'numbers': [1, 2, 3, 4, 5]}),
            
            ("test_tool(mixed=[1, 'text', True, None])",
             {'mixed': [1, 'text', True, None]}),
        ]
        
        for function_call, expected_args in test_cases:
            state = State()
            success = reasoning_agent._parse_function_call(function_call, state)
            
            assert success, f"Failed to parse function call: {function_call}"
            assert state.data['tool_args'] == expected_args, f"Incorrect args for: {function_call}"
            
            # Verify individual arguments are set correctly for dict args
            for key, value in expected_args.items():
                assert state.data[key] == value, f"Individual arg {key} not set correctly"
    
    def test_ast_literal_eval_safety(self, reasoning_agent):
        """Test that ast.literal_eval is used safely and dangerous code is rejected."""
        dangerous_inputs = [
            'Tool: test_tool\nArguments: __import__("os").system("rm -rf /")',
            'Tool: test_tool\nArguments: eval("malicious_code")',
            'Tool: test_tool\nArguments: exec("import os; os.system(\'ls\')")',
        ]
        
        for dangerous_input in dangerous_inputs:
            state = State()
            # Should not crash and should fall back to safe parsing
            reasoning_agent._parse_llm_response(dangerous_input, state)
            
            # Should either fail to parse or fallback to safe string parsing
            tool_args = state.data.get('tool_args')
            # Should not contain any dangerous executable code objects
            assert not callable(tool_args), "Parser should not return callable objects"
    
    def test_complex_nested_data_structures(self, reasoning_agent):
        """Test parsing of complex nested data structures with mixed types."""
        complex_structure = {
            'config': {
                'servers': ['server1.com', 'server2.com'],
                'ports': [8080, 8081, 8082],
                'settings': {
                    'debug': True,
                    'timeout': 30.5,
                    'retries': None,
                    'features': {
                        'logging': True,
                        'monitoring': False,
                        'alerts': ['email', 'sms']
                    }
                }
            },
            'data': [
                {'id': 1, 'values': [1.1, 2.2, 3.3]},
                {'id': 2, 'values': [4.4, 5.5, 6.6]},
            ]
        }
        
        function_call = f"test_tool(config={complex_structure})"
        state = State()
        success = reasoning_agent._parse_function_call(function_call, state)
        
        assert success, "Failed to parse complex nested structure"
        assert state.data['tool_args'] == {'config': complex_structure}
        assert state.data['config'] == complex_structure


class TestReasoningAgentParserIntegration:
    """Integration tests for the complete parsing workflow."""
    
    @pytest.fixture
    def reasoning_agent(self):
        """Create a ReasoningAgent for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "Test tool for parsing"
        
        llm_client = Mock()
        llm_client.invoke = Mock(return_value="Mock response")
        
        return ReasoningAgent("test_agent", [tool], llm_client)
    
    def test_end_to_end_function_call_parsing(self, reasoning_agent):
        """Test complete function call parsing workflow."""
        # Create a plan with function call format
        state = State()
        state.data['plan'] = [
            "test_tool(files=['input.txt', 'output.txt'], config={'mode': 'strict', 'debug': True})"
        ]
        
        # Execute the reasoning agent
        result_state = reasoning_agent.execute(state)
        
        # Verify the parsing results
        assert result_state.data['next_tool'] == 'test_tool'
        expected_args = {
            'files': ['input.txt', 'output.txt'],
            'config': {'mode': 'strict', 'debug': True}
        }
        assert result_state.data['tool_args'] == expected_args
        
        # Verify individual arguments are set in state for tool compatibility
        assert result_state.data['files'] == ['input.txt', 'output.txt']
        assert result_state.data['config'] == {'mode': 'strict', 'debug': True}
    
    def test_fallback_to_llm_on_parse_failure(self, reasoning_agent):
        """Test that system falls back to LLM when direct parsing fails."""
        # Create a plan with descriptive format (not function call)
        state = State()
        state.data['plan'] = ["Create a file with some content"]
        
        # Mock LLM to return a parseable response
        llm_response = """
        Tool: write_file
        Arguments: {"file_path": "test.txt", "content": "Hello World"}
        """
        reasoning_agent.llm_client.invoke.return_value = llm_response
        
        # Execute the reasoning agent
        result_state = reasoning_agent.execute(state)
        
        # Verify it used LLM and parsed the response
        assert result_state.data['next_tool'] == 'write_file'
        assert result_state.data['tool_args'] == {
            'file_path': 'test.txt',
            'content': 'Hello World'
        }
        
        # Verify LLM was actually called
        reasoning_agent.llm_client.invoke.assert_called_once()
    
    def test_parser_preserves_data_types(self, reasoning_agent):
        """Test that parser preserves correct Python data types."""
        function_call = "test_tool(string='text', integer=42, boolean=True, " \
                       "float_num=3.14, none_val=None, list_val=[1, 2], " \
                       "dict_val={'key': 'value'})"
        
        state = State()
        success = reasoning_agent._parse_function_call(function_call, state)
        
        assert success
        args = state.data['tool_args']
        
        # Verify types are preserved correctly
        assert isinstance(args['string'], str) and args['string'] == 'text'
        assert isinstance(args['integer'], int) and args['integer'] == 42
        assert isinstance(args['boolean'], bool) and args['boolean'] is True
        assert isinstance(args['float_num'], float) and args['float_num'] == 3.14
        assert args['none_val'] is None
        assert isinstance(args['list_val'], list) and args['list_val'] == [1, 2]
        assert isinstance(args['dict_val'], dict) and args['dict_val'] == {'key': 'value'}