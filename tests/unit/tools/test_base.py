"""Unit tests for the base Tool class."""

import pytest
from oniks.tools.base import Tool


class TestToolAbstract:
    """Test the abstract Tool class."""
    
    def test_tool_is_abstract(self):
        """Test that Tool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Tool()
    
    def test_tool_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        class IncompleteTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "incomplete"
                self.description = "incomplete tool"
            # Missing execute method
        
        with pytest.raises(TypeError):
            IncompleteTool()
    
    def test_tool_concrete_implementation(self):
        """Test that concrete implementation can be instantiated."""
        class ConcreteTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "concrete_tool"
                self.description = "A concrete tool implementation"
            
            def execute(self, **kwargs):
                return "executed"
        
        tool = ConcreteTool()
        
        assert tool.name == "concrete_tool"
        assert tool.description == "A concrete tool implementation"
        assert tool.execute() == "executed"
    
    def test_tool_initialization_attributes(self):
        """Test Tool initialization sets correct attributes."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "Test description"
            
            def execute(self, **kwargs):
                return "test result"
        
        tool = TestTool()
        
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert tool.name == "test_tool"
        assert tool.description == "Test description"
    
    def test_tool_string_representation(self):
        """Test Tool string representation."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "Test description"
            
            def execute(self, **kwargs):
                return "test result"
        
        tool = TestTool()
        
        assert str(tool) == "TestTool(name='test_tool')"
        assert repr(tool) == "TestTool(name='test_tool')"
    
    def test_tool_execute_signature(self):
        """Test that execute method has correct signature."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "Test description"
            
            def execute(self, **kwargs):
                return f"executed with {kwargs}"
        
        tool = TestTool()
        
        # Test with no arguments
        result1 = tool.execute()
        assert result1 == "executed with {}"
        
        # Test with arguments
        result2 = tool.execute(arg1="value1", arg2="value2")
        assert "value1" in result2
        assert "value2" in result2
    
    def test_tool_execute_return_type(self):
        """Test that execute method returns string."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "test_tool"
                self.description = "Test description"
            
            def execute(self, **kwargs):
                return "string result"
        
        tool = TestTool()
        result = tool.execute()
        
        assert isinstance(result, str)
        assert result == "string result"


class TestToolImplementationPatterns:
    """Test common Tool implementation patterns."""
    
    def test_tool_with_required_arguments(self):
        """Test tool implementation with required arguments."""
        class RequiredArgsTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "required_args_tool"
                self.description = "Tool requiring specific arguments"
            
            def execute(self, **kwargs):
                if 'required_arg' not in kwargs:
                    return "Error: missing required_arg"
                return f"Success: {kwargs['required_arg']}"
        
        tool = RequiredArgsTool()
        
        # Test without required argument
        result1 = tool.execute()
        assert "Error: missing required_arg" in result1
        
        # Test with required argument
        result2 = tool.execute(required_arg="test_value")
        assert "Success: test_value" in result2
    
    def test_tool_with_argument_validation(self):
        """Test tool implementation with argument validation."""
        class ValidatingTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "validating_tool"
                self.description = "Tool with argument validation"
            
            def execute(self, **kwargs):
                number = kwargs.get('number')
                if number is None:
                    return "Error: number is required"
                
                if not isinstance(number, (int, float)):
                    return f"Error: number must be int or float, got {type(number).__name__}"
                
                if number < 0:
                    return "Error: number must be non-negative"
                
                return f"Valid number: {number}"
        
        tool = ValidatingTool()
        
        # Test missing argument
        assert "Error: number is required" in tool.execute()
        
        # Test invalid type
        assert "Error: number must be int or float" in tool.execute(number="not_a_number")
        
        # Test invalid value
        assert "Error: number must be non-negative" in tool.execute(number=-5)
        
        # Test valid arguments
        assert "Valid number: 42" in tool.execute(number=42)
        assert "Valid number: 3.14" in tool.execute(number=3.14)
    
    def test_tool_with_complex_processing(self):
        """Test tool implementation with complex processing logic."""
        class ProcessingTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "processing_tool"
                self.description = "Tool with complex processing"
            
            def execute(self, **kwargs):
                data = kwargs.get('data', [])
                operation = kwargs.get('operation', 'sum')
                
                if not isinstance(data, list):
                    return "Error: data must be a list"
                
                if not data:
                    return "Result: empty data"
                
                try:
                    if operation == 'sum':
                        result = sum(data)
                    elif operation == 'product':
                        result = 1
                        for item in data:
                            result *= item
                    elif operation == 'average':
                        result = sum(data) / len(data)
                    else:
                        return f"Error: unknown operation '{operation}'"
                    
                    return f"Result: {result}"
                    
                except (TypeError, ValueError) as e:
                    return f"Error: processing failed - {str(e)}"
        
        tool = ProcessingTool()
        
        # Test sum operation
        result1 = tool.execute(data=[1, 2, 3, 4, 5], operation='sum')
        assert "Result: 15" in result1
        
        # Test product operation
        result2 = tool.execute(data=[2, 3, 4], operation='product')
        assert "Result: 24" in result2
        
        # Test average operation
        result3 = tool.execute(data=[10, 20, 30], operation='average')
        assert "Result: 20.0" in result3
        
        # Test unknown operation
        result4 = tool.execute(data=[1, 2, 3], operation='unknown')
        assert "Error: unknown operation 'unknown'" in result4
        
        # Test invalid data type
        result5 = tool.execute(data="not_a_list")
        assert "Error: data must be a list" in result5
        
        # Test empty data
        result6 = tool.execute(data=[])
        assert "Result: empty data" in result6
    
    def test_tool_with_exception_handling(self):
        """Test tool implementation with comprehensive exception handling."""
        class ExceptionHandlingTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "exception_tool"
                self.description = "Tool with exception handling"
            
            def execute(self, **kwargs):
                action = kwargs.get('action', 'safe')
                
                try:
                    if action == 'safe':
                        return "Success: safe operation"
                    elif action == 'divide_by_zero':
                        result = 10 / 0
                        return f"Result: {result}"
                    elif action == 'key_error':
                        data = {'key': 'value'}
                        return data['missing_key']
                    elif action == 'type_error':
                        return "string" + 42
                    elif action == 'custom_error':
                        raise RuntimeError("Custom error message")
                    else:
                        return f"Unknown action: {action}"
                        
                except ZeroDivisionError:
                    return "Error: Division by zero"
                except KeyError as e:
                    return f"Error: Key not found - {e}"
                except TypeError as e:
                    return f"Error: Type error - {e}"
                except Exception as e:
                    return f"Error: Unexpected error - {e}"
        
        tool = ExceptionHandlingTool()
        
        # Test safe operation
        assert "Success: safe operation" in tool.execute(action='safe')
        
        # Test specific exception handling
        assert "Error: Division by zero" in tool.execute(action='divide_by_zero')
        assert "Error: Key not found" in tool.execute(action='key_error')
        assert "Error: Type error" in tool.execute(action='type_error')
        assert "Error: Unexpected error - Custom error message" in tool.execute(action='custom_error')
    
    def test_tool_with_default_parameters(self):
        """Test tool implementation with default parameters."""
        class DefaultParamsTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "default_params_tool"
                self.description = "Tool with default parameters"
            
            def execute(self, **kwargs):
                message = kwargs.get('message', 'Hello, World!')
                repeat = kwargs.get('repeat', 1)
                separator = kwargs.get('separator', ' ')
                
                if not isinstance(repeat, int) or repeat < 1:
                    return "Error: repeat must be a positive integer"
                
                result = separator.join([message] * repeat)
                return f"Result: {result}"
        
        tool = DefaultParamsTool()
        
        # Test with defaults
        result1 = tool.execute()
        assert "Result: Hello, World!" in result1
        
        # Test with custom message
        result2 = tool.execute(message="Custom message")
        assert "Result: Custom message" in result2
        
        # Test with repeat
        result3 = tool.execute(message="Hi", repeat=3)
        assert "Result: Hi Hi Hi" in result3
        
        # Test with custom separator
        result4 = tool.execute(message="Word", repeat=3, separator="-")
        assert "Result: Word-Word-Word" in result4
        
        # Test invalid repeat
        result5 = tool.execute(repeat=-1)
        assert "Error: repeat must be a positive integer" in result5


class TestToolEdgeCases:
    """Test Tool edge cases and boundary conditions."""
    
    def test_tool_with_empty_name_and_description(self):
        """Test tool with empty name and description."""
        class EmptyTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = ""
                self.description = ""
            
            def execute(self, **kwargs):
                return "executed"
        
        tool = EmptyTool()
        
        assert tool.name == ""
        assert tool.description == ""
        assert str(tool) == "EmptyTool(name='')"
    
    def test_tool_with_unicode_content(self):
        """Test tool with unicode characters in name, description, and output."""
        class UnicodeTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "unicode_tool_🔧"
                self.description = "Инструмент с unicode символами 🌍"
            
            def execute(self, **kwargs):
                message = kwargs.get('message', 'Привет мир!')
                return f"Unicode result: {message} 🎉"
        
        tool = UnicodeTool()
        
        assert "🔧" in tool.name
        assert "unicode символами" in tool.description
        
        result = tool.execute()
        assert "Привет мир!" in result
        assert "🎉" in result
        
        custom_result = tool.execute(message="你好世界")
        assert "你好世界" in custom_result
    
    def test_tool_with_large_arguments(self):
        """Test tool with large argument values."""
        class LargeArgsTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "large_args_tool"
                self.description = "Tool handling large arguments"
            
            def execute(self, **kwargs):
                data = kwargs.get('data', [])
                text = kwargs.get('text', '')
                
                data_size = len(data) if isinstance(data, (list, str, dict)) else 0
                text_size = len(text) if isinstance(text, str) else 0
                
                return f"Processed data size: {data_size}, text size: {text_size}"
        
        tool = LargeArgsTool()
        
        # Test with large list
        large_list = list(range(10000))
        result1 = tool.execute(data=large_list)
        assert "data size: 10000" in result1
        
        # Test with large string
        large_string = "x" * 100000
        result2 = tool.execute(text=large_string)
        assert "text size: 100000" in result2
        
        # Test with large dictionary
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        result3 = tool.execute(data=large_dict)
        assert "data size: 1000" in result3
    
    def test_tool_with_none_arguments(self):
        """Test tool behavior with None arguments."""
        class NoneArgsTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = "none_args_tool"
                self.description = "Tool handling None arguments"
            
            def execute(self, **kwargs):
                arg1 = kwargs.get('arg1')
                arg2 = kwargs.get('arg2', 'default')
                
                results = []
                results.append(f"arg1 is None: {arg1 is None}")
                results.append(f"arg2 value: {arg2}")
                
                # Test explicit None
                if 'explicit_none' in kwargs:
                    results.append(f"explicit_none is None: {kwargs['explicit_none'] is None}")
                
                return "Results: " + ", ".join(results)
        
        tool = NoneArgsTool()
        
        # Test with missing arguments
        result1 = tool.execute()
        assert "arg1 is None: True" in result1
        assert "arg2 value: default" in result1
        
        # Test with explicit None
        result2 = tool.execute(arg1=None, explicit_none=None)
        assert "arg1 is None: True" in result2
        assert "explicit_none is None: True" in result2
        
        # Test with mixed None and non-None
        result3 = tool.execute(arg1="value", arg2=None)
        assert "arg1 is None: False" in result3
        assert "arg2 value: None" in result3