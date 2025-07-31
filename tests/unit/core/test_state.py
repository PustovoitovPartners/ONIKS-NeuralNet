"""Unit tests for the State class."""

import pytest
from oniks.core.state import State


class TestStateInitialization:
    """Test State class initialization and basic properties."""
    
    def test_state_default_initialization(self):
        """Test that State initializes with proper default values."""
        state = State()
        
        assert isinstance(state.data, dict)
        assert len(state.data) == 0
        assert isinstance(state.message_history, list)
        assert len(state.message_history) == 0
        assert isinstance(state.tool_outputs, dict)
        assert len(state.tool_outputs) == 0
    
    def test_state_initialization_with_data(self):
        """Test State initialization with custom data."""
        initial_data = {"key1": "value1", "key2": 42}
        initial_messages = ["message1", "message2"]
        initial_outputs = {"tool1": "output1"}
        
        state = State(
            data=initial_data,
            message_history=initial_messages,
            tool_outputs=initial_outputs
        )
        
        assert state.data == initial_data
        assert state.message_history == initial_messages
        assert state.tool_outputs == initial_outputs


class TestStateMessageHistory:
    """Test State message history functionality."""
    
    def test_add_message_single(self):
        """Test adding a single message to history."""
        state = State()
        message = "Test message"
        
        state.add_message(message)
        
        assert len(state.message_history) == 1
        assert state.message_history[0] == message
    
    def test_add_message_multiple(self):
        """Test adding multiple messages to history."""
        state = State()
        messages = ["Message 1", "Message 2", "Message 3"]
        
        for message in messages:
            state.add_message(message)
        
        assert len(state.message_history) == 3
        assert state.message_history == messages
    
    def test_add_message_empty_string(self):
        """Test adding empty string message."""
        state = State()
        
        state.add_message("")
        
        assert len(state.message_history) == 1
        assert state.message_history[0] == ""
    
    def test_add_message_with_special_characters(self):
        """Test adding message with special characters."""
        state = State()
        message = "Test message with special chars: !@#$%^&*()"
        
        state.add_message(message)
        
        assert state.message_history[0] == message
    
    def test_clear_history(self):
        """Test clearing message history."""
        state = State()
        state.add_message("Test message 1")
        state.add_message("Test message 2")
        
        assert len(state.message_history) == 2
        
        state.clear_history()
        
        assert len(state.message_history) == 0
        assert state.message_history == []
    
    def test_clear_history_when_empty(self):
        """Test clearing empty message history."""
        state = State()
        
        state.clear_history()
        
        assert len(state.message_history) == 0


class TestStateDataManagement:
    """Test State data dictionary management."""
    
    def test_get_data_existing_key(self):
        """Test retrieving existing data key."""
        state = State()
        state.data["test_key"] = "test_value"
        
        result = state.get_data("test_key")
        
        assert result == "test_value"
    
    def test_get_data_missing_key_no_default(self):
        """Test retrieving missing key without default."""
        state = State()
        
        result = state.get_data("missing_key")
        
        assert result is None
    
    def test_get_data_missing_key_with_default(self):
        """Test retrieving missing key with default value."""
        state = State()
        default_value = "default"
        
        result = state.get_data("missing_key", default_value)
        
        assert result == default_value
    
    def test_get_data_various_types(self):
        """Test retrieving data of various types."""
        state = State()
        test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None
        }
        
        for key, value in test_data.items():
            state.data[key] = value
        
        for key, expected_value in test_data.items():
            assert state.get_data(key) == expected_value
    
    def test_set_data_single_value(self):
        """Test setting a single data value."""
        state = State()
        key = "test_key"
        value = "test_value"
        
        state.set_data(key, value)
        
        assert state.data[key] == value
        assert len(state.data) == 1
    
    def test_set_data_multiple_values(self):
        """Test setting multiple data values."""
        state = State()
        test_data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3]
        }
        
        for key, value in test_data.items():
            state.set_data(key, value)
        
        assert len(state.data) == 3
        for key, expected_value in test_data.items():
            assert state.data[key] == expected_value
    
    def test_set_data_overwrite_existing(self):
        """Test overwriting existing data value."""
        state = State()
        key = "test_key"
        initial_value = "initial"
        new_value = "updated"
        
        state.set_data(key, initial_value)
        assert state.data[key] == initial_value
        
        state.set_data(key, new_value)
        assert state.data[key] == new_value
        assert len(state.data) == 1
    
    def test_clear_data(self):
        """Test clearing all data."""
        state = State()
        state.data["key1"] = "value1"
        state.data["key2"] = "value2"
        
        assert len(state.data) == 2
        
        state.clear_data()
        
        assert len(state.data) == 0
        assert state.data == {}
    
    def test_clear_data_when_empty(self):
        """Test clearing empty data dictionary."""
        state = State()
        
        state.clear_data()
        
        assert len(state.data) == 0


class TestStateToolOutputs:
    """Test State tool outputs functionality."""
    
    def test_tool_outputs_initialization(self):
        """Test tool outputs dictionary initialization."""
        state = State()
        
        assert isinstance(state.tool_outputs, dict)
        assert len(state.tool_outputs) == 0
    
    def test_tool_outputs_assignment(self):
        """Test assigning tool outputs."""
        state = State()
        tool_name = "test_tool"
        output = "tool output"
        
        state.tool_outputs[tool_name] = output
        
        assert state.tool_outputs[tool_name] == output
        assert len(state.tool_outputs) == 1
    
    def test_tool_outputs_multiple_tools(self):
        """Test multiple tool outputs."""
        state = State()
        outputs = {
            "tool1": "output1",
            "tool2": {"result": "success"},
            "tool3": ["item1", "item2"]
        }
        
        for tool, output in outputs.items():
            state.tool_outputs[tool] = output
        
        assert len(state.tool_outputs) == 3
        for tool, expected_output in outputs.items():
            assert state.tool_outputs[tool] == expected_output


class TestStateModelOperations:
    """Test State model operations (Pydantic features)."""
    
    def test_state_model_copy_shallow(self):
        """Test shallow model copy behavior."""
        original_state = State()
        original_state.data["key"] = "value"
        original_state.add_message("test message")
        original_state.tool_outputs["tool"] = "output"
        
        copied_state = original_state.model_copy()
        
        # Verify copies are equal but different objects
        assert copied_state.data == original_state.data
        assert copied_state.message_history == original_state.message_history
        assert copied_state.tool_outputs == original_state.tool_outputs
        assert copied_state is not original_state
        
        # Verify shallow copy behavior (mutable fields are shared)
        copied_state.data["new_key"] = "new_value"
        assert "new_key" in original_state.data  # Shallow copy shares mutable objects
        
        # Verify that replacing the entire dict creates independence
        copied_state.data = {"different": "dict"}
        assert original_state.data["key"] == "value"  # Original unchanged
    
    def test_state_model_copy_deep(self):
        """Test deep model copy."""
        original_state = State()
        original_state.data["nested"] = {"key": "value"}
        original_state.add_message("test message")
        original_state.tool_outputs["tool"] = ["item1", "item2"]
        
        copied_state = original_state.model_copy(deep=True)
        
        # Verify deep copy behavior
        copied_state.data["nested"]["new_key"] = "new_value"
        assert "new_key" not in original_state.data["nested"]
        
        copied_state.tool_outputs["tool"].append("item3")
        assert len(original_state.tool_outputs["tool"]) == 2
    
    def test_state_serialization_json(self):
        """Test State JSON serialization."""
        state = State()
        state.data["test"] = "value"
        state.add_message("test message")
        state.tool_outputs["tool"] = "output"
        
        json_str = state.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "test" in json_str
        assert "value" in json_str
        assert "test message" in json_str
        assert "tool" in json_str
        assert "output" in json_str
    
    def test_state_deserialization_from_dict(self):
        """Test State creation from dictionary."""
        data = {
            "data": {"key": "value"},
            "message_history": ["message1", "message2"],
            "tool_outputs": {"tool": "output"}
        }
        
        state = State(**data)
        
        assert state.data == data["data"]
        assert state.message_history == data["message_history"]
        assert state.tool_outputs == data["tool_outputs"]


class TestStateEdgeCases:
    """Test State edge cases and error conditions."""
    
    def test_state_with_none_values(self):
        """Test State behavior with None values."""
        state = State()
        state.data["none_value"] = None
        
        assert state.data["none_value"] is None
        assert state.get_data("none_value") is None
        assert state.get_data("none_value", "default") is None
    
    def test_state_with_large_data(self):
        """Test State with large amount of data."""
        state = State()
        
        # Add large number of entries
        for i in range(1000):
            state.data[f"key_{i}"] = f"value_{i}"
            state.add_message(f"message_{i}")
            state.tool_outputs[f"tool_{i}"] = f"output_{i}"
        
        assert len(state.data) == 1000
        assert len(state.message_history) == 1000
        assert len(state.tool_outputs) == 1000
        
        # Verify specific entries
        assert state.data["key_500"] == "value_500"
        assert state.message_history[500] == "message_500"
        assert state.tool_outputs["tool_500"] == "output_500"
    
    def test_state_unicode_handling(self):
        """Test State with unicode characters."""
        state = State()
        unicode_data = {
            "russian": "Привет мир",
            "chinese": "你好世界",
            "emoji": "🚀🎉",
            "special": "áñüñÑ"
        }
        
        for key, value in unicode_data.items():
            state.data[key] = value
            state.add_message(f"Processing {value}")
            state.tool_outputs[f"tool_{key}"] = value
        
        for key, expected_value in unicode_data.items():
            assert state.data[key] == expected_value
            assert expected_value in state.message_history[list(unicode_data.keys()).index(key)]
            assert state.tool_outputs[f"tool_{key}"] == expected_value