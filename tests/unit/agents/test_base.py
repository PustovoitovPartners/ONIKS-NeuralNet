"""Unit tests for the BaseAgent class."""

import pytest
from oniks.agents.base import BaseAgent
from oniks.core.state import State


class TestBaseAgentAbstract:
    """Test the abstract BaseAgent class."""
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("test_agent")
    
    def test_base_agent_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        class IncompleteAgent(BaseAgent):
            def __init__(self, name: str):
                super().__init__(name)
            # Missing execute method
        
        with pytest.raises(TypeError):
            IncompleteAgent("incomplete_agent")
    
    def test_base_agent_inherits_from_node(self):
        """Test that BaseAgent properly inherits from Node."""
        class ConcreteAgent(BaseAgent):
            def execute(self, state: State) -> State:
                return state
        
        agent = ConcreteAgent("test_agent")
        
        # Should have Node attributes and methods
        assert hasattr(agent, 'name')
        assert agent.name == "test_agent"
        
        # Should be able to execute (inherited from Node interface)
        state = State()
        result = agent.execute(state)
        assert isinstance(result, State)


class TestBaseAgentConcrete:
    """Test concrete implementations of BaseAgent."""
    
    @pytest.fixture
    def concrete_agent(self):
        """Create a concrete agent for testing."""
        class ConcreteAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message(f"Agent {self.name} executed")
                result_state.data["agent_executed"] = True
                return result_state
        
        return ConcreteAgent("concrete_agent")
    
    def test_concrete_agent_initialization(self, concrete_agent):
        """Test concrete agent initialization."""
        assert concrete_agent.name == "concrete_agent"
        assert hasattr(concrete_agent, 'execute')
    
    def test_concrete_agent_initialization_empty_name_raises_error(self):
        """Test concrete agent initialization with empty name raises ValueError."""
        class ConcreteAgent(BaseAgent):
            def execute(self, state: State) -> State:
                return state
        
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ConcreteAgent("")
    
    def test_concrete_agent_initialization_none_name_raises_error(self):
        """Test concrete agent initialization with None name raises ValueError."""
        class ConcreteAgent(BaseAgent):
            def execute(self, state: State) -> State:
                return state
        
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            ConcreteAgent(None)
    
    def test_concrete_agent_execute_success(self, concrete_agent):
        """Test successful agent execution."""
        initial_state = State()
        initial_state.data["test"] = "value"
        initial_state.add_message("Initial message")
        
        result_state = concrete_agent.execute(initial_state)
        
        # Should be a new state object
        assert result_state is not initial_state
        
        # Should preserve original data
        assert result_state.data["test"] == "value"
        assert "Initial message" in result_state.message_history
        
        # Should add agent-specific changes
        assert result_state.data["agent_executed"] is True
        assert f"Agent {concrete_agent.name} executed" in result_state.message_history
    
    def test_concrete_agent_execute_preserves_original_state(self, concrete_agent):
        """Test that original state is not modified during execution."""
        original_state = State()
        original_state.data["test"] = "value"
        original_state.add_message("Original message")
        
        result_state = concrete_agent.execute(original_state)
        
        # Original state should be unchanged
        assert len(original_state.message_history) == 1
        assert "agent_executed" not in original_state.data
        
        # Result state should have changes
        assert len(result_state.message_history) > 1
        assert "agent_executed" in result_state.data
    
    def test_concrete_agent_string_representation(self, concrete_agent):
        """Test agent string representation."""
        assert str(concrete_agent) == "ConcreteAgent(name='concrete_agent')"
        assert repr(concrete_agent) == "ConcreteAgent(name='concrete_agent')"


class TestBaseAgentImplementationPatterns:
    """Test common BaseAgent implementation patterns."""
    
    def test_agent_with_state_analysis(self):
        """Test agent that analyzes state before acting."""
        class AnalyzingAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                # Analyze current state
                has_goal = 'goal' in state.data
                message_count = len(state.message_history)
                tool_count = len(state.tool_outputs)
                
                # Make decisions based on analysis
                result_state.add_message(f"Analysis: goal={has_goal}, messages={message_count}, tools={tool_count}")
                
                if has_goal:
                    result_state.data["goal_processed"] = True
                    result_state.add_message("Goal found and processed")
                else:
                    result_state.add_message("No goal found")
                
                return result_state
        
        agent = AnalyzingAgent("analyzing_agent")
        
        # Test with goal
        state_with_goal = State()
        state_with_goal.data["goal"] = "test goal"
        state_with_goal.add_message("Initial message")
        
        result1 = agent.execute(state_with_goal)
        
        assert "goal=True" in result1.message_history[-2]
        assert "messages=1" in result1.message_history[-2]
        assert "Goal found and processed" in result1.message_history[-1]
        assert result1.data["goal_processed"] is True
        
        # Test without goal
        state_without_goal = State()
        state_without_goal.add_message("Initial message")
        
        result2 = agent.execute(state_without_goal)
        
        assert "goal=False" in result2.message_history[-2]
        assert "No goal found" in result2.message_history[-1]
        assert "goal_processed" not in result2.data
    
    def test_agent_with_conditional_behavior(self):
        """Test agent with conditional behavior based on state."""
        class ConditionalAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                mode = result_state.data.get("mode", "default")
                
                if mode == "aggressive":
                    result_state.data["action"] = "attack"
                    result_state.add_message("Executing aggressive action")
                elif mode == "defensive":
                    result_state.data["action"] = "defend"
                    result_state.add_message("Executing defensive action")
                elif mode == "passive":
                    result_state.data["action"] = "wait"
                    result_state.add_message("Executing passive action")
                else:
                    result_state.data["action"] = "explore"
                    result_state.add_message("Executing default exploration")
                
                return result_state
        
        agent = ConditionalAgent("conditional_agent")
        
        # Test different modes
        modes_and_actions = [
            ("aggressive", "attack"),
            ("defensive", "defend"),
            ("passive", "wait"),
            ("unknown", "explore"),
            (None, "explore")  # Default case
        ]
        
        for mode, expected_action in modes_and_actions:
            state = State()
            if mode is not None:
                state.data["mode"] = mode
            
            result = agent.execute(state)
            
            assert result.data["action"] == expected_action
            if expected_action == "attack":
                assert "Executing aggressive action" in result.message_history[-1]
            elif expected_action == "defend":
                assert "Executing defensive action" in result.message_history[-1]
            elif expected_action == "wait":
                assert "Executing passive action" in result.message_history[-1]
            elif expected_action == "explore":
                assert "Executing default exploration" in result.message_history[-1]
    
    def test_agent_with_error_handling(self):
        """Test agent with comprehensive error handling."""
        class ErrorHandlingAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                try:
                    operation = result_state.data.get("operation", "safe")
                    
                    if operation == "safe":
                        result_state.data["result"] = "success"
                        result_state.add_message("Safe operation completed")
                    elif operation == "divide_by_zero":
                        result = 10 / 0
                        result_state.data["result"] = result
                    elif operation == "key_error":
                        data = {"key": "value"}
                        result_state.data["result"] = data["missing_key"]
                    elif operation == "custom_error":
                        raise ValueError("Custom error for testing")
                    else:
                        result_state.data["result"] = f"Unknown operation: {operation}"
                        result_state.add_message(f"Unknown operation: {operation}")
                
                except ZeroDivisionError as e:
                    result_state.data["error"] = "Division by zero"
                    result_state.add_message(f"Error: Division by zero - {e}")
                except KeyError as e:
                    result_state.data["error"] = f"Key not found: {e}"
                    result_state.add_message(f"Error: Key not found - {e}")
                except ValueError as e:
                    result_state.data["error"] = f"Value error: {e}"
                    result_state.add_message(f"Error: Value error - {e}")
                except Exception as e:
                    result_state.data["error"] = f"Unexpected error: {e}"
                    result_state.add_message(f"Error: Unexpected error - {e}")
                
                return result_state
        
        agent = ErrorHandlingAgent("error_handling_agent")
        
        # Test safe operation
        safe_state = State()
        safe_state.data["operation"] = "safe"
        
        safe_result = agent.execute(safe_state)
        assert safe_result.data["result"] == "success"
        assert "Safe operation completed" in safe_result.message_history
        
        # Test error conditions
        error_cases = [
            ("divide_by_zero", "Division by zero"),
            ("key_error", "Key not found"),
            ("custom_error", "Value error: Custom error for testing")
        ]
        
        for operation, expected_error in error_cases:
            error_state = State()
            error_state.data["operation"] = operation
            
            error_result = agent.execute(error_state)
            assert "error" in error_result.data
            assert expected_error in error_result.data["error"]
            assert "Error:" in error_result.message_history[-1]
    
    def test_agent_with_state_modification_patterns(self):
        """Test agent with different state modification patterns."""
        class StateModifyingAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                action = result_state.data.get("action", "default")
                
                if action == "accumulate":
                    # Accumulate values
                    current_value = result_state.data.get("accumulator", 0)
                    increment = result_state.data.get("increment", 1)
                    result_state.data["accumulator"] = current_value + increment
                    result_state.add_message(f"Accumulated: {current_value} + {increment} = {current_value + increment}")
                
                elif action == "transform":
                    # Transform existing data
                    text = result_state.data.get("text", "")
                    result_state.data["text"] = text.upper()
                    result_state.data["text_length"] = len(text)
                    result_state.add_message(f"Transformed text to uppercase, length: {len(text)}")
                
                elif action == "filter":
                    # Filter list data
                    items = result_state.data.get("items", [])
                    threshold = result_state.data.get("threshold", 0)
                    filtered_items = [item for item in items if isinstance(item, (int, float)) and item > threshold]
                    result_state.data["filtered_items"] = filtered_items
                    result_state.add_message(f"Filtered {len(items)} items to {len(filtered_items)} items above {threshold}")
                
                else:
                    result_state.add_message("No specific action taken")
                
                return result_state
        
        agent = StateModifyingAgent("state_modifying_agent")
        
        # Test accumulate action
        accumulate_state = State()
        accumulate_state.data.update({
            "action": "accumulate",
            "accumulator": 10,
            "increment": 5
        })
        
        accumulate_result = agent.execute(accumulate_state)
        assert accumulate_result.data["accumulator"] == 15
        
        # Test transform action
        transform_state = State()
        transform_state.data.update({
            "action": "transform",
            "text": "hello world"
        })
        
        transform_result = agent.execute(transform_state)
        assert transform_result.data["text"] == "HELLO WORLD"
        assert transform_result.data["text_length"] == 11
        
        # Test filter action
        filter_state = State()
        filter_state.data.update({
            "action": "filter",
            "items": [1, 5, 3, 8, 2, 10, -1],
            "threshold": 4
        })
        
        filter_result = agent.execute(filter_state)
        assert filter_result.data["filtered_items"] == [5, 8, 10]
    
    def test_agent_with_multi_step_processing(self):
        """Test agent that performs multi-step processing."""
        class MultiStepAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                # Step 1: Validate input
                result_state.add_message("Step 1: Validating input")
                input_data = result_state.data.get("input", [])
                if not isinstance(input_data, list):
                    result_state.data["error"] = "Input must be a list"
                    return result_state
                
                # Step 2: Process data
                result_state.add_message("Step 2: Processing data")
                processed_data = []
                for item in input_data:
                    if isinstance(item, bool):
                        processed_data.append(str(item))
                    elif isinstance(item, (int, float)):
                        processed_data.append(item * 2)
                    elif isinstance(item, str):
                        processed_data.append(item.upper())
                    else:
                        processed_data.append(str(item))
                
                result_state.data["processed"] = processed_data
                
                # Step 3: Generate summary
                result_state.add_message("Step 3: Generating summary")
                summary = {
                    "total_items": len(input_data),
                    "processed_items": len(processed_data),
                    "item_types": {}
                }
                
                for item in input_data:
                    item_type = type(item).__name__
                    summary["item_types"][item_type] = summary["item_types"].get(item_type, 0) + 1
                
                result_state.data["summary"] = summary
                
                # Step 4: Finalize
                result_state.add_message("Step 4: Processing completed successfully")
                result_state.data["status"] = "completed"
                
                return result_state
        
        agent = MultiStepAgent("multi_step_agent")
        
        # Test with valid input
        valid_state = State()
        valid_state.data["input"] = [1, 2, "hello", 3.14, "world", True]
        
        valid_result = agent.execute(valid_state)
        
        assert valid_result.data["status"] == "completed"
        assert valid_result.data["processed"] == [2, 4, "HELLO", 6.28, "WORLD", "True"]
        assert valid_result.data["summary"]["total_items"] == 6
        assert valid_result.data["summary"]["item_types"]["int"] == 2
        assert valid_result.data["summary"]["item_types"]["str"] == 2
        assert len(valid_result.message_history) == 4  # 4 steps
        
        # Test with invalid input
        invalid_state = State()
        invalid_state.data["input"] = "not a list"
        
        invalid_result = agent.execute(invalid_state)
        
        assert invalid_result.data["error"] == "Input must be a list"
        assert "status" not in invalid_result.data


class TestBaseAgentEdgeCases:
    """Test BaseAgent edge cases and boundary conditions."""
    
    def test_agent_with_empty_state(self):
        """Test agent execution with completely empty state."""
        class EmptyStateAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                result_state.add_message("Processing empty state")
                result_state.data["processed_empty"] = True
                return result_state
        
        agent = EmptyStateAgent("empty_state_agent")
        empty_state = State()
        
        result = agent.execute(empty_state)
        
        assert result.data["processed_empty"] is True
        assert "Processing empty state" in result.message_history
    
    def test_agent_with_large_state(self):
        """Test agent execution with large state objects."""
        class LargeStateAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                # Count items in state
                data_count = len(result_state.data)
                message_count = len(result_state.message_history)
                tool_count = len(result_state.tool_outputs)
                
                result_state.add_message(f"Processed large state: {data_count} data items, {message_count} messages, {tool_count} tools")
                result_state.data["state_size_processed"] = data_count + message_count + tool_count
                
                return result_state
        
        agent = LargeStateAgent("large_state_agent")
        
        # Create large state
        large_state = State()
        
        # Add lots of data
        for i in range(1000):
            large_state.data[f"key_{i}"] = f"value_{i}"
            large_state.add_message(f"Message {i}")
            large_state.tool_outputs[f"tool_{i}"] = f"output_{i}"
        
        result = agent.execute(large_state)
        
        assert result.data["state_size_processed"] == 3000  # 1000 + 1000 + 1000
        assert "Processed large state: 1000 data items" in result.message_history[-1]
    
    def test_agent_with_unicode_handling(self):
        """Test agent handling of unicode content."""
        class UnicodeAgent(BaseAgent):
            def execute(self, state: State) -> State:
                result_state = state.model_copy(deep=True)
                
                # Process unicode data
                text = result_state.data.get("text", "")
                result_state.data["text_length"] = len(text)
                result_state.data["text_bytes"] = len(text.encode('utf-8'))
                
                result_state.add_message(f"Processed unicode text: '{text}' (chars: {len(text)}, bytes: {len(text.encode('utf-8'))})")
                
                return result_state
        
        agent = UnicodeAgent("unicode_agent")
        
        unicode_texts = [
            "Hello 世界!",
            "Привет мир!",
            "🌍🚀🎉",
            "áñüñÑ",
            "नमस्ते दुनिया"
        ]
        
        for text in unicode_texts:
            unicode_state = State()
            unicode_state.data["text"] = text
            
            result = agent.execute(unicode_state)
            
            assert result.data["text_length"] == len(text)
            assert result.data["text_bytes"] == len(text.encode('utf-8'))
            assert text in result.message_history[-1]