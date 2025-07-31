"""State management module for the ONIKS NeuralNet graph execution framework.

This module provides the base State class that serves as the foundation for storing
and managing graph execution state throughout the framework.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class State(BaseModel):
    """Base class for storing and managing graph execution state.
    
    This class serves as the foundation for state management in the graph execution
    framework. It provides basic data storage, message history tracking, and tool
    output storage capabilities while being designed for extensibility.
    
    Attributes:
        data: Dictionary for storing arbitrary state data during graph execution.
        message_history: List of messages or events that occurred during execution.
        tool_outputs: Dictionary storing outputs from tool executions during graph execution.
    
    Example:
        >>> state = State()
        >>> state.data["user_input"] = "Hello, world!"
        >>> state.message_history.append("Processing user input")
        >>> state.tool_outputs["read_file"] = "File content"
        >>> print(state.data["user_input"])
        Hello, world!
    """
    
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary for storing arbitrary state data during graph execution"
    )
    
    message_history: List[str] = Field(
        default_factory=list,
        description="List of messages or events that occurred during execution"
    )
    
    tool_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary storing outputs from tool executions during graph execution"
    )
    
    def add_message(self, message: str) -> None:
        """Add a message to the execution history.
        
        Args:
            message: The message to add to the history.
            
        Example:
            >>> state = State()
            >>> state.add_message("Started processing")
            >>> print(state.message_history)
            ['Started processing']
        """
        self.message_history.append(message)
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the state dictionary.
        
        Args:
            key: The key to retrieve from the data dictionary.
            default: Default value to return if key is not found.
            
        Returns:
            The value associated with the key, or the default value if key not found.
            
        Example:
            >>> state = State()
            >>> state.data["test"] = "value"
            >>> print(state.get_data("test"))
            value
            >>> print(state.get_data("missing", "default"))
            default
        """
        return self.data.get(key, default)
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data in the state dictionary.
        
        Args:
            key: The key to set in the data dictionary.
            value: The value to associate with the key.
            
        Example:
            >>> state = State()
            >>> state.set_data("result", 42)
            >>> print(state.data["result"])
            42
        """
        self.data[key] = value
    
    def clear_data(self) -> None:
        """Clear all data from the state dictionary.
        
        Example:
            >>> state = State()
            >>> state.data["test"] = "value"
            >>> state.clear_data()
            >>> print(len(state.data))
            0
        """
        self.data.clear()
    
    def clear_history(self) -> None:
        """Clear all messages from the execution history.
        
        Example:
            >>> state = State()
            >>> state.add_message("test message")
            >>> state.clear_history()
            >>> print(len(state.message_history))
            0
        """
        self.message_history.clear()