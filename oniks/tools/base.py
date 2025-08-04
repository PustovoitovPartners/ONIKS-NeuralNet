"""Base tool abstract class for the ONIKS NeuralNet framework.

This module provides the abstract base class for all tools in the framework.
Tools are utilities that can be executed by agents during graph execution to
perform specific operations like file I/O, network requests, or data processing.
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for all tools in the ONIKS framework.
    
    A tool represents a specific capability that can be executed by agents
    during graph execution. Each tool has a unique name and description that
    helps LLM agents understand when and how to use it.
    
    Attributes:
        name: Unique identifier for the tool (must be unique across all tools).
        description: Detailed description explaining the tool's purpose, arguments,
                   and expected behavior for LLM agents.
    
    Example:
        >>> class MyTool(Tool):
        ...     def __init__(self):
        ...         self.name = "my_tool"
        ...         self.description = "Does something useful. Arguments: {'param': 'str'}"
        ...     
        ...     def execute(self, **kwargs) -> str:
        ...         return f"Executed with {kwargs}"
        >>> 
        >>> tool = MyTool()
        >>> result = tool.execute(param="test")
        >>> print(result)
        Executed with {'param': 'test'}
    """
    
    def __init__(self) -> None:
        """Initialize the tool.
        
        Subclasses must set the name and description attributes during initialization.
        """
        self.name: str = ""
        self.description: str = ""
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool's functionality with the provided arguments.
        
        This method must be implemented by all concrete tool classes to define
        their specific behavior when executed.
        
        Args:
            **kwargs: Variable keyword arguments containing the parameters
                     needed for tool execution. The specific arguments depend
                     on the tool implementation.
        
        Returns:
            String result of the tool execution. This can be a success message,
            error description, or the actual output of the tool operation.
        
        Raises:
            NotImplementedError: If not implemented by concrete subclass.
            Various exceptions: Depending on the specific tool implementation.
        """
        pass
    
    def __str__(self) -> str:
        """Return string representation of the tool.
        
        Returns:
            String representation showing the tool's name and type.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the tool.
        
        Returns:
            Detailed string representation for debugging.
        """
        return self.__str__()