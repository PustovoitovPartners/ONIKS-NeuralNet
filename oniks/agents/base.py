"""Base agent abstract class for the ONIKS NeuralNet framework.

This module provides the abstract base class for all intelligent agents in the framework.
Agents are specialized nodes that can perform complex reasoning and decision-making tasks
during graph execution.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

from oniks.core.graph import Node

if TYPE_CHECKING:
    from oniks.core.state import State


class BaseAgent(Node):
    """Abstract base class for all intelligent agents in the ONIKS framework.
    
    An agent is a specialized node that can perform complex reasoning, decision-making,
    and interaction with tools or external systems. Agents extend the base Node class
    with additional capabilities for intelligent behavior.
    
    All concrete agent implementations must inherit from this class and implement
    the execute method to define their specific reasoning and execution logic.
    
    Attributes:
        name: Unique identifier for the agent (inherited from Node).
    
    Example:
        >>> class MyAgent(BaseAgent):
        ...     def __init__(self, name: str):
        ...         super().__init__(name)
        ...     
        ...     def execute(self, state: State) -> State:
        ...         # Implement agent-specific logic here
        ...         state.add_message(f"Agent {self.name} executed")
        ...         return state
        >>> 
        >>> agent = MyAgent("my_agent")
        >>> state = State()
        >>> result_state = agent.execute(state)
        >>> print(result_state.message_history[-1])
        Agent my_agent executed
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the agent with the given name.
        
        Args:
            name: Unique identifier for this agent.
            
        Raises:
            ValueError: If name is empty or None.
        """
        super().__init__(name)
    
    @abstractmethod
    def execute(self, state: "State") -> "State":
        """Execute the agent's reasoning and decision-making logic.
        
        This method must be implemented by all concrete agent classes to define
        their specific behavior when executed in the graph. Agents typically
        analyze the current state, make decisions about next actions, and
        potentially modify the state with new information or instructions.
        
        Args:
            state: The current state of the graph execution.
            
        Returns:
            The modified state after executing this agent's logic.
            
        Raises:
            NotImplementedError: If not implemented by concrete subclass.
        """
        pass
    
    def __str__(self) -> str:
        """Return string representation of the agent.
        
        Returns:
            String representation showing the agent's name and type.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the agent.
        
        Returns:
            Detailed string representation for debugging.
        """
        return self.__str__()