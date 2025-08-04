"""Core tools for the ONIKS NeuralNet framework.

This module provides essential core tools including formal task completion
functionality for agents in the framework.
"""

from oniks.tools.base import Tool


class TaskCompleteTool(Tool):
    """Formal task completion tool for indicating successful task completion.
    
    This tool provides a formal mechanism for agents to signal that all parts
    of a goal have been accomplished. It should be used exclusively when all
    requirements of the current goal have been met and no further action is needed.
    
    Usage:
        The tool should be invoked with empty arguments when task completion
        is detected. This signals to the graph execution system that the
        workflow should terminate successfully.
    
    Example:
        >>> tool = TaskCompleteTool()
        >>> result = tool.execute()
        >>> print(result)
        Task finished successfully.
    """
    
    def __init__(self) -> None:
        """Initialize the TaskCompleteTool."""
        super().__init__()
        self.name = "task_complete"
        self.description = ("Используй этот инструмент, и только этот, "
                          "когда все шаги в цели пользователя выполнены.")
    
    def execute(self, **kwargs) -> str:
        """Execute the task completion functionality.
        
        This method signals that the task has been completed successfully.
        It does not require any arguments and always returns a success message.
        
        Args:
            **kwargs: Variable keyword arguments (ignored for this tool).
        
        Returns:
            String indicating successful task completion.
        """
        return "Task finished successfully."