"""Planner agent implementation for the ONIKS NeuralNet framework.

This module provides the PlannerAgent class, which serves as the task decomposition
layer. It takes complex user goals and decomposes them into atomic, manageable subtasks
that can be executed sequentially by the reasoning agent.
"""

import json
import logging
from typing import List, Optional, TYPE_CHECKING

from oniks.agents.base import BaseAgent

if TYPE_CHECKING:
    from oniks.core.state import State
    from oniks.llm.client import OllamaClient


logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """An intelligent agent that decomposes complex goals into atomic subtasks.
    
    The PlannerAgent serves as the entry point for task decomposition in the framework.
    It analyzes complex, multi-step user goals and breaks them down into a structured
    list of atomic subtasks that can be executed sequentially. This approach eliminates
    the burden on the ReasoningAgent to manage complex state comparisons and allows
    for more reliable task execution.
    
    The agent uses LLM-powered decomposition to create a clear, numbered list of
    subtasks stored in state.data['plan'], where each subtask is a simple string
    describing a single, atomic action.
    
    Attributes:
        llm_client: OllamaClient instance for LLM interactions.
    
    Example:
        >>> from oniks.llm.client import OllamaClient
        >>> llm_client = OllamaClient()
        >>> agent = PlannerAgent("planner", llm_client)
        >>> state = State()
        >>> state.data['goal'] = 'Create file hello.txt with Hello ONIKS!, then display it'
        >>> result_state = agent.execute(state)
        >>> print(result_state.data['plan'])
        [
            "Create a file named 'hello.txt' with the content 'Hello ONIKS!'",
            "Display the content of 'hello.txt' to the console",
            "Confirm that all previous steps are complete"
        ]
    """
    
    def __init__(self, name: str, llm_client: "OllamaClient") -> None:
        """Initialize the PlannerAgent with LLM client.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            
        Raises:
            ValueError: If name is empty, None, or llm_client is None.
        """
        super().__init__(name)
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        self.llm_client = llm_client
    
    def execute(self, state: "State") -> "State":
        """Execute task decomposition logic to create a structured plan.
        
        This method implements the core planning logic of the agent:
        1. Extracts the high-level goal from state.data['goal']
        2. Generates a structured prompt for LLM-powered task decomposition
        3. Invokes the LLM to create a list of atomic subtasks
        4. Parses the LLM response to extract the task list
        5. Stores the plan in state.data['plan'] as a list of strings
        6. Adds a final confirmation task to ensure completion detection
        
        Args:
            state: The current state containing the goal to decompose.
            
        Returns:
            The modified state with the decomposed plan in state.data['plan'].
        """
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Add message about planner execution
        result_state.add_message(f"Planner agent {self.name} starting task decomposition")
        
        # Extract the high-level goal
        goal = result_state.data.get('goal', '')
        
        if not goal:
            result_state.add_message("No goal found in state data")
            # Provide empty plan as fallback
            result_state.data['plan'] = []
            return result_state
        
        # Generate structured prompt for task decomposition
        decomposition_prompt = self._generate_decomposition_prompt(goal)
        result_state.data['decomposition_prompt'] = decomposition_prompt
        
        result_state.add_message("Generated task decomposition prompt")
        
        # Invoke LLM to get task decomposition
        try:
            raw_llm_response = self.llm_client.invoke(decomposition_prompt)
            result_state.data['decomposition_response'] = raw_llm_response
            result_state.add_message("Successfully received decomposition from LLM")
            
            # Parse the LLM response to extract task list
            task_list = self._parse_decomposition_response(raw_llm_response)
            
            # Add final confirmation task
            task_list.append("Confirm that all previous steps are complete")
            
            # Store the plan in state
            result_state.data['plan'] = task_list
            result_state.add_message(f"Created plan with {len(task_list)} subtasks")
            
            # Log the created plan for debugging
            for i, task in enumerate(task_list, 1):
                result_state.add_message(f"  Task {i}: {task}")
            
        except Exception as e:
            logger.error(f"Error during task decomposition: {e}")
            result_state.add_message(f"Task decomposition failed: {str(e)}")
            result_state.add_message("Falling back to basic decomposition")
            
            # Fall back to basic decomposition if LLM fails
            fallback_plan = self._perform_basic_decomposition(goal)
            result_state.data['plan'] = fallback_plan
            result_state.add_message(f"Created fallback plan with {len(fallback_plan)} subtasks")
        
        result_state.add_message(f"Planner agent {self.name} completed task decomposition")
        
        return result_state
    
    def _generate_decomposition_prompt(self, goal: str) -> str:
        """Generate a structured prompt for task decomposition.
        
        Creates a comprehensive prompt that instructs the LLM to break down
        the complex goal into atomic, executable subtasks. The prompt emphasizes
        clarity, specificity, and actionability of each subtask.
        
        Args:
            goal: The high-level goal to decompose.
            
        Returns:
            Formatted prompt string for task decomposition.
        """
        prompt = f"""--- TASK DECOMPOSITION REQUEST ---

You are a task planning expert. Your job is to break down complex goals into simple, atomic subtasks.

--- GOAL TO DECOMPOSE ---
{goal}

--- DECOMPOSITION RULES ---
1. Break the goal into the smallest possible atomic steps
2. Each step should be a single, specific action
3. Steps should be executable by tools like file operations, bash commands, etc.
4. Order steps logically from first to last
5. Use clear, action-oriented language
6. Be specific about file names, paths, and content
7. Do not include a final confirmation step - this will be added automatically

--- OUTPUT FORMAT ---
Provide the decomposed tasks as a simple numbered list, one task per line:
1. First atomic task description
2. Second atomic task description
3. Third atomic task description
(and so on...)

--- EXAMPLES ---

Example 1:
Goal: Create a file hello.txt with 'Hello World' and display its content
Output:
1. Create a file named 'hello.txt' with the content 'Hello World'
2. Display the content of 'hello.txt' to the console

Example 2:
Goal: List files in current directory and count them
Output:
1. List all files in the current directory
2. Count the number of files found

--- YOUR DECOMPOSITION ---
Break down the goal into atomic subtasks:"""

        return prompt
    
    def _parse_decomposition_response(self, response: str) -> List[str]:
        """Parse LLM response to extract the list of subtasks.
        
        Extracts numbered tasks from the LLM response and returns them as
        a clean list of strings. Handles various formatting variations
        that might appear in LLM responses.
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            List of subtask strings extracted from the response.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid decomposition response")
            return []
        
        tasks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered list items (1., 2., etc.)
            import re
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                task_description = match.group(1).strip()
                if task_description:
                    tasks.append(task_description)
                    continue
            
            # Look for dash/bullet list items
            if line.startswith('- ') or line.startswith('* '):
                task_description = line[2:].strip()
                if task_description:
                    tasks.append(task_description)
                    continue
            
            # If line contains actionable language, include it
            action_keywords = ['create', 'write', 'read', 'display', 'execute', 'run', 'list', 'count', 'show', 'copy', 'move', 'delete']
            if any(keyword in line.lower() for keyword in action_keywords):
                # Clean up the line
                cleaned_line = re.sub(r'^[^\w]*', '', line)  # Remove leading non-word chars
                if cleaned_line:
                    tasks.append(cleaned_line)
        
        logger.info(f"Parsed {len(tasks)} tasks from decomposition response")
        return tasks
    
    def _perform_basic_decomposition(self, goal: str) -> List[str]:
        """Perform basic fallback decomposition when LLM is unavailable.
        
        This method implements simple hardcoded decomposition logic as a fallback
        when the LLM service is unavailable or encounters errors.
        
        Args:
            goal: The high-level goal to decompose.
            
        Returns:
            List of basic subtasks based on pattern matching.
        """
        original_goal = goal
        if not isinstance(goal, str):
            goal = str(goal) if goal is not None else ""
        
        goal_lower = goal.lower()
        
        # Handle the common demo case
        if ("create" in goal_lower and "hello.txt" in goal_lower and 
            "hello oniks" in goal_lower and "display" in goal_lower):
            return [
                "Create a file named 'hello.txt' with the content 'Hello ONIKS!'",
                "Display the content of 'hello.txt' to the console",
                "Confirm that all previous steps are complete"
            ]
        
        # Handle simple file operations
        if "create" in goal_lower and "file" in goal_lower:
            if "display" in goal_lower or "show" in goal_lower:
                filename = "example.txt"
                # Try to extract filename from goal
                import re
                file_match = re.search(r'(\w+\.\w+)', goal)
                if file_match:
                    filename = file_match.group(1)
                
                return [
                    f"Create a file named '{filename}' with appropriate content",
                    f"Display the content of '{filename}' to the console",
                    "Confirm that all previous steps are complete"
                ]
            else:
                return [
                    "Create the requested file with specified content",
                    "Confirm that all previous steps are complete"
                ]
        
        # Handle read operations
        if "read" in goal_lower and "file" in goal_lower:
            return [
                "Read the specified file content",
                "Confirm that all previous steps are complete"
            ]
        
        # Generic fallback - try to create at least one meaningful task
        goal_str = str(original_goal) if original_goal is not None else "None"
        return [
            f"Execute the following goal: {goal_str}",
            "Confirm that all previous steps are complete"
        ]