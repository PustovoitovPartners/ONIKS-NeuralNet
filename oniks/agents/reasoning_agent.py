"""Reasoning agent implementation for the ONIKS NeuralNet framework.

This module provides the ReasoningAgent class, which serves as a foundational
intelligent agent capable of analyzing goals, selecting appropriate tools,
and making decisions about next steps in graph execution.
"""

import re
import json
import logging
from typing import List, Optional, TYPE_CHECKING

from oniks.agents.base import BaseAgent

if TYPE_CHECKING:
    from oniks.core.state import State
    from oniks.tools.base import Tool
    from oniks.llm.client import OllamaClient


logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """An intelligent agent that performs LLM-powered reasoning and tool selection.
    
    The ReasoningAgent analyzes the current state, extracts high-level goals,
    and uses a local LLM through OllamaClient to determine which tools should
    be used to achieve those goals. It generates structured prompts, sends them
    to the LLM, and parses the responses to extract tool recommendations.
    
    This agent integrates with Ollama to provide real AI-powered reasoning
    capabilities for intelligent decision-making in the framework.
    
    Attributes:
        tools: List of available tools that this agent can recommend for use.
        llm_client: OllamaClient instance for LLM interactions.
    
    Example:
        >>> from oniks.tools.file_tools import ReadFileTool
        >>> from oniks.llm.client import OllamaClient
        >>> tools = [ReadFileTool()]
        >>> llm_client = OllamaClient()
        >>> agent = ReasoningAgent("reasoning_agent", tools, llm_client)
        >>> state = State()
        >>> state.data['goal'] = 'Read the contents of file task.txt'
        >>> result_state = agent.execute(state)
        >>> print(result_state.data['next_tool'])
        read_file
    """
    
    def __init__(self, name: str, tools: List["Tool"], llm_client: "OllamaClient") -> None:
        """Initialize the ReasoningAgent with available tools and LLM client.
        
        Args:
            name: Unique identifier for this agent.
            tools: List of Tool instances that this agent can recommend for use.
            llm_client: OllamaClient instance for LLM interactions.
            
        Raises:
            ValueError: If name is empty, None, tools list is None, or llm_client is None.
            TypeError: If tools is not a list.
        """
        super().__init__(name)
        
        if tools is None:
            raise ValueError("Tools list cannot be None")
        
        if not isinstance(tools, list):
            raise TypeError(f"Tools must be a list, got {type(tools).__name__}")
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        self.tools = tools
        self.llm_client = llm_client
    
    def execute(self, state: "State") -> "State":
        """Execute LLM-powered reasoning logic to determine next tool and arguments.
        
        This method implements the core reasoning logic of the agent:
        1. Extracts the current high-level goal from state.data['goal']
        2. Generates a structured prompt for LLM integration
        3. Invokes the LLM through OllamaClient to get reasoning results
        4. Parses the LLM response to extract tool and arguments
        5. Updates the state with reasoning results
        
        The LLM response is expected to contain "Tool: [tool_name]" and
        "Arguments: [json_object]" lines that the agent will parse.
        
        Args:
            state: The current state containing goal and other execution data.
            
        Returns:
            The modified state with reasoning results, including:
            - last_prompt: Generated prompt sent to LLM
            - llm_response: Raw response from the LLM
            - next_tool: Recommended tool name (if determined)
            - tool_args: Arguments for the recommended tool (if determined)
        """
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Add message about reasoning execution
        result_state.add_message(f"Reasoning agent {self.name} starting LLM-powered analysis")
        
        # Extract the current high-level goal
        goal = result_state.data.get('goal', '')
        
        if not goal:
            result_state.add_message("No goal found in state data")
            return result_state
        
        # Generate structured prompt for LLM integration
        generated_prompt = self._generate_llm_prompt(goal)
        result_state.data['last_prompt'] = generated_prompt
        
        result_state.add_message("Generated LLM prompt for goal analysis")
        
        # Invoke LLM to get reasoning results
        try:
            llm_response = self.llm_client.invoke(generated_prompt)
            result_state.data['llm_response'] = llm_response
            result_state.add_message("Successfully received response from LLM")
            
            # Parse LLM response to extract tool and arguments
            self._parse_llm_response(llm_response, result_state)
            
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            result_state.add_message(f"LLM invocation failed: {str(e)}")
            result_state.add_message("Falling back to basic reasoning")
            
            # Fall back to basic reasoning if LLM fails
            self._perform_basic_reasoning(goal, result_state)
        
        result_state.add_message(f"Reasoning agent {self.name} completed analysis")
        
        return result_state
    
    def _generate_llm_prompt(self, goal: str) -> str:
        """Generate a structured prompt for LLM integration.
        
        Creates a comprehensive prompt that includes the goal, available tools,
        and a clear question for the LLM to answer about tool selection.
        
        Args:
            goal: The high-level goal extracted from the state.
            
        Returns:
            Formatted prompt string ready for LLM integration.
        """
        # Build the tools list section
        tools_section = "Available tools:\n"
        
        if not self.tools:
            tools_section += "No tools available.\n"
        else:
            for tool in self.tools:
                description = getattr(tool, 'description', None) or "[No description provided]"
                tools_section += f"- {tool.name}: {description}\n"
        
        # Construct the complete prompt
        prompt = f"""Goal Analysis and Tool Selection

Current Goal: {goal}

{tools_section}
Question: Which tool should be used next and with what arguments to achieve the goal?

Please provide your reasoning and specify:
1. The tool name to use
2. The arguments required for the tool
3. Why this tool is appropriate for the current goal

Response format:
Tool: [tool_name]
Arguments: [tool_arguments]
Reasoning: [explanation]"""
        
        return prompt
    
    def _parse_llm_response(self, llm_response: str, state: "State") -> None:
        """Parse LLM response to extract tool name and arguments.
        
        This method parses the LLM response looking for specific patterns:
        - "Tool: [tool_name]" to extract the recommended tool
        - "Arguments: [json_object]" to extract the tool arguments
        
        Args:
            llm_response: The raw response text from the LLM.
            state: The state object to update with parsed results.
        """
        logger.info(f"Parsing LLM response (length: {len(llm_response)} chars)")
        
        # Look for Tool: pattern
        tool_match = re.search(r'Tool:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            state.data['next_tool'] = tool_name
            state.add_message(f"Extracted tool from LLM response: {tool_name}")
            logger.info(f"Extracted tool: {tool_name}")
        else:
            state.add_message("No tool found in LLM response")
            logger.warning("No 'Tool:' pattern found in LLM response")
        
        # Look for Arguments: pattern
        args_match = re.search(r'Arguments:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if args_match:
            args_str = args_match.group(1).strip()
            try:
                # Try to parse as JSON
                tool_args = json.loads(args_str)
                state.data['tool_args'] = tool_args
                
                # Also set individual argument keys in state.data for ToolNode compatibility
                if isinstance(tool_args, dict):
                    for key, value in tool_args.items():
                        state.data[key] = value
                
                state.add_message(f"Extracted arguments from LLM response: {tool_args}")
                logger.info(f"Extracted arguments: {tool_args}")
                
            except json.JSONDecodeError as e:
                state.add_message(f"Failed to parse arguments as JSON: {args_str}")
                logger.error(f"JSON parsing error for arguments '{args_str}': {e}")
        else:
            state.add_message("No arguments found in LLM response")
            logger.warning("No 'Arguments:' pattern found in LLM response")
    
    def _perform_basic_reasoning(self, goal: str, state: "State") -> None:
        """Perform basic fallback reasoning when LLM is unavailable.
        
        This method implements simple hardcoded reasoning logic as a fallback
        when the LLM service is unavailable or encounters errors.
        
        Current logic:
        - If goal contains "read" and "file", recommend read_file tool
        - If goal contains "прочитать" and "файл", recommend read_file tool
        - Set appropriate arguments for the recommended tool
        
        Args:
            goal: The high-level goal to analyze.
            state: The state object to update with reasoning results.
        """
        # Handle non-string goals gracefully
        if not isinstance(goal, str):
            goal = str(goal) if goal is not None else ""
        
        goal_lower = goal.lower()
        
        # Basic reasoning: if goal is about reading a file (English or Russian)
        if (("read" in goal_lower and "file" in goal_lower) or 
            ("прочитать" in goal_lower and "файл" in goal_lower)):
            state.data['next_tool'] = 'read_file'
            state.data['tool_args'] = {'file_path': 'task.txt'}
            # Also set the file_path directly in state.data for ToolNode to use
            state.data['file_path'] = 'task.txt'
            
            state.add_message(
                "Fallback reasoning result: Goal involves reading a file, "
                "recommending read_file tool with task.txt"
            )
        else:
            state.add_message(
                f"No specific fallback reasoning rule matched for goal: {goal}"
            )
    
    def get_available_tools(self) -> List["Tool"]:
        """Get the list of available tools for this agent.
        
        Returns:
            List of Tool instances available to this agent.
        """
        return self.tools.copy()
    
    def add_tool(self, tool: "Tool") -> None:
        """Add a new tool to the agent's available tools.
        
        Args:
            tool: The Tool instance to add.
            
        Raises:
            ValueError: If tool is None.
            TypeError: If tool is not a Tool instance.
        """
        if tool is None:
            raise ValueError("Tool cannot be None")
        
        # Note: We can't do isinstance check here due to circular imports
        # In a real implementation, this would be handled differently
        
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent's available tools.
        
        Args:
            tool_name: Name of the tool to remove.
            
        Returns:
            True if a tool was removed, False if no tool with that name was found.
        """
        original_length = len(self.tools)
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        return len(self.tools) < original_length