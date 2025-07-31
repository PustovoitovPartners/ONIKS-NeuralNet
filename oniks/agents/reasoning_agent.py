"""Reasoning agent implementation for the ONIKS NeuralNet framework.

This module provides the ReasoningAgent class, which serves as a foundational
intelligent agent capable of analyzing goals, selecting appropriate tools,
and making decisions about next steps in graph execution.
"""

import re
import json
import ast
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
        
        # Check for task completion before proceeding
        if self._check_task_completion(result_state):
            result_state.add_message("Task completion detected - no further action needed")
            return result_state
        
        # Invoke LLM to get reasoning results
        try:
            raw_llm_response = self.llm_client.invoke(generated_prompt)
            result_state.data['llm_response'] = raw_llm_response
            result_state.add_message("Successfully received response from LLM")
            
            # Sanitize the raw LLM response before parsing
            sanitized_response = self._sanitize_llm_response(raw_llm_response)
            result_state.add_message("Applied sanitization to LLM response")
            
            # Parse the sanitized LLM response to extract tool and arguments
            self._parse_llm_response(sanitized_response, result_state)
            
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            result_state.add_message(f"LLM invocation failed: {str(e)}")
            result_state.add_message("Falling back to basic reasoning")
            
            # Fall back to basic reasoning if LLM fails
            self._perform_basic_reasoning(goal, result_state)
        
        # Check for task completion after reasoning
        self._check_task_completion(result_state)
        
        result_state.add_message(f"Reasoning agent {self.name} completed analysis")
        
        return result_state
    
    def _generate_llm_prompt(self, goal: str) -> str:
        """Generate an optimized structured prompt for weak LLM models.
        
        Creates a comprehensive prompt with clear section dividers and instruction
        at the end to leverage recency bias. Optimized for llama3:8b and other
        small models while maintaining tool selection functionality.
        
        Args:
            goal: The high-level goal extracted from the state.
            
        Returns:
            Formatted prompt string optimized for weak LLMs with clear sections.
        """
        # Build the tools list section
        tools_section = "--- AVAILABLE TOOLS ---\n"
        
        if not self.tools:
            tools_section += "No tools available.\n"
        else:
            for tool in self.tools:
                description = getattr(tool, 'description', None) or "[Description not provided]"
                tools_section += f"- {tool.name}: {description}\n"
        
        # Build examples section with clear dividers
        examples_section = """--- CORRECT FORMAT EXAMPLES ---

Example 1 (File Reading):
Goal: Read the contents of file task.txt
Tool: read_file
Arguments: {"file_path": "task.txt"}
Reasoning: Goal requires reading a file, so the read_file tool is most suitable.

Example 2 (Data Processing):
Goal: Process data from input.json and save results
Tool: process_data
Arguments: {"input_file": "input.json", "output_format": "json"}
Reasoning: Data processing is needed, so we use process_data with the input file.

Example 3 (Calculations):
Goal: Calculate the sum of numbers
Tool: calculate
Arguments: {"operation": "sum", "values": [1, 2, 3, 4, 5]}
Reasoning: Mathematical calculation is required, using the calculate tool.

--- FORMATTING RULES ---
- Tool name must be on a line starting with "Tool:"
- Arguments must be on a line starting with "Arguments:" with valid JSON
- Use double quotes in JSON, not single quotes
- JSON must be properly formatted
- Include all required parameters for the selected tool"""
        
        # Construct the complete prompt with clear sections
        prompt = f"""--- GOAL ANALYSIS AND TOOL SELECTION ---

--- CURRENT GOAL ---
{goal}

{tools_section}

{examples_section}

--- QUESTION ---
Which tool should be used and with what arguments to achieve the goal?

--- INSTRUCTION ---
Your response MUST contain ONLY the "Tool", "Arguments" and "Reasoning" sections.
DO NOT ADD any other reasoning, questions or comments.
Follow the format from the examples."""
        
        return prompt
    
    def _sanitize_llm_response(self, raw_response: str) -> str:
        """Sanitize raw LLM response before parsing to improve reliability.
        
        This method performs coarse cleaning operations to make the LLM response
        more suitable for parsing:
        1. Removes popular Markdown symbols (*, _, #) from the entire response
        2. Strips leading and trailing whitespace from each line
        3. Fixes common JSON errors like replacing single quotes with double quotes
        
        This "coarse cleaning filter" makes the system significantly more resilient
        to noise and unpredictability from language models.
        
        Args:
            raw_response: The raw response text from the LLM.
            
        Returns:
            Sanitized response string ready for parsing.
            
        Example:
            >>> agent = ReasoningAgent("test", [], llm_client)
            >>> raw = "# Tool: *read_file*\n  Arguments: {'file_path': 'test.txt'}  \n"
            >>> sanitized = agent._sanitize_llm_response(raw)
            >>> print(sanitized)
            Tool: read_file
            Arguments: {"file_path": "test.txt"}
        """
        if not isinstance(raw_response, str):
            logger.warning(f"Expected string input, got {type(raw_response).__name__}")
            return str(raw_response) if raw_response is not None else ""
        
        logger.debug(f"Sanitizing LLM response (length: {len(raw_response)} chars)")
        
        # Start with the raw response
        sanitized = raw_response
        
        # Step 1: Remove popular Markdown symbols (more targeted approach)
        # Process in order from most specific to least specific
        
        # Remove triple+ asterisks first
        sanitized = re.sub(r'\*{3,}', '', sanitized)
        
        # Remove double asterisks for bold (**text** -> text)
        sanitized = re.sub(r'\*\*([^*\n]+?)\*\*', r'\1', sanitized)
        
        # Remove single asterisks for italic (*text* -> text)  
        sanitized = re.sub(r'\*([^*\n]+?)\*', r'\1', sanitized)
        
        # Remove any remaining isolated asterisks used for emphasis
        sanitized = re.sub(r'\*+', '', sanitized)
        
        # Remove underscores used for emphasis, but preserve underscores in identifiers
        # This is tricky because we need to distinguish between _emphasis_ and file_path
        
        # Process double underscores first (__text__ -> text) 
        # Allow underscores within double underscore patterns too
        sanitized = re.sub(r'\b__([^_\s]+(?:_[^_\s]+)*?)__\b', r'\1', sanitized)
        
        # Process single underscores for emphasis (_text_ -> text)
        # This handles both simple words and compound identifiers like _read_file_
        # Using word boundaries to properly detect emphasis patterns
        sanitized = re.sub(r'\b_([^_\s]+(?:_[^_\s]+)*?)_\b', r'\1', sanitized)
        
        # Remove hash symbols used for headers at start of lines
        sanitized = re.sub(r'^#+\s*', '', sanitized, flags=re.MULTILINE)
        
        # Step 2: Strip leading and trailing whitespace from each line
        lines = sanitized.split('\n')
        stripped_lines = [line.strip() for line in lines]
        sanitized = '\n'.join(stripped_lines)
        
        # Step 3: Fix common JSON errors
        # Replace single quotes with double quotes in JSON-like structures
        # This is a targeted fix for argument values
        sanitized = re.sub(r"'([^']*)'", r'"\1"', sanitized)
        
        # Additional JSON fixes for common malformed patterns
        # Fix unquoted keys in JSON-like structures (key: value -> "key": value)
        sanitized = re.sub(r'(\w+)(\s*:\s*"[^"]*")', r'"\1"\2', sanitized)
        
        # Remove any remaining empty lines caused by cleaning
        lines = [line for line in sanitized.split('\n') if line.strip()]
        sanitized = '\n'.join(lines)
        
        logger.debug(f"Sanitization complete (new length: {len(sanitized)} chars)")
        
        return sanitized
    
    def _parse_llm_response(self, llm_response: str, state: "State") -> None:
        """Parse LLM response to extract tool name and arguments using multi-stage approach.
        
        This method implements a robust multi-stage parsing strategy:
        1. First tries clean JSON parsing with json.loads()
        2. Falls back to regex extraction for malformed responses
        3. Normalizes various argument formats (tuples, strings, etc.) to dictionaries
        
        Parsing patterns:
        - "Tool: [tool_name]" to extract the recommended tool
        - "Arguments: [json_object]" to extract the tool arguments
        
        Args:
            llm_response: The raw response text from the LLM.
            state: The state object to update with parsed results.
        """
        logger.info(f"Parsing LLM response (length: {len(llm_response)} chars)")
        
        # Extract tool name using regex
        self._extract_tool_name(llm_response, state)
        
        # Extract and parse arguments using multi-stage approach
        self._extract_and_parse_arguments(llm_response, state)
    
    def _extract_tool_name(self, llm_response: str, state: "State") -> None:
        """Extract tool name from LLM response.
        
        Args:
            llm_response: The raw response text from the LLM.
            state: The state object to update with extracted tool name.
        """
        tool_match = re.search(r'Tool:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            # Clean up tool name (remove quotes, extra whitespace)
            tool_name = tool_name.strip("\"'`").strip()
            
            state.data['next_tool'] = tool_name
            state.add_message(f"Extracted tool from LLM response: {tool_name}")
            logger.info(f"Extracted tool: {tool_name}")
        else:
            # Try fallback patterns for tool detection
            fallback_match = re.search(r'(?:tool to use is|use tool|use the tool)\s+(\w+)', 
                                     llm_response, re.IGNORECASE)
            if fallback_match:
                tool_name = fallback_match.group(1).strip()
                state.data['next_tool'] = tool_name
                state.add_message(f"Extracted tool using fallback pattern: {tool_name}")
                logger.info(f"Extracted tool (fallback): {tool_name}")
            else:
                state.add_message("No tool found in LLM response")
                logger.warning("No 'Tool:' pattern found in LLM response")
    
    def _extract_and_parse_arguments(self, llm_response: str, state: "State") -> None:
        """Extract and parse arguments using multi-stage parsing approach.
        
        Implements robust parsing with multiple fallback strategies:
        1. Direct JSON parsing
        2. Regex extraction with bracket matching
        3. Tuple/list normalization to dictionary format
        4. Single value normalization
        
        Args:
            llm_response: The raw response text from the LLM.
            state: The state object to update with parsed arguments.
        """
        # Stage 1: Look for Arguments: pattern
        args_match = re.search(r'Arguments:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if not args_match:
            # Try to find arguments in parentheses or brackets anywhere in response
            bracket_match = re.search(r'Arguments?\s*[\(\[\{]([^\)\]\}]+)[\)\]\}]', 
                                    llm_response, re.IGNORECASE | re.DOTALL)
            if bracket_match:
                args_str = f"({bracket_match.group(1).strip()})"  # Wrap in parentheses for tuple parsing
                logger.info("Found arguments in brackets using fallback regex")
            else:
                state.add_message("No arguments found in LLM response")
                logger.warning("No 'Arguments:' pattern found in LLM response")
                return
        else:
            args_str = args_match.group(1).strip()
        
        # Stage 2: Multi-stage parsing of extracted arguments
        parsed_args = self._parse_arguments_multi_stage(args_str, state)
        
        if parsed_args is not None:
            state.data['tool_args'] = parsed_args
            
            # Set individual argument keys in state.data for ToolNode compatibility
            if isinstance(parsed_args, dict):
                for key, value in parsed_args.items():
                    state.data[key] = value
            
            state.add_message(f"Extracted arguments from LLM response: {parsed_args}")
            logger.info(f"Extracted arguments: {parsed_args}")
    
    def _parse_arguments_multi_stage(self, args_str: str, state: "State") -> any:
        """Parse arguments string using multi-stage approach with normalization.
        
        Parsing stages:
        1. Clean JSON parsing with json.loads()
        2. Regex extraction for common patterns
        3. Normalization of tuples, lists, and single values to dictionaries
        4. Fallback string parsing
        
        Args:
            args_str: The arguments string to parse.
            state: The state object for logging messages.
            
        Returns:
            Parsed arguments as dictionary, list, or None if parsing fails.
        """
        logger.debug(f"Multi-stage parsing arguments: {args_str}")
        
        # Stage 1: Try clean JSON parsing
        try:
            parsed_args = json.loads(args_str)
            logger.debug("Stage 1: Successfully parsed as clean JSON")
            return self._normalize_arguments(parsed_args, state)
        except json.JSONDecodeError:
            logger.debug("Stage 1: Clean JSON parsing failed, trying AST parsing")
        
        # Stage 1b: Try AST literal_eval for Python literals (tuples, lists, etc.)
        try:
            parsed_args = ast.literal_eval(args_str)
            logger.debug("Stage 1b: Successfully parsed as Python literal")
            return self._normalize_arguments(parsed_args, state)
        except (ValueError, SyntaxError):
            logger.debug("Stage 1b: AST parsing failed, trying regex extraction")
            
            # Special case: if it looks like a parenthesized expression that failed AST,
            # try manual tuple parsing
            if args_str.startswith('(') and args_str.endswith(')'):
                content = args_str[1:-1].strip()
                if ',' in content:
                    values = [v.strip().strip("\"'`") for v in content.split(',') if v.strip()]
                    logger.debug(f"Stage 1c: Manual tuple parsing result: {values}")
                    return self._normalize_arguments(tuple(values), state)
        
        # Stage 2: Try regex extraction for common malformed patterns
        normalized_args = self._extract_with_regex(args_str, state)
        if normalized_args is not None:
            return normalized_args
        
        # Stage 3: Try to extract values from various bracket patterns
        bracket_extracted = self._extract_from_brackets(args_str, state)
        if bracket_extracted is not None:
            return bracket_extracted
        
        # Stage 4: Final fallback - treat as single string value
        logger.warning(f"All parsing stages failed for arguments: {args_str}")
        state.add_message(f"Failed to parse arguments, falling back to string interpretation: {args_str}")
        
        # Clean up the string
        clean_str = args_str.strip("\"'`")
        
        # Try to guess the parameter name based on common patterns
        if (any(keyword in clean_str.lower() for keyword in ['file', 'path', 'filename']) or
            '.' in clean_str or '/' in clean_str or '\\' in clean_str):
            return {'file_path': clean_str}
        
        return {'value': clean_str}
    
    def _normalize_arguments(self, parsed_args: any, state: "State") -> any:
        """Normalize various argument formats to appropriate dictionaries.
        
        Handles normalization of:
        - Tuples like ('task.txt',) -> {'file_path': 'task.txt'}
        - Lists like ['task.txt'] -> {'file_path': 'task.txt'}
        - Single strings -> {'file_path': string} or {'value': string}
        
        Args:
            parsed_args: The parsed arguments to normalize.
            state: The state object for logging messages.
            
        Returns:
            Normalized arguments as dictionary.
        """
        if isinstance(parsed_args, dict):
            logger.debug("Arguments already in dictionary format")
            return parsed_args
        
        elif isinstance(parsed_args, (tuple, list)):
            logger.debug(f"Normalizing {type(parsed_args).__name__} to dictionary")
            if len(parsed_args) == 1:
                value = parsed_args[0]
                # Guess parameter name based on value content
                if isinstance(value, str) and ('.' in value or '/' in value or '\\' in value):
                    return {'file_path': value}
                else:
                    return {'value': value}
            elif len(parsed_args) == 2:
                # Check if first element looks like a filename/path
                first_val = parsed_args[0]
                if isinstance(first_val, str) and ('.' in first_val or '/' in first_val or '\\' in first_val):
                    # Treat as file_path and additional argument
                    return {'file_path': first_val, 'arg_1': parsed_args[1]}
                else:
                    # Assume key-value pair
                    return {str(parsed_args[0]): parsed_args[1]}
            else:
                # Multiple values - create indexed dictionary
                return {f'arg_{i}': val for i, val in enumerate(parsed_args)}
        
        elif isinstance(parsed_args, str):
            logger.debug("Normalizing string to dictionary")
            # Single string value - guess parameter name
            if '.' in parsed_args or '/' in parsed_args or '\\' in parsed_args:
                return {'file_path': parsed_args}
            else:
                return {'value': parsed_args}
        
        else:
            logger.debug(f"Normalizing {type(parsed_args).__name__} to dictionary")
            return {'value': parsed_args}
    
    def _extract_with_regex(self, args_str: str, state: "State") -> any:
        """Extract arguments using regex patterns for common malformed formats.
        
        Handles patterns like:
        - {'file_path': 'task.txt'} with missing quotes
        - (file_path: task.txt) with wrong brackets
        - file_path=task.txt parameter format
        
        Args:
            args_str: The arguments string to parse with regex.
            state: The state object for logging messages.
            
        Returns:
            Extracted arguments as dictionary or None if extraction fails.
        """
        logger.debug("Stage 2: Attempting regex extraction")
        
        # Try to fix common JSON formatting issues
        fixed_json = args_str
        
        # Fix missing quotes around keys
        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
        
        # Fix missing quotes around string values
        fixed_json = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_./\\-]*)', r': "\1"', fixed_json)
        
        # Fix single quotes to double quotes
        fixed_json = fixed_json.replace("'", '"')
        
        # Try parsing the fixed JSON
        try:
            parsed_args = json.loads(fixed_json)
            logger.debug("Stage 2: Successfully parsed with regex fixes")
            return self._normalize_arguments(parsed_args, state)
        except json.JSONDecodeError:
            logger.debug("Stage 2: Regex-fixed JSON parsing failed")
        
        # Try key=value pattern extraction
        kv_matches = re.findall(r'(\w+)\s*[=:]\s*([^,\s]+)', args_str)
        if kv_matches:
            result = {}
            for key, value in kv_matches:
                # Clean up value
                value = value.strip("\"'`")
                result[key] = value
            logger.debug(f"Stage 2: Extracted key-value pairs: {result}")
            return result
        
        return None
    
    def _extract_from_brackets(self, args_str: str, state: "State") -> any:
        """Extract arguments from various bracket patterns.
        
        Handles patterns like:
        - (task.txt) -> {'file_path': 'task.txt'}
        - [task.txt, other] -> {'file_path': 'task.txt', 'arg_1': 'other'}
        - {malformed json content}
        
        Args:
            args_str: The arguments string to parse.
            state: The state object for logging messages.
            
        Returns:
            Extracted arguments as dictionary or None if extraction fails.
        """
        logger.debug("Stage 3: Attempting bracket content extraction")
        
        # Look for content in parentheses (tuple-like)
        paren_match = re.search(r'\(([^)]+)\)', args_str)
        if paren_match:
            content = paren_match.group(1).strip()
            # Split by comma and clean up
            values = [v.strip().strip("\"'`") for v in content.split(',') if v.strip()]
            logger.debug(f"Stage 3: Found parentheses content: {values}")
            return self._normalize_arguments(tuple(values), state)
        
        # Look for content in square brackets (list-like)
        bracket_match = re.search(r'\[([^\]]+)\]', args_str)
        if bracket_match:
            content = bracket_match.group(1).strip()
            # Split by comma and clean up
            values = [v.strip().strip("\"'`") for v in content.split(',') if v.strip()]
            logger.debug(f"Stage 3: Found square bracket content: {values}")
            return self._normalize_arguments(values, state)
        
        # Look for content in curly braces (dict-like)
        brace_match = re.search(r'\{([^}]+)\}', args_str)
        if brace_match:
            content = brace_match.group(1).strip()
            # Try to extract key-value pairs
            kv_matches = re.findall(r'["\']?(\w+)["\']?\s*:\s*["\']?([^,}]+)["\']?', content)
            if kv_matches:
                result = {}
                for key, value in kv_matches:
                    result[key] = value.strip().strip("\"'`")
                logger.debug(f"Stage 3: Extracted from braces: {result}")
                return result
        
        logger.debug("Stage 3: No bracket patterns matched")
        return None
    
    def _check_task_completion(self, state: "State") -> bool:
        """Check if the task has been completed based on goal and tool outputs.
        
        This method implements the completion logic:
        - If goal contains 'выведи' (display) and tool_outputs from execute_bash_command
          contains 'Hello ONIKS!', then task is completed.
        
        Args:
            state: The current state to check for completion.
            
        Returns:
            True if task is completed, False otherwise.
        """
        goal = state.data.get('goal', '')
        
        # Check if goal contains 'display'
        if 'display' in goal.lower():
            # Check if execute_bash_command has been executed and contains 'Hello ONIKS!'
            if state.tool_outputs:
                bash_output = state.tool_outputs.get('execute_bash_command', '')
                if isinstance(bash_output, str) and 'Hello ONIKS!' in bash_output:
                    state.data['task_completed'] = True
                    state.add_message("Task completion detected: file created and content displayed")
                    logger.info("Task marked as completed")
                    return True
        
        return False
    
    def _perform_basic_reasoning(self, goal: str, state: "State") -> None:
        """Perform basic fallback reasoning when LLM is unavailable.
        
        This method implements simple hardcoded reasoning logic as a fallback
        when the LLM service is unavailable or encounters errors.
        
        Current logic:
        - Multi-step goal handling for file creation and display task
        - Step 1: Create hello.txt file with 'Hello ONIKS!' content
        - Step 2: Display file content using bash command
        
        Args:
            goal: The high-level goal to analyze.
            state: The state object to update with reasoning results.
        """
        # Handle non-string goals gracefully
        if not isinstance(goal, str):
            goal = str(goal) if goal is not None else ""
        
        goal_lower = goal.lower()
        
        # Check if this is the multi-step demo goal
        if ("create" in goal_lower and "hello.txt" in goal_lower and "hello oniks" in goal_lower and "display" in goal_lower):
            # Check if file has been created yet
            if not state.tool_outputs.get('write_file'):
                # Step 1: Create the file
                state.data['next_tool'] = 'write_file'
                state.data['tool_args'] = {
                    'file_path': 'hello.txt',
                    'content': 'Hello ONIKS!'
                }
                state.data['file_path'] = 'hello.txt'
                state.data['content'] = 'Hello ONIKS!'
                
                state.add_message(
                    "Fallback reasoning: Step 1 - Creating hello.txt file with 'Hello ONIKS!' content"
                )
            elif not state.tool_outputs.get('execute_bash_command'):
                # Step 2: Display file content
                state.data['next_tool'] = 'execute_bash_command'
                state.data['tool_args'] = {'command': 'cat hello.txt'}
                state.data['command'] = 'cat hello.txt'
                
                state.add_message(
                    "Fallback reasoning: Step 2 - Displaying hello.txt content with cat command"
                )
            else:
                state.add_message(
                    "Fallback reasoning: Both steps completed, checking for task completion"
                )
        
        # Legacy support for simple file reading tasks
        elif ("read" in goal_lower and "file" in goal_lower):
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