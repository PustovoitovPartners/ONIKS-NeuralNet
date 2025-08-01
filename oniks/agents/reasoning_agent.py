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
        """Execute simplified reasoning logic for single subtask from plan[0].
        
        This method implements simplified reasoning logic that works with atomic subtasks:
        1. Extracts the current subtask from state.data['plan'][0]
        2. Generates a simple prompt for tool selection based on the atomic task
        3. Invokes the LLM to get tool selection results
        4. Parses the LLM response to extract tool and arguments
        5. Updates the state with reasoning results
        
        The agent no longer needs complex state management or history tracking since
        each subtask is atomic and self-contained.
        
        Args:
            state: The current state containing plan and other execution data.
            
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
        result_state.add_message(f"Reasoning agent {self.name} starting analysis of current subtask")
        
        # Extract the current subtask from plan[0]
        plan = result_state.data.get('plan', [])
        
        if not plan or not isinstance(plan, list):
            result_state.add_message("No plan found in state data or plan is empty")
            return result_state
        
        current_task = plan[0]
        result_state.add_message(f"Current subtask: {current_task}")
        
        # Check if this is a function call string (new tool-based format)
        if self._is_function_call_format(current_task):
            result_state.add_message("Detected function call format, parsing directly")
            success = self._parse_function_call(current_task, result_state)
            if success:
                result_state.add_message(f"Successfully parsed function call: {current_task}")
                return result_state
            else:
                result_state.add_message(f"Failed to parse function call: {current_task}, falling back to LLM analysis")
                # Continue with LLM analysis as fallback
        
        # Check if this is the final confirmation task (legacy format)
        if "confirm that all previous steps are complete" in current_task.lower():
            result_state.data['next_tool'] = 'task_complete'
            result_state.data['tool_args'] = {}
            result_state.add_message("Final confirmation task detected, selecting task_complete tool")
            return result_state
        
        # Generate simplified prompt for tool selection
        generated_prompt = self._generate_task_prompt(current_task)
        result_state.data['last_prompt'] = generated_prompt
        
        result_state.add_message("Generated simplified prompt for tool selection")
        
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
            self._perform_basic_reasoning(current_task, result_state)
        
        result_state.add_message(f"Reasoning agent {self.name} completed analysis")
        
        return result_state
    
    def _generate_task_prompt(self, current_task: str) -> str:
        """Generate a simplified prompt for tool selection based on atomic subtask.
        
        Creates a focused prompt that asks the LLM to select the appropriate tool
        for a single, atomic subtask. This is much simpler than the previous complex
        goal analysis since each task is self-contained and actionable.
        
        Args:
            current_task: The current atomic subtask to accomplish.
            
        Returns:
            Formatted prompt string for tool selection.
        """
        # Build the tools list section
        tools_section = "--- AVAILABLE TOOLS ---\n"
        
        if not self.tools:
            tools_section += "No tools available.\n"
        else:
            for tool in self.tools:
                description = getattr(tool, 'description', None) or "[Description not provided]"
                tools_section += f"- {tool.name}: {description}\n"
        
        # Build simplified examples section
        examples_section = """--- EXAMPLES ---

Example 1:
Task: Create a file named 'hello.txt' with the content 'Hello World'
Tool: write_file
Arguments: {"file_path": "hello.txt", "content": "Hello World"}

Example 2:
Task: Display the content of 'data.txt' to the console
Tool: execute_bash_command
Arguments: {"command": "cat data.txt"}

Example 3:
Task: List all files in the current directory
Tool: list_files
Arguments: {}

--- FORMATTING RULES ---
- Tool name must be on a line starting with "Tool:"
- Arguments must be on a line starting with "Arguments:" with valid JSON
- Use double quotes in JSON, not single quotes
- JSON must be properly formatted"""
        
        # Construct the simplified prompt
        prompt = f"""--- TOOL SELECTION FOR ATOMIC TASK ---

--- CURRENT TASK ---
{current_task}

{tools_section}

{examples_section}

--- QUESTION ---
Which tool should be used and with what arguments to complete this specific task?

--- INSTRUCTION ---
Analyze the current task and select the most appropriate tool with correct arguments.
The task is atomic and self-contained - simply choose the tool that directly accomplishes it."""
        
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
        
        # Always set tool_args since _parse_arguments_multi_stage never returns None for failure
        # (it always falls back to some parsed result or creates a fallback dict)
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
        """Normalize various argument formats while preserving valid data structures.
        
        This method now preserves properly structured data types (lists, dicts, booleans, etc.)
        that were successfully parsed by ast.literal_eval or json.loads. Normalization to 
        dictionaries only occurs for ambiguous structures like single-element tuples or
        unstructured strings that need parameter name guessing.
        
        Handles normalization of:
        - Dictionaries: Preserved as-is
        - Lists: Preserved as-is (FIXED: no longer converted to indexed dicts)
        - Booleans, None, numbers: Preserved as-is (FIXED)
        - Single-element tuples: Converted to {'param': value} with guessed param name
        - Multi-element tuples: Only normalized if they appear to be unstructured arguments
        - Single strings: Converted to {'param': value} with guessed param name
        
        Args:
            parsed_args: The parsed arguments to normalize.
            state: The state object for logging messages.
            
        Returns:
            Normalized arguments preserving original data types when appropriate.
        """
        # Preserve dictionaries as-is - they're already properly structured
        if isinstance(parsed_args, dict):
            logger.debug("Arguments already in dictionary format - preserving as-is")
            return parsed_args
        
        # FIXED: Preserve lists as-is - they're valid Python data structures
        # Lists should not be converted to indexed dictionaries
        elif isinstance(parsed_args, list):
            logger.debug("Arguments are a list - preserving as-is")
            return parsed_args
        
        # FIXED: Preserve other primitive types (bool, None, numbers) as-is
        elif isinstance(parsed_args, (bool, type(None), int, float)):
            logger.debug(f"Arguments are {type(parsed_args).__name__} - preserving as-is")
            return parsed_args
        
        # Handle tuples with special normalization logic
        elif isinstance(parsed_args, tuple):
            logger.debug(f"Normalizing tuple to appropriate format")
            if len(parsed_args) == 1:
                value = parsed_args[0]
                # Single-element tuple gets converted to dict with guessed parameter name
                if isinstance(value, str) and ('.' in value or '/' in value or '\\' in value):
                    return {'file_path': value}
                else:
                    return {'value': value}
            elif len(parsed_args) == 2:
                # Two-element tuple: check if it's a key-value pair or path + argument
                first_val = parsed_args[0]
                if isinstance(first_val, str) and ('.' in first_val or '/' in first_val or '\\' in first_val):
                    # Treat as file_path and additional argument
                    return {'file_path': first_val, 'arg_1': parsed_args[1]}
                else:
                    # Assume key-value pair
                    return {str(parsed_args[0]): parsed_args[1]}
            else:
                # Multi-element tuple: preserve as-is unless it appears to be unstructured args
                # For now, preserve tuples as-is to maintain data integrity
                logger.debug("Multi-element tuple preserved as-is")
                return parsed_args
        
        # Handle single strings by guessing parameter names
        elif isinstance(parsed_args, str):
            logger.debug("Normalizing string to dictionary with guessed parameter name")
            # Single string value - guess parameter name
            if '.' in parsed_args or '/' in parsed_args or '\\' in parsed_args:
                return {'file_path': parsed_args}
            else:
                return {'value': parsed_args}
        
        # Handle any other types by preserving them as-is
        else:
            logger.debug(f"Preserving {type(parsed_args).__name__} as-is")
            return parsed_args
    
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
    
    
    def _perform_basic_reasoning(self, current_task: str, state: "State") -> None:
        """Perform basic fallback reasoning for atomic subtasks when LLM is unavailable.
        
        This method implements simple hardcoded reasoning logic as a fallback
        when the LLM service is unavailable or encounters errors. Since tasks
        are now atomic, the logic is much simpler.
        
        Args:
            current_task: The current atomic subtask to analyze.
            state: The state object to update with reasoning results.
        """
        # Handle non-string tasks gracefully
        if not isinstance(current_task, str):
            current_task = str(current_task) if current_task is not None else ""
        
        task_lower = current_task.lower()
        
        # File creation tasks
        if "create" in task_lower and "file" in task_lower:
            import re
            # Try to extract filename and content
            filename_match = re.search(r"'([^']+\.\w+)'", current_task)
            content_match = re.search(r"content '([^']+)'", current_task)
            
            filename = filename_match.group(1) if filename_match else "example.txt"
            content = content_match.group(1) if content_match else "Example content"
            
            state.data['next_tool'] = 'write_file'
            state.data['tool_args'] = {
                'file_path': filename,
                'content': content
            }
            state.data['file_path'] = filename
            state.data['content'] = content
            
            state.add_message(
                f"Fallback reasoning: Creating file {filename} with content '{content}'"
            )
        
        # Display/show file content tasks
        elif ("display" in task_lower or "show" in task_lower) and "content" in task_lower:
            import re
            # Try to extract filename
            filename_match = re.search(r"'([^']+\.\w+)'", current_task)
            filename = filename_match.group(1) if filename_match else "hello.txt"
            
            state.data['next_tool'] = 'execute_bash_command'
            state.data['tool_args'] = {'command': f'cat {filename}'}
            state.data['command'] = f'cat {filename}'
            
            state.add_message(
                f"Fallback reasoning: Displaying content of {filename} using cat command"
            )
        
        # List files tasks
        elif "list" in task_lower and "file" in task_lower:
            state.data['next_tool'] = 'list_files'
            state.data['tool_args'] = {}
            
            state.add_message(
                "Fallback reasoning: Listing files using list_files tool"
            )
        
        # Read file tasks
        elif "read" in task_lower and "file" in task_lower:
            import re
            filename_match = re.search(r"'([^']+\.\w+)'", current_task)
            filename = filename_match.group(1) if filename_match else "task.txt"
            
            state.data['next_tool'] = 'read_file'
            state.data['tool_args'] = {'file_path': filename}
            state.data['file_path'] = filename
            
            state.add_message(
                f"Fallback reasoning: Reading file {filename} using read_file tool"
            )
        
        else:
            state.add_message(
                f"No specific fallback reasoning rule matched for task: {current_task}"
            )
            # Default to task completion if we can't figure out what to do
            state.data['next_tool'] = 'task_complete'
            state.data['tool_args'] = {}
    
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
    
    def _is_function_call_format(self, task: str) -> bool:
        """Check if a task string is in function call format.
        
        Detects if the task follows the new tool-based format like:
        "write_file(file_path='hello.txt', content='Hello World')"
        instead of descriptive format like:
        "Create a file named 'hello.txt' with the content 'Hello World'"
        
        Args:
            task: The task string to check.
            
        Returns:
            True if the task is in function call format, False otherwise.
        """
        if not isinstance(task, str) or not task.strip():
            return False
        
        task = task.strip()
        
        # Check if it matches basic function call pattern: name(...)
        import re
        match = re.match(r'^(\w+)\s*\(.*\)$', task)
        if not match:
            return False
        
        tool_name = match.group(1)
        
        # Check if the tool name is one of our available tools
        available_tool_names = [tool.name for tool in self.tools]
        
        # Always allow task_complete as it's a standard completion tool
        if tool_name == 'task_complete':
            return True
            
        return tool_name in available_tool_names
    
    def _parse_function_call(self, function_call: str, state: "State") -> bool:
        """Parse a function call string to extract tool name and arguments.
        
        Parses function call strings like:
        "write_file(file_path='hello.txt', content='Hello World')"
        
        And sets the appropriate state variables:
        - state.data['next_tool'] = 'write_file'
        - state.data['tool_args'] = {'file_path': 'hello.txt', 'content': 'Hello World'}
        - state.data[key] = value for each argument
        
        Args:
            function_call: The function call string to parse.
            state: The state object to update with parsed results.
            
        Returns:
            True if parsing was successful, False otherwise.
        """
        if not isinstance(function_call, str) or not function_call.strip():
            logger.warning("Invalid function call string provided")
            return False
        
        function_call = function_call.strip()
        
        try:
            # Extract tool name using regex
            import re
            match = re.match(r'^(\w+)\s*\((.*)\)$', function_call)
            if not match:
                logger.warning(f"Function call does not match expected pattern: {function_call}")
                return False
            
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Set the tool name
            state.data['next_tool'] = tool_name
            state.add_message(f"Extracted tool name: {tool_name}")
            
            # Parse arguments if any
            if args_str:
                parsed_args = self._parse_function_arguments(args_str)
                if parsed_args is not None:
                    state.data['tool_args'] = parsed_args
                    
                    # Set individual argument keys in state.data for ToolNode compatibility
                    # Only set individual keys if parsed_args is a dictionary
                    if isinstance(parsed_args, dict):
                        for key, value in parsed_args.items():
                            state.data[key] = value
                    
                    state.add_message(f"Extracted arguments: {parsed_args}")
                else:
                    logger.warning(f"Failed to parse arguments: {args_str}")
                    return False
            else:
                # No arguments
                state.data['tool_args'] = {}
                state.add_message("No arguments found in function call")
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing function call '{function_call}': {e}")
            return False
    
    def _parse_function_arguments(self, args_str: str) -> Optional[any]:
        """Parse function call arguments string preserving original data types.
        
        This method now uses ast.literal_eval as the preferred parsing method for
        robust handling of all Python data types including lists, dicts, booleans, etc.
        
        Handles various argument formats:
        - Keyword arguments: file_path='hello.txt', content='Hello World'
        - Mixed quotes: file_path="hello.txt", content='Hello World'
        - Lists: files=['a.txt', 'b.txt']
        - Dicts: config={'debug': True, 'port': 8080}
        - Booleans: enabled=True, debug=False
        - Numbers: count=42, ratio=3.14
        - None values: optional=None
        - No arguments: (empty string)
        
        Args:
            args_str: The arguments string to parse (content inside parentheses).
            
        Returns:
            Parsed arguments as dictionary, list, or other Python types, or None if parsing fails.
        """
        if not args_str.strip():
            return {}
        
        try:
            # Method 1: Try using AST to parse as function call arguments
            # Create a dummy function call and parse it
            dummy_call = f"dummy_func({args_str})"
            import ast
            
            try:
                parsed = ast.parse(dummy_call, mode='eval')
                if isinstance(parsed.body, ast.Call):
                    args_dict = {}
                    
                    # Handle keyword arguments
                    for keyword in parsed.body.keywords:
                        if keyword.arg:  # keyword.arg is the parameter name
                            # Extract the value using ast.literal_eval for safe parsing
                            try:
                                # First try to extract simple constants
                                if isinstance(keyword.value, ast.Constant):
                                    args_dict[keyword.arg] = keyword.value.value
                                elif isinstance(keyword.value, ast.Str):  # Python < 3.8 compatibility
                                    args_dict[keyword.arg] = keyword.value.s
                                elif isinstance(keyword.value, ast.Num):  # Python < 3.8 compatibility
                                    args_dict[keyword.arg] = keyword.value.n
                                else:
                                    # For complex expressions (lists, dicts, etc.), use ast.literal_eval
                                    # Convert the AST node back to string and parse it safely
                                    value_str = ast.unparse(keyword.value)
                                    args_dict[keyword.arg] = ast.literal_eval(value_str)
                            except (ValueError, SyntaxError) as e:
                                # If ast.literal_eval fails, fall back to string representation
                                logger.debug(f"Failed to parse complex value for {keyword.arg}: {e}")
                                args_dict[keyword.arg] = ast.unparse(keyword.value)
                    
                    # Handle positional arguments (if any)
                    for i, arg in enumerate(parsed.body.args):
                        try:
                            # First try to extract simple constants
                            if isinstance(arg, ast.Constant):
                                args_dict[f'arg_{i}'] = arg.value
                            elif isinstance(arg, ast.Str):  # Python < 3.8 compatibility
                                args_dict[f'arg_{i}'] = arg.s
                            elif isinstance(arg, ast.Num):  # Python < 3.8 compatibility
                                args_dict[f'arg_{i}'] = arg.n
                            else:
                                # For complex expressions (lists, dicts, etc.), use ast.literal_eval
                                # Convert the AST node back to string and parse it safely
                                value_str = ast.unparse(arg)
                                args_dict[f'arg_{i}'] = ast.literal_eval(value_str)
                        except (ValueError, SyntaxError) as e:
                            # If ast.literal_eval fails, fall back to string representation
                            logger.debug(f"Failed to parse complex value for arg_{i}: {e}")
                            args_dict[f'arg_{i}'] = ast.unparse(arg)
                    
                    logger.debug(f"AST parsing successful: {args_dict}")
                    return args_dict
                    
            except (SyntaxError, ValueError, AttributeError) as e:
                logger.debug(f"AST parsing failed: {e}, trying regex method")
        
        except Exception as e:
            logger.debug(f"AST method failed: {e}, trying regex method")
        
        # Method 2: Fallback to regex parsing for keyword arguments
        try:
            import re
            args_dict = {}
            
            # Enhanced pattern to match key=value pairs with various value types
            # This pattern handles: strings, numbers, booleans, lists, dicts, None
            pattern = r"(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|$)"
            matches = re.findall(pattern, args_str)
            
            for match in matches:
                key = match[0]
                value_str = match[1].strip()
                
                # Try to parse the value using ast.literal_eval for safe evaluation
                try:
                    # Remove trailing comma if present
                    value_str = value_str.rstrip(',').strip()
                    parsed_value = ast.literal_eval(value_str)
                    args_dict[key] = parsed_value
                    logger.debug(f"Regex parsed {key}={value_str} as {type(parsed_value).__name__}: {parsed_value}")
                except (ValueError, SyntaxError):
                    # If ast.literal_eval fails, try simple string processing
                    if value_str.startswith(("'", '"')) and value_str.endswith(("'", '"')):
                        # Remove quotes for string values
                        args_dict[key] = value_str[1:-1]
                    else:
                        # Keep as string
                        args_dict[key] = value_str
                    logger.debug(f"Regex fallback for {key}={value_str} as string")
            
            if args_dict:
                logger.debug(f"Enhanced regex parsing successful: {args_dict}")
                return args_dict
            else:
                logger.warning(f"No keyword arguments found in: {args_str}")
                
        except Exception as e:
            logger.warning(f"Enhanced regex parsing failed: {e}")
        
        # Method 3: Last resort - try to extract simple values
        try:
            # If it's just a single quoted string, treat it as a single argument
            import re
            single_value_match = re.match(r"^['\"]([^'\"]*)['\"]$", args_str.strip())
            if single_value_match:
                value = single_value_match.group(1)
                # Try to guess the parameter name based on common patterns
                if '.' in value or '/' in value or '\\' in value:
                    return {'file_path': value}
                else:
                    return {'value': value}
                    
        except Exception as e:
            logger.warning(f"Simple value parsing failed: {e}")
        
        logger.error(f"All parsing methods failed for arguments: {args_str}")
        return None