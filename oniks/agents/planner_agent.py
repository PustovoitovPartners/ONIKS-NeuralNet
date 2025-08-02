"""Planner agent implementation for the ONIKS NeuralNet framework.

This module provides the PlannerAgent class, which serves as the task decomposition
layer. It takes complex user goals and decomposes them into atomic, manageable subtasks
that can be executed sequentially by the reasoning agent.

This agent operates in STRICT LLM-ONLY mode - no fallbacks, no defaults, no caches.
If the LLM is unavailable or fails, the agent throws a critical exception that
stops the entire graph execution.
"""

import json
import logging
import re
import signal
import time
import traceback
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from oniks.agents.base import BaseAgent
from oniks.core.exceptions import LLMUnavailableError, PlanningTimeoutError

if TYPE_CHECKING:
    from oniks.core.state import State
    from oniks.llm.client import OllamaClient
    from oniks.tools.base import Tool


logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """STRICT LLM-ONLY agent that decomposes complex goals into tool-based execution plans.
    
    CRITICAL: This agent operates in STRICT LLM-ONLY mode with NO FALLBACKS.
    - If LLM is unavailable, agent throws LLMUnavailableError immediately
    - If LLM returns HTTP error, agent throws LLMUnavailableError immediately  
    - If LLM returns empty/invalid response, agent throws LLMUnavailableError immediately
    - NO default plans, NO cached plans, NO hardcoded plans, NO silent degradation
    
    The agent uses ONLY LLM-powered decomposition to create function call sequences
    stored in state.data['plan'], where each step is a concrete tool invocation
    with specific arguments rather than abstract descriptions.
    
    SUCCESS CRITERIA:
    - LLM must return HTTP 200 OK
    - Response body must contain non-empty generated content
    - Response content must be parseable into valid tool calls
    - Only then is the operation tagged as [LLM-POWERED]
    
    FAILURE BEHAVIOR:
    - ANY other condition results in LLMUnavailableError
    - Error is tagged as [LLM-ERROR] with full error details
    - Graph execution stops immediately
    
    Attributes:
        llm_client: OllamaClient instance for LLM interactions.
        available_tools: List of Tool instances that can be used in plans.
    
    Example:
        >>> from oniks.llm.client import OllamaClient
        >>> from oniks.tools.fs_tools import WriteFileTool
        >>> from oniks.tools.shell_tools import ExecuteBashCommandTool
        >>> llm_client = OllamaClient()
        >>> tools = [WriteFileTool(), ExecuteBashCommandTool()]
        >>> agent = PlannerAgent("planner", llm_client, tools)
        >>> state = State()
        >>> state.data['goal'] = 'Create file hello.txt with Hello ONIKS!, then display it'
        >>> # This will either succeed with [LLM-POWERED] plan or fail with LLMUnavailableError
        >>> result_state = agent.execute(state)  # May raise LLMUnavailableError
        >>> print(result_state.data['plan'])
        [
            "write_file(file_path='hello.txt', content='Hello ONIKS!')",
            "execute_bash_command(command='cat hello.txt')",
            "task_complete()"
        ]
    """
    
    def __init__(self, name: str, llm_client: "OllamaClient", available_tools: Optional[List["Tool"]] = None, timeout_seconds: float = 60.0) -> None:
        """Initialize the PlannerAgent with LLM client and available tools.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            available_tools: List of Tool instances that can be used in plans.
                           If None, defaults to empty list.
            timeout_seconds: Maximum time allowed for planning cycle in seconds.
                           Defaults to 60.0 seconds.
            
        Raises:
            ValueError: If name is empty, None, llm_client is None, or timeout_seconds is not positive.
            TypeError: If available_tools is not a list.
        """
        super().__init__(name)
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        if available_tools is None:
            available_tools = []
        
        if not isinstance(available_tools, list):
            raise TypeError(f"Available tools must be a list, got {type(available_tools).__name__}")
        
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError(f"Timeout seconds must be a positive number, got {timeout_seconds}")
        
        self.llm_client = llm_client
        self.available_tools = available_tools
        self.timeout_seconds = timeout_seconds
    
    def execute(self, state: "State") -> "State":
        """Execute STRICT LLM-ONLY task decomposition logic to create an executable plan.
        
        CRITICAL: This method operates in STRICT LLM-ONLY mode with NO FALLBACKS.
        
        The method:
        1. Validates that a goal exists (fails immediately if not)
        2. Generates a structured prompt for LLM-powered tool-based decomposition
        3. Invokes the LLM with strict validation (HTTP 200 + non-empty content)
        4. Parses and validates the LLM response using robust regex-based parsing
        5. Creates tool call sequence ONLY from successful LLM response
        6. Tags operation as [LLM-POWERED] ONLY on complete success
        7. Enforces timeout limit to prevent infinite hangs
        
        FAILURE CONDITIONS:
        - No goal in state (throws LLMUnavailableError)
        - Planning cycle exceeds timeout (throws PlanningTimeoutError)
        - LLM connection failure (throws LLMUnavailableError)
        - LLM HTTP error response (throws LLMUnavailableError)
        - Empty or invalid LLM response content (throws LLMUnavailableError)
        - Unparseable LLM response (throws LLMUnavailableError)
        
        Args:
            state: The current state containing the goal to decompose.
            
        Returns:
            The modified state with the LLM-generated plan in state.data['plan'].
            
        Raises:
            LLMUnavailableError: If LLM is unavailable or returns invalid response.
            PlanningTimeoutError: If planning cycle exceeds timeout limit.
        """
        # Record start time for timeout enforcement
        start_time = time.time()
        
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Generate unique agent execution ID for correlation
        agent_execution_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Add message about planner execution with timeout info
        result_state.add_message(f"Planner agent {self.name} starting STRICT LLM-ONLY task decomposition (timeout: {self.timeout_seconds}s)")
        
        # CRITICAL: Extract and validate the goal - fail immediately if missing
        goal = result_state.data.get('goal', '').strip()
        
        if not goal:
            error_msg = "No goal found in state data - cannot proceed without goal"
            logger.error(f"[LLM-ERROR-{agent_execution_id}] {error_msg}")
            result_state.add_message(f"[LLM-ERROR] {error_msg}")
            
            raise LLMUnavailableError(
                message="Goal validation failed - no goal provided",
                request_details={
                    "agent_execution_id": agent_execution_id,
                    "state_data_keys": list(result_state.data.keys()),
                    "goal_value": goal
                },
                correlation_id=agent_execution_id
            )
        
        # BULLETPROOF LOGGING: Log strict LLM-only operation start
        logger.info(f"[PLANNER-{agent_execution_id}] Starting STRICT LLM-ONLY decomposition at {timestamp}")
        logger.info(f"[PLANNER-{agent_execution_id}] Goal to decompose: {goal}")
        logger.info(f"[PLANNER-{agent_execution_id}] Available tools: {[tool.name for tool in self.available_tools]}")
        logger.info(f"[PLANNER-{agent_execution_id}] NO FALLBACKS - LLM must succeed or operation fails")
        
        # Generate structured prompt for task decomposition
        decomposition_prompt = self._generate_decomposition_prompt(goal)
        result_state.data['decomposition_prompt'] = decomposition_prompt
        
        result_state.add_message("Generated task decomposition prompt for LLM")
        
        # CRITICAL: Invoke LLM with strict validation and timeout enforcement
        try:
            # Check timeout before LLM call
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                error_msg = f"Planning timeout before LLM call - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s"
                logger.error(f"[TIMEOUT-{agent_execution_id}] {error_msg}")
                result_state.add_message(f"[TIMEOUT-ERROR] {error_msg}")
                
                raise PlanningTimeoutError(
                    message="Planning cycle timed out before LLM invocation",
                    timeout_seconds=self.timeout_seconds,
                    elapsed_seconds=elapsed_time,
                    correlation_id=agent_execution_id,
                    request_details={
                        "agent_execution_id": agent_execution_id,
                        "goal": goal,
                        "phase": "pre_llm_call"
                    }
                )
            logger.info(f"[PLANNER-{agent_execution_id}] Calling LLM for task decomposition (STRICT MODE)")
            
            # Call LLM with timeout awareness - this may throw OllamaConnectionError or other exceptions
            raw_llm_response = self.llm_client.invoke(decomposition_prompt)
            
            # Check timeout after LLM call
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                error_msg = f"Planning timeout after LLM call - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s"
                logger.error(f"[TIMEOUT-{agent_execution_id}] {error_msg}")
                result_state.add_message(f"[TIMEOUT-ERROR] {error_msg}")
                
                raise PlanningTimeoutError(
                    message="Planning cycle timed out after LLM response",
                    timeout_seconds=self.timeout_seconds,
                    elapsed_seconds=elapsed_time,
                    correlation_id=agent_execution_id,
                    request_details={
                        "agent_execution_id": agent_execution_id,
                        "goal": goal,
                        "phase": "post_llm_call",
                        "response_length": len(raw_llm_response) if raw_llm_response else 0
                    }
                )
            
            # STRICT VALIDATION: Validate LLM response is non-empty and usable
            if not raw_llm_response or not isinstance(raw_llm_response, str) or not raw_llm_response.strip():
                error_msg = "LLM returned empty or invalid response content"
                logger.error(f"[LLM-ERROR-{agent_execution_id}] {error_msg}")
                logger.error(f"[LLM-ERROR-{agent_execution_id}] Raw response: {repr(raw_llm_response)}")
                result_state.add_message(f"[LLM-ERROR] {error_msg}")
                
                raise LLMUnavailableError(
                    message="LLM response validation failed - empty or invalid content",
                    request_details={
                        "agent_execution_id": agent_execution_id,
                        "goal": goal,
                        "response_type": type(raw_llm_response).__name__,
                        "response_length": len(raw_llm_response) if raw_llm_response else 0,
                        "response_repr": repr(raw_llm_response)
                    },
                    correlation_id=agent_execution_id
                )
            
            result_state.data['decomposition_response'] = raw_llm_response
            
            # STRICT VALIDATION: Parse the LLM response to extract tool call sequence with robust parsing
            tool_call_list = self._parse_decomposition_response_robust(raw_llm_response)
            
            # Check timeout after parsing
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                error_msg = f"Planning timeout after parsing - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s"
                logger.error(f"[TIMEOUT-{agent_execution_id}] {error_msg}")
                result_state.add_message(f"[TIMEOUT-ERROR] {error_msg}")
                
                raise PlanningTimeoutError(
                    message="Planning cycle timed out during response parsing",
                    timeout_seconds=self.timeout_seconds,
                    elapsed_seconds=elapsed_time,
                    correlation_id=agent_execution_id,
                    request_details={
                        "agent_execution_id": agent_execution_id,
                        "goal": goal,
                        "phase": "post_parsing",
                        "parsed_tool_calls": len(tool_call_list)
                    }
                )
            
            # STRICT VALIDATION: Ensure we got valid tool calls
            if not tool_call_list:
                error_msg = "LLM response produced no valid tool calls"
                logger.error(f"[LLM-ERROR-{agent_execution_id}] {error_msg}")
                logger.error(f"[LLM-ERROR-{agent_execution_id}] Raw response: {raw_llm_response}")
                result_state.add_message(f"[LLM-ERROR] {error_msg}")
                
                raise LLMUnavailableError(
                    message="LLM response parsing failed - no valid tool calls extracted",
                    request_details={
                        "agent_execution_id": agent_execution_id,
                        "goal": goal,
                        "raw_response": raw_llm_response,
                        "parsed_tool_calls": tool_call_list
                    },
                    correlation_id=agent_execution_id
                )
            
            # SUCCESS: Add final task_complete() call
            tool_call_list.append("task_complete()")
            
            # SUCCESS: Store the LLM-generated plan in state
            result_state.data['plan'] = tool_call_list
            
            # BULLETPROOF LOGGING: Log successful LLM-powered operation with timing
            success_timestamp = datetime.now().isoformat()
            final_elapsed_time = time.time() - start_time
            logger.info(f"[PLANNER-{agent_execution_id}] LLM decomposition completed successfully at {success_timestamp} (elapsed: {final_elapsed_time:.2f}s)")
            result_state.add_message(f"[LLM-POWERED] Successfully received decomposition from LLM (execution: {agent_execution_id}, elapsed: {final_elapsed_time:.2f}s)")
            result_state.add_message(f"[LLM-POWERED] Created tool-based plan with {len(tool_call_list)} steps")
            
            # Log the LLM-generated plan for debugging
            logger.info(f"[PLANNER-{agent_execution_id}] LLM-generated plan with {len(tool_call_list)} steps:")
            for i, tool_call in enumerate(tool_call_list, 1):
                result_state.add_message(f"  [LLM-POWERED] Step {i}: {tool_call}")
                logger.info(f"[PLANNER-{agent_execution_id}] [LLM-POWERED] Step {i}: {tool_call}")
            
            logger.info(f"[PLANNER-{agent_execution_id}] STRICT LLM-ONLY decomposition completed successfully")
            
        except (LLMUnavailableError, PlanningTimeoutError):
            # Re-raise our own errors without modification
            raise
            
        except Exception as e:
            # BULLETPROOF LOGGING: Log complete error details and convert to LLMUnavailableError
            error_timestamp = datetime.now().isoformat()
            logger.error(f"[LLM-ERROR-{agent_execution_id}] LLM decomposition failed at {error_timestamp}")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] Error type: {type(e).__name__}")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] Error message: {str(e)}")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] FULL ERROR TRACEBACK BEGINS:")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] {traceback.format_exc()}")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] FULL ERROR TRACEBACK ENDS")
            logger.error(f"[LLM-ERROR-{agent_execution_id}] STRICT MODE: NO FALLBACKS - FAILING IMMEDIATELY")
            
            result_state.add_message(f"[LLM-ERROR] Task decomposition failed: {str(e)} (execution: {agent_execution_id})")
            result_state.add_message(f"[LLM-ERROR] STRICT MODE: No fallbacks available - operation failed")
            
            # Convert all other exceptions to LLMUnavailableError
            raise LLMUnavailableError(
                message=f"LLM decomposition failed: {str(e)}",
                original_error=e,
                request_details={
                    "agent_execution_id": agent_execution_id,
                    "goal": goal,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": error_timestamp
                },
                correlation_id=agent_execution_id
            ) from e
        
        result_state.add_message(f"Planner agent {self.name} completed STRICT LLM-ONLY task decomposition")
        
        return result_state
    
    def _generate_decomposition_prompt(self, goal: str) -> str:
        """Generate a structured prompt with harsh rules to combat lazy optimization.
        
        Creates a comprehensive prompt that FORCES the LLM to break down
        the complex goal into MANDATORY sequential steps. The prompt uses
        imperative language and explicit prohibitions to prevent optimization.
        
        Args:
            goal: The high-level goal to decompose.
            
        Returns:
            Formatted prompt string for MANDATORY sequential decomposition.
        """
        # Build the available tools section
        tools_section = "--- AVAILABLE TOOLS ---\n"
        
        if not self.available_tools:
            tools_section += "No tools available.\n"
        else:
            for tool in self.available_tools:
                description = getattr(tool, 'description', None) or "[Description not provided]"
                tools_section += f"- {tool.name}: {description}\n"
        
        prompt = f"""--- CRITICAL: MANDATORY SEQUENTIAL TASK DECOMPOSITION ---

ATTENTION: You are in STRICT SEQUENTIAL MODE. This is NOT a suggestion - it is a MANDATORY REQUIREMENT.

--- GOAL TO ACHIEVE ---
{goal}

{tools_section}
--- CRITICAL FAILURE WARNING ---
ATTENTION: The goal MUST be executed in the EXACT sequence specified. Violating the sequence of steps is a CRITICAL FAILURE.

YOU ARE ABSOLUTELY FORBIDDEN FROM:
- Optimizing or combining steps
- Jumping directly to final results
- Writing final content in initial steps
- Skipping intermediate states
- Creating "smart shortcuts"

--- MANDATORY SEQUENTIAL EXECUTION RULES ---

RULE 1: SEQUENCE VIOLATION IS PROHIBITED
- Each step MUST create a distinct intermediate state
- Each step MUST depend on the previous step's output
- You CANNOT optimize the sequence - this is STRICTLY PROHIBITED

RULE 2: STATE VALIDATION REQUIRED
- Before each step, analyze what state was created by the previous step
- Each step must operate on the intermediate state, not jump to final result
- You MUST validate that each step produces a different state

RULE 3: INTERMEDIATE CONTENT CREATION MANDATORY
- If the goal involves content modification, you MUST create initial content first
- NEVER write final content directly - this is FORBIDDEN
- Create initial state, THEN modify it in subsequent steps

RULE 4: STEP DEPENDENCY ANALYSIS REQUIRED
- Each step must explicitly depend on outputs from previous steps
- You CANNOT skip dependencies - this is a CRITICAL FAILURE
- Think: "What does step N need from step N-1?"

RULE 5: PYTHON EXECUTION MUST USE VIRTUAL ENVIRONMENT
- ALL Python commands MUST use virtual environment activation
- Format: execute_bash_command(command='source venv/bin/activate && python script.py')
- NEVER execute Python directly without venv activation

--- MANDATORY EXECUTION PATTERN ---

For content creation + modification goals, you MUST follow this pattern:
STEP 1: Create file with INITIAL content (not final content)
STEP 2: Modify the initial content using search/replace operations
STEP 3: Execute/verify the modified content

CRITICAL: Step 1 CANNOT contain final content - this is FORBIDDEN

--- VALIDATION CHECKLIST ---
Before providing your answer, verify:
□ Each step creates a different intermediate state
□ No step jumps directly to the final result
□ Initial content is NOT the final content
□ Each step has explicit dependencies on previous steps
□ Python commands use venv activation
□ No optimization or "smart shortcuts" were applied

--- OUTPUT FORMAT ---
Provide the tool call sequence as a numbered list:
1. first_tool(parameters_for_initial_state)
2. second_tool(parameters_that_modify_previous_state)
3. third_tool(parameters_that_use_modified_state)

--- MANDATORY EXAMPLES ---

CORRECT Example - Content Creation + Modification:
Goal: Create hello_oniks.py that prints "K Prize Mission Ready!" and execute it
CORRECT sequence (MANDATORY):
1. write_file(file_path='hello_oniks.py', content='print("Hello ONIKS")')  # Initial content
2. file_search_replace(file_path='hello_oniks.py', search_pattern='"Hello ONIKS"', replace_with='"K Prize Mission Ready!"')  # Modify content
3. execute_bash_command(command='source venv/bin/activate && python hello_oniks.py')  # Execute modified file

FORBIDDEN Example - Direct Optimization:
1. write_file(file_path='hello_oniks.py', content='print("K Prize Mission Ready!")')  # FORBIDDEN - skips intermediate state
2. execute_bash_command(command='source venv/bin/activate && python hello_oniks.py')

WHY THE CORRECT EXAMPLE IS MANDATORY:
- Step 1 creates intermediate state (file with initial content)
- Step 2 depends on Step 1's output and modifies it
- Step 3 depends on Step 2's modification and executes result
- Each step produces a DIFFERENT state

CORRECT Example - File Processing:
Goal: Create config.json with initial values, then update port to 8080, then validate
CORRECT sequence (MANDATORY):
1. write_file(file_path='config.json', content='{{"port": 3000, "host": "localhost"}}')  # Initial state
2. file_search_replace(file_path='config.json', search_pattern='"port": 3000', replace_with='"port": 8080')  # Modified state
3. execute_bash_command(command='cat config.json')  # Validate final state

--- YOUR MANDATORY TOOL SEQUENCE ---
CRITICAL REMINDER: You MUST create intermediate states. You CANNOT optimize. Follow the sequence exactly.

Create your MANDATORY sequential tool calls to achieve the goal:"""

        return prompt
    
    def _parse_decomposition_response_robust(self, response: str) -> List[str]:
        """Parse LLM response using robust regex-based extraction to cut through fluff.
        
        This method uses multiple parsing strategies to extract tool calls from ANY text
        environment, cutting through "polite" fluff and getting to the essence. It can
        handle various formats and finds numbered lists with tool calls regardless of
        surrounding text.
        
        Parsing strategies (in order):
        1. Numbered list with function calls (primary)
        2. Bullet/dash list with function calls  
        3. Standalone function calls anywhere in text
        4. Multi-line function calls with comments
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            List of tool call strings extracted from the response.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid decomposition response")
            return []
        
        tool_calls = []
        
        # Strategy 1: Process all lines looking for any kind of list items with function calls
        # This unified approach handles numbered, bullet, and asterisk lists all at once
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for numbered items: "1. function_name(...)"
            numbered_match = re.match(r'^\s*(\d+)\.\s*(.+?)(?:\s*#.*)?$', line)
            if numbered_match:
                num, potential_call = numbered_match.groups()
                potential_call = potential_call.strip()
                
                extracted_call = self._extract_function_call_from_line(potential_call)
                if extracted_call and self._is_valid_tool_call(extracted_call):
                    tool_calls.append(extracted_call)
                    logger.debug(f"Extracted numbered tool call {num}: {extracted_call}")
                continue
            
            # Check for bullet items: "- function_name(...)" or "* function_name(...)"
            bullet_match = re.match(r'^\s*[-*]\s*(.+?)(?:\s*#.*)?$', line)
            if bullet_match:
                potential_call = bullet_match.group(1).strip()
                
                extracted_call = self._extract_function_call_from_line(potential_call)
                if extracted_call and self._is_valid_tool_call(extracted_call):
                    tool_calls.append(extracted_call)
                    logger.debug(f"Extracted bullet tool call: {extracted_call}")
                continue
            
            # If it's not a list item but contains a function call, extract it
            extracted_call = self._extract_function_call_from_line(line)
            if extracted_call and self._is_valid_tool_call(extracted_call):
                tool_calls.append(extracted_call)
                logger.debug(f"Extracted standalone tool call: {extracted_call}")
        
        if tool_calls:
            logger.info(f"Found {len(tool_calls)} tool calls using unified approach")
        
        # Strategy 2: If no matches yet, try more aggressive pattern matching for any function calls
        if not tool_calls:
            # More aggressive pattern to find function calls anywhere in the text
            function_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\))'
            function_matches = re.findall(function_pattern, response)
            
            if function_matches:
                logger.info(f"Found {len(function_matches)} potential function calls in text")
                for tool_call in function_matches:
                    clean_tool_call = tool_call.strip()
                    if self._is_valid_tool_call(clean_tool_call):
                        tool_calls.append(clean_tool_call)
                        logger.debug(f"Extracted function call: {clean_tool_call}")
        
        # Strategy 3: Multi-line function calls (for complex arguments)
        if not tool_calls:
            # Handle cases where function calls span multiple lines
            multiline_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*(?:\n[^)]*)*\))'
            multiline_matches = re.findall(multiline_pattern, response, re.DOTALL)
            
            if multiline_matches:
                logger.info(f"Found {len(multiline_matches)} potential multi-line function calls")
                for tool_call in multiline_matches:
                    # Clean up multi-line call (remove extra whitespace/newlines)
                    clean_tool_call = re.sub(r'\s+', ' ', tool_call.strip())
                    if self._is_valid_tool_call(clean_tool_call):
                        tool_calls.append(clean_tool_call)
                        logger.debug(f"Extracted multi-line tool call: {clean_tool_call}")
        
        # Remove duplicates while preserving order
        unique_tool_calls = []
        seen = set()
        for tool_call in tool_calls:
            if tool_call not in seen:
                unique_tool_calls.append(tool_call)
                seen.add(tool_call)
        
        logger.info(f"Robust parser extracted {len(unique_tool_calls)} unique tool calls from response")
        
        if not unique_tool_calls:
            logger.warning("Robust parser could not extract any valid tool calls")
            logger.debug(f"Raw response content: {response[:500]}...")  # Log first 500 chars for debugging
        
        return unique_tool_calls
    
    def _extract_function_call_from_line(self, line: str) -> Optional[str]:
        """Extract a function call from a line, handling balanced parentheses and quotes.
        
        This method carefully parses a line to extract a complete function call,
        properly handling nested quotes, escaped characters, and balanced parentheses.
        
        Args:
            line: The line of text to extract a function call from.
            
        Returns:
            The extracted function call string, or None if no valid function call found.
        """
        line = line.strip()
        
        # Find the start of a function call
        func_start_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
        if not func_start_match:
            return None
        
        func_name = func_start_match.group(1)
        start_pos = func_start_match.start()
        paren_start = func_start_match.end() - 1  # Position of opening parenthesis
        
        # Find the matching closing parenthesis
        paren_count = 1
        pos = paren_start + 1
        in_single_quote = False
        in_double_quote = False
        escape_next = False
        
        while pos < len(line) and paren_count > 0:
            char = line[pos]
            
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            pos += 1
        
        if paren_count == 0:
            # Found matching closing parenthesis
            function_call = line[start_pos:pos]
            return function_call.strip()
        
        return None
    
    def _parse_decomposition_response(self, response: str) -> List[str]:
        """Legacy parser method - kept for backward compatibility.
        
        This method is deprecated. Use _parse_decomposition_response_robust instead.
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            List of tool call strings extracted from the response.
        """
        logger.warning("Using legacy parser - consider using robust parser instead")
        return self._parse_decomposition_response_robust(response)
    
    
    def _is_valid_tool_call(self, tool_call: str) -> bool:
        """Validate if a string represents a valid tool call format.
        
        In STRICT LLM-ONLY mode, this method validates that the LLM response
        contains properly formatted function calls. It does NOT validate tool
        availability - that happens at execution time. We trust the LLM to
        generate appropriate tool calls based on the prompt.
        
        Args:
            tool_call: String to validate as a tool call.
            
        Returns:
            True if the string is a valid function call format, False otherwise.
            
        Example:
            >>> agent = PlannerAgent("test", llm_client, [])
            >>> agent._is_valid_tool_call("write_file(file_path='test.txt', content='Hello')")
            True
            >>> agent._is_valid_tool_call("invalid_syntax(")
            False
            >>> agent._is_valid_tool_call("just some text")
            False
        """
        if not isinstance(tool_call, str) or not tool_call.strip():
            return False
        
        tool_call = tool_call.strip()
        
        # Check if it has basic function call structure: name(...)
        import re
        match = re.match(r'^(\w+)\s*\(.*\)$', tool_call)
        if not match:
            return False
        
        tool_name = match.group(1)
        
        # In STRICT LLM-ONLY mode, we accept any well-formed function call
        # The LLM is responsible for generating appropriate tool calls
        # Tool availability is checked at execution time, not parsing time
        
        # Always allow task_complete as it's a standard completion tool
        if tool_name == 'task_complete':
            return True
        
        # Allow any valid function call format - trust the LLM
        # This enables the LLM to suggest tools that might be available
        
        # Try to parse the arguments to ensure valid syntax
        try:
            # Extract arguments part
            args_match = re.match(r'^\w+\s*\((.*)\)$', tool_call)
            if not args_match:
                return False
            
            args_str = args_match.group(1).strip()
            
            # If no arguments, it's valid
            if not args_str:
                return True
            
            # Try to parse as function call arguments
            try:
                # Create a dummy function call and try to parse it
                dummy_call = f"dummy_func({args_str})"
                import ast
                parsed = ast.parse(dummy_call, mode='eval')
                
                # Check if it's a valid call expression
                if not isinstance(parsed.body, ast.Call):
                    return False
                
                return True
                
            except (SyntaxError, ValueError):
                # If AST parsing fails, try a simple regex check for key=value pairs
                # This handles cases like: tool_name(key1='value1', key2='value2')
                kv_pattern = r"^\s*\w+\s*=\s*['\"][^'\"]*['\"](\s*,\s*\w+\s*=\s*['\"][^'\"]*['\"])*\s*$"
                if re.match(kv_pattern, args_str):
                    return True
                
                return False
                
        except Exception as e:
            logger.warning(f"Error validating tool call syntax '{tool_call}': {e}")
            return False