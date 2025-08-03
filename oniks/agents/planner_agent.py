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
    
    def __init__(self, name: str, llm_client: "OllamaClient", available_tools: Optional[List["Tool"]] = None, timeout_seconds: float = 1200.0) -> None:
        """Initialize the PlannerAgent with LLM client and available tools.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            available_tools: List of Tool instances that can be used in plans.
                           If None, defaults to empty list.
            timeout_seconds: Maximum time allowed for planning cycle in seconds.
                           Defaults to 1200.0 seconds.
            
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
        
        # Check execution path to determine prompt type
        execution_path = result_state.data.get('execution_path', 'hierarchical')
        
        # Generate appropriate prompt based on execution path
        if execution_path == 'direct':
            decomposition_prompt = self._generate_optimized_prompt(goal)
            result_state.add_message("Using optimized prompt for simple task (direct path)")
        else:
            decomposition_prompt = self._generate_decomposition_prompt(goal)
            result_state.add_message("Using comprehensive prompt for complex task (hierarchical path)")
        
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
            # Always use the main model (llama3:8b) for complex planning
            raw_llm_response = self.llm_client.invoke(decomposition_prompt, model="llama3:8b")
            
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
- Format: execute_bash_command(command='python3 script.py')
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
3. execute_bash_command(command='python3 hello_oniks.py')  # Execute modified file

FORBIDDEN Example - Direct Optimization:
1. write_file(file_path='hello_oniks.py', content='print("K Prize Mission Ready!")')  # FORBIDDEN - skips intermediate state
2. execute_bash_command(command='python3 hello_oniks.py')

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
    
    def _generate_optimized_prompt(self, goal: str) -> str:
        """Generate a lightweight prompt for simple tasks without strict sequential rules.
        
        Creates a simplified prompt that allows direct optimization for simple tasks,
        removing the strict sequential requirements that are only necessary for
        complex multi-step workflows.
        
        Args:
            goal: The simple goal to decompose.
            
        Returns:
            Formatted prompt string for optimized simple task decomposition.
        """
        # Build the available tools section
        tools_section = "--- AVAILABLE TOOLS ---\n"
        
        if not self.available_tools:
            tools_section += "No tools available.\n"
        else:
            for tool in self.available_tools:
                description = getattr(tool, 'description', None) or "[Description not provided]"
                tools_section += f"- {tool.name}: {description}\n"
        
        prompt = f"""--- SIMPLE TASK OPTIMIZATION ---

GOAL TO ACHIEVE:
{goal}

{tools_section}
--- OPTIMIZATION INSTRUCTIONS ---

This is a SIMPLE TASK that can be completed efficiently. You are allowed to:
- Use direct approaches to achieve the goal
- Combine related operations where appropriate
- Focus on the most efficient path to completion
- Skip unnecessary intermediate states for simple operations

GUIDELINES:
- Create the final desired content directly when possible
- Use the minimum number of steps needed
- Prioritize efficiency and directness
- Only use verification steps if explicitly requested

--- EXECUTION RULES ---

RULE 1: DIRECT CONTENT CREATION ALLOWED
- For simple file creation, write the final content directly
- No need for initial content + modification unless specifically required
- Aim for the shortest path to the goal

RULE 2: PYTHON EXECUTION MUST USE VIRTUAL ENVIRONMENT
- ALL Python commands MUST use virtual environment activation
- Format: execute_bash_command(command='python3 script.py')
- NEVER execute Python directly without venv activation

--- OUTPUT FORMAT ---
Provide the tool call sequence as a numbered list:
1. tool_name(parameters_for_direct_completion)
2. additional_tool(parameters_if_needed)

--- OPTIMIZED EXAMPLES ---

EFFICIENT Example - Direct Content Creation:
Goal: Create hello_world.py that prints "Hello World" and execute it
OPTIMIZED sequence:
1. write_file(file_path='hello_world.py', content='print("Hello World")')  # Direct final content
2. execute_bash_command(command='python3 hello_world.py')  # Execute directly

EFFICIENT Example - Simple File Operation:
Goal: Create config.json with port 8080 and host localhost
OPTIMIZED sequence:
1. write_file(file_path='config.json', content='{{"port": 8080, "host": "localhost"}}')  # Direct final content

--- YOUR OPTIMIZED TOOL SEQUENCE ---
Create your EFFICIENT tool calls to achieve the goal directly:"""

        return prompt
    
    def _parse_decomposition_response_robust(self, response: str) -> List[str]:
        """Parse LLM response using fast, optimized step-by-step approach.
        
        This method uses a simple, fast two-step approach:
        1. Find lines starting with numbers (1., 2., 3., etc.)
        2. Extract tool calls from those specific lines only
        
        This approach is thousands of times faster than complex regex patterns
        and avoids catastrophic backtracking that can cause 800+ second delays.
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            List of tool call strings extracted from the response.
        """
        try:
            parse_start_time = time.time()
        except Exception:
            # Handle mocking issues in tests
            parse_start_time = 0.0
        
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid decomposition response")
            return []
        
        tool_calls = []
        
        # STEP 1: Find lines starting with numbers or bullets (fast string operations)
        lines = response.split('\n')
        numbered_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            # Check for numbered items: "1. function_name(...)"
            if len(stripped_line) > 2 and stripped_line[0].isdigit():
                # Find the dot after the number
                dot_pos = -1
                for i in range(1, min(4, len(stripped_line))):  # Check first 4 chars for number
                    if stripped_line[i] == '.':
                        dot_pos = i
                        break
                    elif not stripped_line[i].isdigit():
                        break
                
                if dot_pos > 0:
                    # Extract content after number and dot
                    content = stripped_line[dot_pos + 1:].strip()
                    if content:
                        numbered_lines.append(content)
                        continue
            
            # Check for bullet items: "- function_name(...)" or "* function_name(...)"
            if len(stripped_line) > 2 and (stripped_line[0] == '-' or stripped_line[0] == '*'):
                if stripped_line[1] == ' ':
                    content = stripped_line[2:].strip()
                    if content:
                        numbered_lines.append(content)
        
        # STEP 2: Extract tool calls from numbered lines only (fast string operations)
        for content in numbered_lines:
            # Look for pattern: word_name(  
            paren_pos = content.find('(')
            if paren_pos == -1:
                continue
                
            # Extract potential function name
            func_name_part = content[:paren_pos].strip()
            if not func_name_part or not func_name_part.replace('_', '').replace('-', '').isalnum():
                continue
                
            # Find matching closing parenthesis using simple counting
            paren_count = 0
            end_pos = -1
            in_quote = False
            quote_char = None
            
            for i in range(paren_pos, len(content)):
                char = content[i]
                
                # Handle quotes (simple approach)
                if not in_quote and (char == '"' or char == "'"):
                    in_quote = True
                    quote_char = char
                elif in_quote and char == quote_char:
                    # Check if it's escaped
                    if i == 0 or content[i-1] != '\\':
                        in_quote = False
                        quote_char = None
                
                # Count parentheses only when not in quotes
                if not in_quote:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end_pos = i
                            break
            
            if end_pos > paren_pos:
                # Extract complete function call
                tool_call = content[:end_pos + 1].strip()
                
                # Basic validation: has function name and parentheses
                if tool_call and '(' in tool_call and tool_call.endswith(')'):
                    tool_calls.append(tool_call)
                    logger.debug(f"Extracted tool call: {tool_call}")
        
        # FALLBACK: If no structured lists found, look for function calls in text
        if not tool_calls:
            # Simple scan for function_name( patterns in the entire text
            words = response.split()
            for word in words:
                # Look for function call pattern
                if '(' in word:
                    # Try to extract from this word and surrounding context
                    start_idx = response.find(word)
                    if start_idx != -1:
                        # Look for complete function call starting from this position
                        extracted = self._extract_function_call_from_line(response[start_idx:start_idx + 200])
                        if extracted:
                            tool_calls.append(extracted)
        
        # Remove duplicates while preserving order
        unique_tool_calls = []
        seen = set()
        for tool_call in tool_calls:
            if tool_call not in seen:
                unique_tool_calls.append(tool_call)
                seen.add(tool_call)
        
        try:
            parse_end_time = time.time()
            parse_duration = parse_end_time - parse_start_time
            logger.info(f"Fast parser extracted {len(unique_tool_calls)} unique tool calls in {parse_duration:.4f}s")
        except Exception:
            # Handle mocking issues in tests
            logger.info(f"Fast parser extracted {len(unique_tool_calls)} unique tool calls")
        
        if not unique_tool_calls:
            logger.warning("Fast parser could not extract any valid tool calls")
            logger.debug(f"Raw response content: {response[:200]}...")  # Log first 200 chars for debugging
        
        return unique_tool_calls
    
    def _extract_function_call_from_line(self, line: str) -> Optional[str]:
        """Extract a function call from a line using fast string operations.
        
        This method is kept for backward compatibility with existing tests.
        It uses the same fast logic as the optimized parser.
        
        Args:
            line: The line of text to extract a function call from.
            
        Returns:
            The extracted function call string, or None if no valid function call found.
        """
        if not line or not isinstance(line, str):
            return None
            
        line = line.strip()
        
        # Look for pattern: word_name(  
        paren_pos = line.find('(')
        if paren_pos == -1:
            return None
            
        # Extract potential function name
        func_name_part = line[:paren_pos].strip()
        if not func_name_part or not func_name_part.replace('_', '').replace('-', '').isalnum():
            return None
            
        # Find matching closing parenthesis using simple counting
        paren_count = 0
        end_pos = -1
        in_quote = False
        quote_char = None
        
        for i in range(paren_pos, len(line)):
            char = line[i]
            
            # Handle quotes (simple approach)
            if not in_quote and (char == '"' or char == "'"):
                in_quote = True
                quote_char = char
            elif in_quote and char == quote_char:
                # Check if it's escaped
                if i == 0 or line[i-1] != '\\':
                    in_quote = False
                    quote_char = None
            
            # Count parentheses only when not in quotes
            if not in_quote:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos = i
                        break
        
        if end_pos > paren_pos:
            # Extract complete function call
            tool_call = line[:end_pos + 1].strip()
            
            # Basic validation: has function name and parentheses
            if tool_call and '(' in tool_call and tool_call.endswith(')'):
                return tool_call
        
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
        """Fast validation of tool call format using simple string operations.
        
        This optimized method uses simple string operations instead of regex
        and AST parsing to achieve sub-second performance.
        
        Args:
            tool_call: String to validate as a tool call.
            
        Returns:
            True if the string is a valid function call format, False otherwise.
        """
        if not isinstance(tool_call, str) or not tool_call.strip():
            return False
        
        tool_call = tool_call.strip()
        
        # Fast check: must contain parentheses
        if '(' not in tool_call or not tool_call.endswith(')'):
            return False
        
        # Find the opening parenthesis
        paren_pos = tool_call.find('(')
        if paren_pos == 0:
            return False  # Can't start with parenthesis
        
        # Extract function name (everything before opening parenthesis)
        func_name = tool_call[:paren_pos].strip()
        
        # Fast validation: function name should be alphanumeric with underscores
        if not func_name or not func_name.replace('_', '').replace('-', '').isalnum():
            return False
        
        # Always allow task_complete as it's a standard completion tool
        if func_name == 'task_complete':
            return True
        
        # Basic check: should have balanced parentheses (simple count)
        open_count = tool_call.count('(')
        close_count = tool_call.count(')')
        
        if open_count != close_count:
            return False
        
        # If we get here, it's likely a valid function call format
        return True