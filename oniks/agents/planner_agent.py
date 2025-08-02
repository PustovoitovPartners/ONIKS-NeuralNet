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
import traceback
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from oniks.agents.base import BaseAgent
from oniks.core.exceptions import LLMUnavailableError

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
    
    def __init__(self, name: str, llm_client: "OllamaClient", available_tools: Optional[List["Tool"]] = None) -> None:
        """Initialize the PlannerAgent with LLM client and available tools.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            available_tools: List of Tool instances that can be used in plans.
                           If None, defaults to empty list.
            
        Raises:
            ValueError: If name is empty, None, or llm_client is None.
            TypeError: If available_tools is not a list.
        """
        super().__init__(name)
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        if available_tools is None:
            available_tools = []
        
        if not isinstance(available_tools, list):
            raise TypeError(f"Available tools must be a list, got {type(available_tools).__name__}")
        
        self.llm_client = llm_client
        self.available_tools = available_tools
    
    def execute(self, state: "State") -> "State":
        """Execute STRICT LLM-ONLY task decomposition logic to create an executable plan.
        
        CRITICAL: This method operates in STRICT LLM-ONLY mode with NO FALLBACKS.
        
        The method:
        1. Validates that a goal exists (fails immediately if not)
        2. Generates a structured prompt for LLM-powered tool-based decomposition
        3. Invokes the LLM with strict validation (HTTP 200 + non-empty content)
        4. Parses and validates the LLM response
        5. Creates tool call sequence ONLY from successful LLM response
        6. Tags operation as [LLM-POWERED] ONLY on complete success
        
        FAILURE CONDITIONS (all throw LLMUnavailableError):
        - No goal in state
        - LLM connection failure
        - LLM HTTP error response
        - Empty or invalid LLM response content
        - Unparseable LLM response
        
        Args:
            state: The current state containing the goal to decompose.
            
        Returns:
            The modified state with the LLM-generated plan in state.data['plan'].
            
        Raises:
            LLMUnavailableError: If LLM is unavailable or returns invalid response.
        """
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Generate unique agent execution ID for correlation
        agent_execution_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Add message about planner execution
        result_state.add_message(f"Planner agent {self.name} starting STRICT LLM-ONLY task decomposition")
        
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
        
        # CRITICAL: Invoke LLM with strict validation - fail immediately on any error
        try:
            logger.info(f"[PLANNER-{agent_execution_id}] Calling LLM for task decomposition (STRICT MODE)")
            
            # Call LLM - this may throw OllamaConnectionError or other exceptions
            raw_llm_response = self.llm_client.invoke(decomposition_prompt)
            
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
            
            # STRICT VALIDATION: Parse the LLM response to extract tool call sequence
            tool_call_list = self._parse_decomposition_response(raw_llm_response)
            
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
            
            # BULLETPROOF LOGGING: Log successful LLM-powered operation
            success_timestamp = datetime.now().isoformat()
            logger.info(f"[PLANNER-{agent_execution_id}] LLM decomposition completed successfully at {success_timestamp}")
            result_state.add_message(f"[LLM-POWERED] Successfully received decomposition from LLM (execution: {agent_execution_id})")
            result_state.add_message(f"[LLM-POWERED] Created tool-based plan with {len(tool_call_list)} steps")
            
            # Log the LLM-generated plan for debugging
            logger.info(f"[PLANNER-{agent_execution_id}] LLM-generated plan with {len(tool_call_list)} steps:")
            for i, tool_call in enumerate(tool_call_list, 1):
                result_state.add_message(f"  [LLM-POWERED] Step {i}: {tool_call}")
                logger.info(f"[PLANNER-{agent_execution_id}] [LLM-POWERED] Step {i}: {tool_call}")
            
            logger.info(f"[PLANNER-{agent_execution_id}] STRICT LLM-ONLY decomposition completed successfully")
            
        except LLMUnavailableError:
            # Re-raise our own LLMUnavailableError without modification
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
1. write_file(file_path='config.json', content='{"port": 3000, "host": "localhost"}')  # Initial state
2. file_search_replace(file_path='config.json', search_pattern='"port": 3000', replace_with='"port": 8080')  # Modified state
3. execute_bash_command(command='cat config.json')  # Validate final state

--- YOUR MANDATORY TOOL SEQUENCE ---
CRITICAL REMINDER: You MUST create intermediate states. You CANNOT optimize. Follow the sequence exactly.

Create your MANDATORY sequential tool calls to achieve the goal:"""

        return prompt
    
    def _parse_decomposition_response(self, response: str) -> List[str]:
        """Parse LLM response to extract the list of tool calls.
        
        Extracts numbered tool calls from the LLM response and returns them as
        a clean list of function call strings. Handles various formatting variations
        that might appear in LLM responses.
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            List of tool call strings extracted from the response.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid decomposition response")
            return []
        
        tool_calls = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered list items (1., 2., etc.)
            import re
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                tool_call = match.group(1).strip()
                if self._is_valid_tool_call(tool_call):
                    tool_calls.append(tool_call)
                    continue
            
            # Look for dash/bullet list items
            if line.startswith('- ') or line.startswith('* '):
                tool_call = line[2:].strip()
                if self._is_valid_tool_call(tool_call):
                    tool_calls.append(tool_call)
                    continue
            
            # If line looks like a function call, include it
            if self._is_valid_tool_call(line):
                tool_calls.append(line)
        
        logger.info(f"Parsed {len(tool_calls)} tool calls from decomposition response")
        return tool_calls
    
    
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