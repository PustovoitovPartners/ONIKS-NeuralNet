"""Router agent implementation for the ONIKS NeuralNet framework.

This module provides the RouterAgent class, which serves as the task complexity
classification layer. It makes fast LLM queries to classify tasks as "simple" 
or "complex" and routes them to appropriate execution paths.

The RouterAgent operates with aggressive timeouts and graceful fallbacks to ensure
system reliability while providing significant performance improvements for simple tasks.
"""

import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from oniks.agents.base import BaseAgent
from oniks.core.exceptions import LLMUnavailableError

if TYPE_CHECKING:
    from oniks.core.state import State
    from oniks.llm.client import OllamaClient

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Fast LLM-powered agent that classifies task complexity for intelligent routing.
    
    The RouterAgent makes one fast LLM query to classify tasks as "simple" or "complex"
    and adds the appropriate execution_path to the state. This enables the system to
    optimize simple tasks for speed while maintaining quality for complex workflows.
    
    Key Features:
    - Fast classification with 30-second timeout
    - Lightweight prompts focused only on classification
    - Graceful degradation (defaults to 'hierarchical' on failure)
    - Clear execution path routing for graph optimization
    
    Classification Logic:
    - Simple tasks: Single-step operations, direct file operations, simple commands
    - Complex tasks: Multi-step workflows, conditional logic, interdependent operations
    
    Attributes:
        llm_client: OllamaClient instance for LLM interactions.
        timeout_seconds: Maximum time for classification (default: 30 seconds).
    
    Example:
        >>> from oniks.llm.client import OllamaClient
        >>> llm_client = OllamaClient()
        >>> agent = RouterAgent("router", llm_client)
        >>> state = State()
        >>> state.data['goal'] = 'Create file hello.txt with Hello World!'
        >>> result_state = agent.execute(state)
        >>> print(result_state.data['execution_path'])
        direct
    """
    
    def __init__(self, name: str, llm_client: "OllamaClient", timeout_seconds: float = 30.0) -> None:
        """Initialize the RouterAgent with LLM client and timeout.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            timeout_seconds: Maximum time allowed for classification in seconds.
                           Defaults to 30.0 seconds for fast routing.
            
        Raises:
            ValueError: If name is empty, None, llm_client is None, or timeout_seconds is not positive.
        """
        super().__init__(name)
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError(f"Timeout seconds must be a positive number, got {timeout_seconds}")
        
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds
    
    def execute(self, state: "State") -> "State":
        """Execute fast task complexity classification logic.
        
        This method performs rapid classification of the task goal to determine
        the appropriate execution path:
        
        1. Validates that a goal exists in the state
        2. Generates a lightweight classification prompt
        3. Makes a fast LLM query with aggressive timeout
        4. Parses classification result from LLM response
        5. Adds execution_path to state ('direct' or 'hierarchical')
        6. Falls back gracefully to 'hierarchical' on any failure
        
        Classification Results:
        - 'direct': Simple tasks that can bypass the PlannerAgent
        - 'hierarchical': Complex tasks that require full planning workflow
        
        Args:
            state: The current state containing the goal to classify.
            
        Returns:
            The modified state with execution_path added to state.data.
        """
        # Record start time for timeout enforcement
        start_time = time.time()
        
        # Create a copy of the state to avoid modifying the original
        result_state = state.model_copy(deep=True)
        
        # Generate unique agent execution ID for correlation
        agent_execution_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Add message about router execution with timeout info
        result_state.add_message(f"Router agent {self.name} starting fast task classification (timeout: {self.timeout_seconds}s)")
        
        # Extract and validate the goal
        goal = result_state.data.get('goal', '').strip()
        
        if not goal:
            # No goal found - default to hierarchical path for safety
            result_state.data['execution_path'] = 'hierarchical'
            result_state.add_message("[ROUTER-FALLBACK] No goal found - defaulting to hierarchical path")
            logger.warning(f"[ROUTER-{agent_execution_id}] No goal found, defaulting to hierarchical path")
            return result_state
        
        # Log router operation start
        logger.info(f"[ROUTER-{agent_execution_id}] Starting fast task classification at {timestamp}")
        logger.info(f"[ROUTER-{agent_execution_id}] Goal to classify: {goal}")
        logger.info(f"[ROUTER-{agent_execution_id}] Fast timeout: {self.timeout_seconds}s")
        
        # Generate lightweight classification prompt
        classification_prompt = self._generate_classification_prompt(goal)
        result_state.data['classification_prompt'] = classification_prompt
        
        result_state.add_message("Generated fast classification prompt for LLM")
        
        # Attempt fast LLM classification with timeout and graceful fallback
        try:
            # Check timeout before LLM call
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                result_state.data['execution_path'] = 'hierarchical'
                result_state.add_message(f"[ROUTER-TIMEOUT] Classification timeout before LLM call - defaulting to hierarchical")
                logger.warning(f"[ROUTER-{agent_execution_id}] Timeout before LLM call - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s")
                return result_state
            
            logger.info(f"[ROUTER-{agent_execution_id}] Calling LLM for fast classification")
            
            # Make fast LLM call for classification
            raw_llm_response = self.llm_client.invoke(classification_prompt)
            
            # Check timeout after LLM call
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                result_state.data['execution_path'] = 'hierarchical'
                result_state.add_message(f"[ROUTER-TIMEOUT] Classification timeout after LLM call - defaulting to hierarchical")
                logger.warning(f"[ROUTER-{agent_execution_id}] Timeout after LLM call - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s")
                return result_state
            
            # Validate LLM response
            if not raw_llm_response or not isinstance(raw_llm_response, str) or not raw_llm_response.strip():
                result_state.data['execution_path'] = 'hierarchical'
                result_state.add_message("[ROUTER-FALLBACK] Invalid LLM response - defaulting to hierarchical")
                logger.warning(f"[ROUTER-{agent_execution_id}] Invalid LLM response, defaulting to hierarchical")
                return result_state
            
            result_state.data['classification_response'] = raw_llm_response
            
            # Parse classification result from LLM response
            execution_path = self._parse_classification_response(raw_llm_response)
            
            # Final timeout check after parsing
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                result_state.data['execution_path'] = 'hierarchical'
                result_state.add_message(f"[ROUTER-TIMEOUT] Classification timeout after parsing - defaulting to hierarchical")
                logger.warning(f"[ROUTER-{agent_execution_id}] Timeout after parsing - elapsed {elapsed_time:.2f}s >= {self.timeout_seconds}s")
                return result_state
            
            # Store the execution path
            result_state.data['execution_path'] = execution_path
            
            # Log successful classification
            success_timestamp = datetime.now().isoformat()
            final_elapsed_time = time.time() - start_time
            logger.info(f"[ROUTER-{agent_execution_id}] Classification completed at {success_timestamp} (elapsed: {final_elapsed_time:.2f}s)")
            result_state.add_message(f"[ROUTER-SUCCESS] Task classified as '{execution_path}' path (execution: {agent_execution_id}, elapsed: {final_elapsed_time:.2f}s)")
            
        except Exception as e:
            # Graceful fallback on any error - default to hierarchical path
            result_state.data['execution_path'] = 'hierarchical'
            
            error_timestamp = datetime.now().isoformat()
            logger.warning(f"[ROUTER-{agent_execution_id}] Classification failed at {error_timestamp}")
            logger.warning(f"[ROUTER-{agent_execution_id}] Error type: {type(e).__name__}")
            logger.warning(f"[ROUTER-{agent_execution_id}] Error message: {str(e)}")
            logger.warning(f"[ROUTER-{agent_execution_id}] GRACEFUL FALLBACK: Defaulting to hierarchical path")
            
            result_state.add_message(f"[ROUTER-FALLBACK] Classification failed: {str(e)} - defaulting to hierarchical (execution: {agent_execution_id})")
        
        result_state.add_message(f"Router agent {self.name} completed task classification")
        
        return result_state
    
    def _generate_classification_prompt(self, goal: str) -> str:
        """Generate a lightweight prompt focused only on task complexity classification.
        
        Creates a minimal, fast prompt that asks the LLM to classify the task
        as either simple or complex without requiring detailed analysis.
        
        Args:
            goal: The user goal to classify.
            
        Returns:
            Formatted prompt string for fast classification.
        """
        prompt = f"""--- FAST TASK CLASSIFICATION ---

GOAL TO CLASSIFY:
{goal}

CLASSIFICATION RULES:

SIMPLE TASKS (use "SIMPLE"):
- Single file operations (create, read, write one file)
- Single command executions
- Direct operations with no dependencies
- Tasks that can be completed in 1-2 steps

Examples of SIMPLE:
- "Create a file named hello.txt with content X"
- "Execute command ls -la"
- "Read file data.txt"
- "Write Hello World to output.txt"

COMPLEX TASKS (use "COMPLEX"):
- Multi-step workflows
- Tasks with "then" or "after" conditions
- Multiple file operations
- Tasks requiring intermediate states
- Conditional logic or dependencies

Examples of COMPLEX:
- "Create file X, then modify it, then execute it"
- "First do A, then do B based on result"
- "Create multiple files and configure them"
- "Process file A, extract data, write to file B"

INSTRUCTION:
Respond with exactly one word: either "SIMPLE" or "COMPLEX"

CLASSIFICATION:"""

        return prompt
    
    def _parse_classification_response(self, response: str) -> str:
        """Parse LLM response to extract classification result.
        
        Extracts the classification from the LLM response, looking for
        "SIMPLE" or "COMPLEX" keywords. Defaults to 'hierarchical' if
        the response is unclear or contains neither keyword.
        
        Args:
            response: Raw response from the LLM.
            
        Returns:
            'direct' for simple tasks, 'hierarchical' for complex tasks or unclear responses.
        """
        if not isinstance(response, str):
            logger.warning(f"Invalid response type for classification: {type(response)}")
            return 'hierarchical'
        
        response_upper = response.strip().upper()
        
        # Look for explicit classification keywords
        if 'SIMPLE' in response_upper:
            logger.debug("Classification: SIMPLE -> direct path")
            return 'direct'
        elif 'COMPLEX' in response_upper:
            logger.debug("Classification: COMPLEX -> hierarchical path")
            return 'hierarchical'
        else:
            # Unclear response - default to hierarchical for safety
            logger.warning(f"Unclear classification response: {response[:100]}... - defaulting to hierarchical")
            return 'hierarchical'