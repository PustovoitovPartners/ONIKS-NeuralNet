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
    """Dual-circuit fast LLM-powered agent that classifies task complexity for intelligent routing.
    
    The RouterAgent implements a dual-circuit decision-making system for optimal performance:
    - Circuit 1: Fast classification using lightweight LLM (phi3:mini) with 15-second timeout
    - Circuit 2: Keyword-based fallback classification when LLM fails
    
    This approach provides lightning-fast task classification while maintaining reliability
    through graceful degradation to keyword-based analysis.
    
    Key Features:
    - Ultra-fast classification with 15-second timeout (aggressive)
    - Lightweight phi3:mini model for speed
    - Minimal 50-word prompts for fast processing
    - Multi-layer fallback system (LLM → keyword → hierarchical)
    - Clear execution path routing for graph optimization
    
    Classification Logic:
    - Simple tasks: Single-step operations, direct file operations, simple commands
    - Complex tasks: Multi-step workflows, conditional logic, interdependent operations
    
    Attributes:
        llm_client: OllamaClient instance for LLM interactions.
        routing_model: Lightweight model for fast classification (default: "phi3:mini").
        main_model: Main model for complex operations (default: "llama3:8b").
        timeout_seconds: Maximum time for classification (default: 15 seconds).
    
    Example:
        >>> from oniks.llm.client import OllamaClient
        >>> llm_client = OllamaClient()
        >>> agent = RouterAgent("router", llm_client, routing_model="phi3:mini")
        >>> state = State()
        >>> state.data['goal'] = 'Create file hello.txt with Hello World!'
        >>> result_state = agent.execute(state)
        >>> print(result_state.data['execution_path'])
        direct
    """
    
    def __init__(self, name: str, llm_client: "OllamaClient", routing_model: str = "phi3:mini", main_model: str = "llama3:8b", timeout_seconds: float = 15.0) -> None:
        """Initialize the RouterAgent with LLM client and dual-circuit configuration.
        
        Args:
            name: Unique identifier for this agent.
            llm_client: OllamaClient instance for LLM interactions.
            routing_model: Lightweight model for fast classification (default: "phi3:mini").
            main_model: Main model for complex operations (default: "llama3:8b").
            timeout_seconds: Maximum time allowed for classification in seconds.
                           Defaults to 15.0 seconds for fast classification.
            
        Raises:
            ValueError: If name is empty, None, llm_client is None, or timeout_seconds is not positive.
        """
        super().__init__(name)
        
        if llm_client is None:
            raise ValueError("LLM client cannot be None")
        
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError(f"Timeout seconds must be a positive number, got {timeout_seconds}")
        
        self.llm_client = llm_client
        self.routing_model = routing_model
        self.main_model = main_model
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
            
            logger.info(f"[ROUTER-{agent_execution_id}] Calling LLM for fast classification using {self.routing_model}")
            
            # Make fast LLM call for classification using routing model
            try:
                raw_llm_response = self.llm_client.invoke(classification_prompt, model=self.routing_model)
            except Exception as routing_error:
                # Fallback to keyword-based classification if routing model fails
                logger.warning(f"[ROUTER-{agent_execution_id}] Routing model {self.routing_model} failed: {routing_error}")
                result_state.add_message(f"[ROUTER-FALLBACK] Routing model failed: {routing_error} - using keyword classification")
                execution_path = self._keyword_classification_fallback(goal)
                result_state.data['execution_path'] = execution_path
                result_state.add_message(f"[ROUTER-FALLBACK] Keyword classification result: {execution_path}")
                logger.info(f"[ROUTER-{agent_execution_id}] Keyword fallback classified as: {execution_path}")
                return result_state
            
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
        """Generate an ultra-lightweight prompt for fast classification with phi3:mini.
        
        Creates a minimal prompt optimized for speed with the lightweight routing model.
        Uses only essential information and expects a single word response.
        
        Args:
            goal: The user goal to classify.
            
        Returns:
            Formatted prompt string for ultra-fast classification.
        """
        prompt = f"""Task: {goal}

SIMPLE = single file operation, one step
COMPLEX = multiple steps, dependencies, "then"

Examples:
SIMPLE: "Create hello.txt with content X"
COMPLEX: "Create file, then modify it, then execute"

Response (one word): SIMPLE or COMPLEX

Classification:"""

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
    
    def _keyword_classification_fallback(self, goal: str) -> str:
        """Fallback keyword-based classification when routing model is unavailable.
        
        Uses simple keyword analysis to classify tasks when the lightweight LLM
        routing model fails or is unavailable. Always defaults to 'hierarchical'
        for safety when in doubt.
        
        Args:
            goal: The user goal to classify.
            
        Returns:
            'direct' for simple tasks, 'hierarchical' for complex tasks or unclear cases.
        """
        if not isinstance(goal, str):
            logger.warning("Invalid goal type for keyword classification")
            return 'hierarchical'
        
        goal_lower = goal.lower().strip()
        
        # Keywords that indicate complex tasks (multi-step, conditional)
        complex_keywords = [
            'then', 'after', 'next', 'first', 'second', 'third',
            'before', 'once', 'when', 'if', 'configure', 'setup',
            'modify', 'update', 'change', 'process', 'extract',
            'based on', 'depending on', 'multiple', 'several'
        ]
        
        # Keywords that indicate simple tasks (single operations)
        simple_keywords = [
            'create a file', 'write file', 'read file', 'execute',
            'run command', 'display', 'show', 'list', 'print'
        ]
        
        # Check for complex keywords first (safety-first approach)
        for keyword in complex_keywords:
            if keyword in goal_lower:
                logger.debug(f"Keyword classification: COMPLEX (found '{keyword}')")
                return 'hierarchical'
        
        # Check for simple keywords
        for keyword in simple_keywords:
            if keyword in goal_lower:
                logger.debug(f"Keyword classification: SIMPLE (found '{keyword}')")
                return 'direct'
        
        # Default to hierarchical for safety
        logger.debug("Keyword classification: COMPLEX (no clear indicators, defaulting to safe option)")
        return 'hierarchical'