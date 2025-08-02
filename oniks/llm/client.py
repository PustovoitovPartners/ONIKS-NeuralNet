"""Ollama client implementation for LLM integration.

This module provides the OllamaClient class for interacting with locally
running Ollama instances to enable real LLM-powered reasoning in agents.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Optional

import ollama
from ollama import ResponseError


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama service fails."""
    pass


class OllamaClient:
    """Client for interacting with locally running Ollama LLM service.
    
    This client provides a simple interface for sending prompts to Ollama
    and receiving text responses. It includes proper error handling for
    cases where the Ollama service is unavailable or encounters issues.
    
    Attributes:
        host: The host address for the Ollama service (default: localhost:11434).
        timeout: Request timeout in seconds (default: 30).
    
    Example:
        >>> client = OllamaClient()
        >>> response = client.invoke("What is the capital of France?", model="llama3:8b")
        >>> print(response)
        The capital of France is Paris.
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30) -> None:
        """Initialize the Ollama client.
        
        Args:
            host: The host address for the Ollama service.
            timeout: Request timeout in seconds.
        """
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host)
    
    def invoke(self, prompt: str, model: str = "llama3:8b") -> str:
        """Send a prompt to the Ollama service and return the response.
        
        This method sends a text prompt to the specified model running on
        the local Ollama service and returns the generated text response.
        All requests and responses are logged in full for complete transparency.
        
        Args:
            prompt: The text prompt to send to the model.
            model: The name of the model to use (default: "llama3:8b").
            
        Returns:
            The text response from the model.
            
        Raises:
            OllamaConnectionError: If unable to connect to the Ollama service.
            ValueError: If prompt is empty or None.
            
        Example:
            >>> client = OllamaClient()
            >>> response = client.invoke("Analyze this goal: Read file task.txt")
            >>> print(response)
            Tool: read_file
            Arguments: {"file_path": "task.txt"}
            Reasoning: The goal clearly states to read a file...
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or None")
        
        # Generate unique request ID for correlation
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # BULLETPROOF LOGGING: Log complete request details
        logger.info(f"[LLM-REQUEST-{request_id}] Starting LLM call at {timestamp}")
        logger.info(f"[LLM-REQUEST-{request_id}] Model: {model}")
        logger.info(f"[LLM-REQUEST-{request_id}] Host: {self.host}")
        logger.info(f"[LLM-REQUEST-{request_id}] Prompt length: {len(prompt)} characters")
        logger.info(f"[LLM-REQUEST-{request_id}] FULL PROMPT BEGINS:")
        logger.info(f"[LLM-REQUEST-{request_id}] {prompt}")
        logger.info(f"[LLM-REQUEST-{request_id}] FULL PROMPT ENDS")
        
        try:
            response = self._client.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'num_ctx': 4096,
                }
            )
            
            # STRICT VALIDATION: Validate response structure and content
            if not response or not isinstance(response, dict):
                logger.error(f"[LLM-ERROR-{request_id}] Invalid response type from Ollama")
                logger.error(f"[LLM-ERROR-{request_id}] Response type: {type(response)}")
                logger.error(f"[LLM-ERROR-{request_id}] Response value: {response}")
                raise OllamaConnectionError("Received invalid response type from Ollama")
            
            # STRICT VALIDATION: Extract and validate message content
            if 'message' not in response:
                logger.error(f"[LLM-ERROR-{request_id}] No 'message' field in Ollama response")
                logger.error(f"[LLM-ERROR-{request_id}] Available fields: {list(response.keys())}")
                logger.error(f"[LLM-ERROR-{request_id}] FULL RESPONSE DUMP: {response}")
                raise OllamaConnectionError("Ollama response missing 'message' field")
            
            message = response['message']
            if not isinstance(message, dict) or 'content' not in message:
                logger.error(f"[LLM-ERROR-{request_id}] Invalid message structure in Ollama response")
                logger.error(f"[LLM-ERROR-{request_id}] Message type: {type(message)}")
                logger.error(f"[LLM-ERROR-{request_id}] Message fields: {list(message.keys()) if isinstance(message, dict) else 'Not a dict'}")
                logger.error(f"[LLM-ERROR-{request_id}] FULL RESPONSE DUMP: {response}")
                raise OllamaConnectionError("Ollama response message missing 'content' field")
            
            content = message['content']
            
            # STRICT VALIDATION: Ensure content is a non-empty string
            if not isinstance(content, str):
                logger.error(f"[LLM-ERROR-{request_id}] Invalid content type in Ollama response")
                logger.error(f"[LLM-ERROR-{request_id}] Content type: {type(content)}")
                logger.error(f"[LLM-ERROR-{request_id}] Content value: {content}")
                raise OllamaConnectionError("Ollama response content is not a string")
            
            content = content.strip()
            
            if not content:
                logger.error(f"[LLM-ERROR-{request_id}] Empty content in Ollama response")
                logger.error(f"[LLM-ERROR-{request_id}] Original content: {repr(message['content'])}")
                raise OllamaConnectionError("Ollama response content is empty")
            
            # SUCCESS: Log complete response details
            response_timestamp = datetime.now().isoformat()
            logger.info(f"[LLM-RESPONSE-{request_id}] LLM call completed at {response_timestamp}")
            logger.info(f"[LLM-RESPONSE-{request_id}] Response length: {len(content)} characters")
            logger.info(f"[LLM-RESPONSE-{request_id}] FULL RESPONSE BEGINS:")
            logger.info(f"[LLM-RESPONSE-{request_id}] {content}")
            logger.info(f"[LLM-RESPONSE-{request_id}] FULL RESPONSE ENDS")
            logger.info(f"[LLM-SUCCESS-{request_id}] LLM call completed successfully - HTTP 200 + valid content")
            
            return content
                
        except ResponseError as e:
            # BULLETPROOF LOGGING: Log complete error details with full traceback
            error_timestamp = datetime.now().isoformat()
            logger.error(f"[LLM-ERROR-{request_id}] Ollama API error at {error_timestamp}")
            logger.error(f"[LLM-ERROR-{request_id}] Error type: {type(e).__name__}")
            logger.error(f"[LLM-ERROR-{request_id}] Error message: {str(e)}")
            logger.error(f"[LLM-ERROR-{request_id}] FULL TRACEBACK BEGINS:")
            logger.error(f"[LLM-ERROR-{request_id}] {traceback.format_exc()}")
            logger.error(f"[LLM-ERROR-{request_id}] FULL TRACEBACK ENDS")
            
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaConnectionError(
                    f"Unable to connect to Ollama service at {self.host}. "
                    f"Please ensure Ollama is running and the model '{model}' is available."
                ) from e
            else:
                raise OllamaConnectionError(f"Ollama API error: {e}") from e
                
        except Exception as e:
            # BULLETPROOF LOGGING: Log complete unexpected error details
            error_timestamp = datetime.now().isoformat()
            logger.error(f"[LLM-ERROR-{request_id}] Unexpected error at {error_timestamp}")
            logger.error(f"[LLM-ERROR-{request_id}] Error type: {type(e).__name__}")
            logger.error(f"[LLM-ERROR-{request_id}] Error message: {str(e)}")
            logger.error(f"[LLM-ERROR-{request_id}] FULL TRACEBACK BEGINS:")
            logger.error(f"[LLM-ERROR-{request_id}] {traceback.format_exc()}")
            logger.error(f"[LLM-ERROR-{request_id}] FULL TRACEBACK ENDS")
            
            raise OllamaConnectionError(
                f"Unexpected error communicating with Ollama service: {e}"
            ) from e
    
    def check_model_availability(self, model: str = "llama3:8b") -> bool:
        """Check if a specific model is available on the Ollama service.
        
        Args:
            model: The name of the model to check.
            
        Returns:
            True if the model is available, False otherwise.
        """
        try:
            models = self._client.list()
            available_models = [m['name'] for m in models.get('models', [])]
            is_available = any(model in available_model for available_model in available_models)
            
            logger.info(f"Model '{model}' availability check: {is_available}")
            if not is_available:
                logger.info(f"Available models: {available_models}")
                
            return is_available
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def list_available_models(self) -> list[str]:
        """Get a list of all available models on the Ollama service.
        
        Returns:
            List of available model names.
            
        Raises:
            OllamaConnectionError: If unable to connect to the service.
        """
        try:
            response = self._client.list()
            models = [m['name'] for m in response.get('models', [])]
            logger.info(f"Retrieved {len(models)} available models from Ollama")
            return models
            
        except Exception as e:
            logger.error(f"Error listing available models: {e}")
            raise OllamaConnectionError(
                f"Unable to retrieve model list from Ollama service: {e}"
            ) from e