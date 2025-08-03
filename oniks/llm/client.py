"""Ollama client implementation for LLM integration.

This module provides the OllamaClient class for interacting with locally
running Ollama instances to enable real LLM-powered reasoning in agents.
"""

import logging
import signal
import traceback
import uuid
from datetime import datetime
from typing import Optional

import ollama
from ollama import ResponseError, ChatResponse


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama service fails."""
    pass


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


class OllamaClient:
    """Client for interacting with locally running Ollama LLM service.
    
    This client provides a simple interface for sending prompts to Ollama
    and receiving text responses. It includes proper error handling for
    cases where the Ollama service is unavailable or encounters issues.
    
    Attributes:
        host: The host address for the Ollama service (default: localhost:11434).
        timeout: Request timeout in seconds (default: 1200).
    
    Example:
        >>> client = OllamaClient()
        >>> response = client.invoke("What is the capital of France?", model="llama3:8b")
        >>> print(response)
        The capital of France is Paris.
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 1200, default_model: str = "llama3:8b") -> None:
        """Initialize the Ollama client.
        
        Args:
            host: The host address for the Ollama service.
            timeout: Request timeout in seconds.
            default_model: Default model to use if none specified in invoke calls.
        """
        self.host = host
        self.timeout = timeout
        self.default_model = default_model
        self._client = ollama.Client(host=host)
    
    def invoke(self, prompt: str, model: Optional[str] = None) -> str:
        """Send a prompt to the Ollama service and return the response.
        
        This method sends a text prompt to the specified model running on
        the local Ollama service and returns the generated text response.
        All requests and responses are logged in full for complete transparency.
        
        Args:
            prompt: The text prompt to send to the model.
            model: The name of the model to use. If None, uses the default_model.
            
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
        
        # Use default model if none specified
        if model is None:
            model = self.default_model
            
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
            # Set up timeout using signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
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
            finally:
                # Always clear the alarm
                signal.alarm(0)
            
            # STRICT VALIDATION: Validate response structure and content
            if not response:
                logger.error(f"[LLM-ERROR-{request_id}] Received empty response from Ollama")
                logger.error(f"[LLM-ERROR-{request_id}] Response value: {response}")
                raise OllamaConnectionError("Received empty response from Ollama")
            
            # Handle both ChatResponse objects and dictionary responses
            if isinstance(response, ChatResponse):
                # Extract content from ChatResponse object
                if not hasattr(response, 'message') or not response.message:
                    logger.error(f"[LLM-ERROR-{request_id}] ChatResponse missing message attribute")
                    logger.error(f"[LLM-ERROR-{request_id}] Response type: {type(response)}")
                    logger.error(f"[LLM-ERROR-{request_id}] Available attributes: {dir(response)}")
                    raise OllamaConnectionError("ChatResponse missing message attribute")
                
                message = response.message
                if not hasattr(message, 'content'):
                    logger.error(f"[LLM-ERROR-{request_id}] Message object missing content attribute")
                    logger.error(f"[LLM-ERROR-{request_id}] Message type: {type(message)}")
                    logger.error(f"[LLM-ERROR-{request_id}] Available attributes: {dir(message)}")
                    raise OllamaConnectionError("Message object missing content attribute")
                
                content = message.content
                
            elif isinstance(response, dict):
                # Handle dictionary response format (legacy compatibility)
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
                
            else:
                logger.error(f"[LLM-ERROR-{request_id}] Invalid response type from Ollama")
                logger.error(f"[LLM-ERROR-{request_id}] Response type: {type(response)}")
                logger.error(f"[LLM-ERROR-{request_id}] Expected: ChatResponse or dict")
                logger.error(f"[LLM-ERROR-{request_id}] Response value: {response}")
                raise OllamaConnectionError("Received invalid response type from Ollama")
            
            # STRICT VALIDATION: Ensure content is a non-empty string
            if not isinstance(content, str):
                logger.error(f"[LLM-ERROR-{request_id}] Invalid content type in Ollama response")
                logger.error(f"[LLM-ERROR-{request_id}] Content type: {type(content)}")
                logger.error(f"[LLM-ERROR-{request_id}] Content value: {content}")
                raise OllamaConnectionError("Ollama response content is not a string")
            
            content = content.strip()
            
            if not content:
                logger.error(f"[LLM-ERROR-{request_id}] Empty content in Ollama response")
                if isinstance(response, ChatResponse):
                    logger.error(f"[LLM-ERROR-{request_id}] Original content: {repr(response.message.content)}")
                else:
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
        
        except TimeoutError:
            # Handle timeout specifically
            error_timestamp = datetime.now().isoformat()
            logger.error(f"[LLM-TIMEOUT-{request_id}] LLM request timed out after {self.timeout} seconds at {error_timestamp}")
            raise OllamaConnectionError(f"LLM request timed out after {self.timeout} seconds")
                
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
            available_models = []
            
            # Handle different response formats from Ollama
            if hasattr(models, 'models'):
                for m in models.models:
                    # Try multiple methods to extract model name
                    model_name = None
                    
                    # Method 1: Try standard attributes (name, model)
                    for attr_name in ['name', 'model']:
                        if hasattr(m, attr_name):
                            try:
                                model_name = getattr(m, attr_name)
                                if model_name:
                                    break
                            except:
                                continue
                    
                    # Method 2: If it's a dict-like object
                    if not model_name and hasattr(m, 'get'):
                        try:
                            model_name = m.get('name') or m.get('model')
                        except:
                            pass
                    
                    # Method 3: If it's a dict (fallback)
                    if not model_name and isinstance(m, dict):
                        model_name = m.get('name', m.get('model', ''))
                    
                    # Method 4: Try accessing as attribute dynamically
                    if not model_name:
                        try:
                            # Get all attributes and look for name-like ones
                            obj_dict = getattr(m, '__dict__', {})
                            for key, value in obj_dict.items():
                                if key in ['name', 'model', 'id'] and value:
                                    model_name = value
                                    break
                        except:
                            pass
                    
                    # Method 5: String representation as last resort
                    if not model_name:
                        try:
                            str_repr = str(m)
                            # Only use if it's not just the type representation
                            if str_repr and not str_repr.startswith('<') and ':' in str_repr:
                                # Try to extract model name from string representation
                                if 'name=' in str_repr:
                                    parts = str_repr.split('name=')
                                    if len(parts) > 1:
                                        name_part = parts[1].split(',')[0].split(')')[0].strip('\'"')
                                        if name_part:
                                            model_name = name_part
                        except:
                            pass
                    
                    if model_name and isinstance(model_name, str) and model_name.strip():
                        available_models.append(model_name.strip())
                        logger.debug(f"Extracted model name: {model_name}")
                    else:
                        # Log for debugging but don't fail
                        logger.debug(f"Could not extract model name from object: {type(m)}, repr: {repr(m)[:100]}")
            else:
                # Fallback for older Ollama versions
                models_list = getattr(models, 'models', [])
                if isinstance(models_list, list):
                    for m in models_list:
                        if isinstance(m, dict):
                            name = m.get('name', m.get('model', ''))
                            if name:
                                available_models.append(name)
            
            # Remove duplicates and empty strings
            available_models = list(set([m for m in available_models if m and m.strip()]))
            is_available = model in available_models
            
            logger.info(f"Model '{model}' availability check: {is_available}")
            logger.info(f"Available models: {available_models}")
                
            return is_available
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            models = []
            
            # Handle different response formats from Ollama
            if hasattr(response, 'models'):
                for m in response.models:
                    # Try multiple methods to extract model name (same as check_model_availability)
                    model_name = None
                    
                    # Method 1: Try standard attributes (name, model)
                    for attr_name in ['name', 'model']:
                        if hasattr(m, attr_name):
                            try:
                                model_name = getattr(m, attr_name)
                                if model_name:
                                    break
                            except:
                                continue
                    
                    # Method 2: If it's a dict-like object
                    if not model_name and hasattr(m, 'get'):
                        try:
                            model_name = m.get('name') or m.get('model')
                        except:
                            pass
                    
                    # Method 3: If it's a dict (fallback)
                    if not model_name and isinstance(m, dict):
                        model_name = m.get('name', m.get('model', ''))
                    
                    # Method 4: Try accessing as attribute dynamically
                    if not model_name:
                        try:
                            # Get all attributes and look for name-like ones
                            obj_dict = getattr(m, '__dict__', {})
                            for key, value in obj_dict.items():
                                if key in ['name', 'model', 'id'] and value:
                                    model_name = value
                                    break
                        except:
                            pass
                    
                    if model_name and isinstance(model_name, str) and model_name.strip():
                        models.append(model_name.strip())
            else:
                # Fallback for older Ollama versions
                models_list = getattr(response, 'models', [])
                if isinstance(models_list, list):
                    for m in models_list:
                        if isinstance(m, dict):
                            name = m.get('name', m.get('model', ''))
                            if name:
                                models.append(name)
            
            # Remove duplicates and empty strings
            models = list(set([m for m in models if m and m.strip()]))
            logger.info(f"Retrieved {len(models)} available models from Ollama")
            return models
            
        except Exception as e:
            logger.error(f"Error listing available models: {e}")
            raise OllamaConnectionError(
                f"Unable to retrieve model list from Ollama service: {e}"
            ) from e