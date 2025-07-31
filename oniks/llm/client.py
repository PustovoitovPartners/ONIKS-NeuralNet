"""Ollama client implementation for LLM integration.

This module provides the OllamaClient class for interacting with locally
running Ollama instances to enable real LLM-powered reasoning in agents.
"""

import logging
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
        >>> response = client.invoke("What is the capital of France?", model="llama3")
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
    
    def invoke(self, prompt: str, model: str = "llama3") -> str:
        """Send a prompt to the Ollama service and return the response.
        
        This method sends a text prompt to the specified model running on
        the local Ollama service and returns the generated text response.
        
        Args:
            prompt: The text prompt to send to the model.
            model: The name of the model to use (default: "llama3").
            
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
        
        logger.info(f"Sending prompt to Ollama model '{model}' (length: {len(prompt)} chars)")
        
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
            
            # Extract the message content from the response
            if 'message' in response and 'content' in response['message']:
                content = response['message']['content'].strip()
                logger.info(f"Received response from Ollama (length: {len(content)} chars)")
                return content
            else:
                logger.error(f"Unexpected response format: {response}")
                raise OllamaConnectionError("Received unexpected response format from Ollama")
                
        except ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaConnectionError(
                    f"Unable to connect to Ollama service at {self.host}. "
                    f"Please ensure Ollama is running and the model '{model}' is available."
                ) from e
            else:
                raise OllamaConnectionError(f"Ollama API error: {e}") from e
                
        except Exception as e:
            logger.error(f"Unexpected error during Ollama invocation: {e}")
            raise OllamaConnectionError(
                f"Unexpected error communicating with Ollama service: {e}"
            ) from e
    
    def check_model_availability(self, model: str = "llama3") -> bool:
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