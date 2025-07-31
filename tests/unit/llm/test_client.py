"""Unit tests for the OllamaClient class."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from oniks.llm.client import OllamaClient, OllamaConnectionError


class TestOllamaClient:
    """Test OllamaClient functionality."""
    
    def test_initialization_default_values(self):
        """Test OllamaClient initialization with default values."""
        client = OllamaClient()
        
        assert client.host == "http://localhost:11434"
        assert client.timeout == 30
    
    def test_initialization_custom_values(self):
        """Test OllamaClient initialization with custom values."""
        client = OllamaClient(host="http://custom-host:8080", timeout=60)
        
        assert client.host == "http://custom-host:8080"
        assert client.timeout == 60
    
    def test_invoke_empty_prompt_raises_error(self):
        """Test invoke with empty prompt raises ValueError."""
        client = OllamaClient()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty or None"):
            client.invoke("")
        
        with pytest.raises(ValueError, match="Prompt cannot be empty or None"):
            client.invoke(None)
        
        with pytest.raises(ValueError, match="Prompt cannot be empty or None"):
            client.invoke("   ")
    
    @patch('oniks.llm.client.ollama.Client')
    def test_invoke_success(self, mock_ollama_client_class):
        """Test successful invoke call."""
        # Mock the Ollama client instance
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        # Mock successful response
        mock_response = {
            'message': {
                'content': 'Tool: read_file\\nArguments: {"file_path": "test.txt"}'
            }
        }
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient()
        result = client.invoke("Test prompt")
        
        assert result == 'Tool: read_file\\nArguments: {"file_path": "test.txt"}'
        
        # Verify the client was called correctly
        mock_client_instance.chat.assert_called_once_with(
            model="tinyllama",
            messages=[
                {
                    'role': 'user',
                    'content': "Test prompt"
                }
            ],
            options={
                'temperature': 0.7,
                'num_ctx': 4096,
            }
        )
    
    @patch('oniks.llm.client.ollama.Client')
    def test_invoke_with_custom_model(self, mock_ollama_client_class):
        """Test invoke with custom model."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        mock_response = {
            'message': {
                'content': 'Custom model response'
            }
        }
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient()
        result = client.invoke("Test prompt", model="custom-model")
        
        assert result == "Custom model response"
        mock_client_instance.chat.assert_called_once()
        call_args = mock_client_instance.chat.call_args
        assert call_args[1]['model'] == "custom-model"
    
    @patch('oniks.llm.client.ollama.Client')
    def test_invoke_connection_error(self, mock_ollama_client_class):
        """Test invoke raises OllamaConnectionError on connection failure."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        # Mock connection error
        from ollama import ResponseError
        mock_client_instance.chat.side_effect = ResponseError("Connection refused")
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError, match="Unable to connect to Ollama service"):
            client.invoke("Test prompt")
    
    @patch('oniks.llm.client.ollama.Client')
    def test_invoke_unexpected_response_format(self, mock_ollama_client_class):
        """Test invoke handles unexpected response format."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        # Mock response with unexpected format
        mock_response = {"unexpected": "format"}
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError, match="Received unexpected response format"):
            client.invoke("Test prompt")
    
    @patch('oniks.llm.client.ollama.Client')
    def test_check_model_availability_success(self, mock_ollama_client_class):
        """Test successful model availability check."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        mock_response = {
            'models': [
                {'name': 'tinyllama:latest'},
                {'name': 'llama2:latest'},
            ]
        }
        mock_client_instance.list.return_value = mock_response
        
        client = OllamaClient()
        
        assert client.check_model_availability("tinyllama") is True
        assert client.check_model_availability("llama2") is True
        assert client.check_model_availability("nonexistent") is False
    
    @patch('oniks.llm.client.ollama.Client')
    def test_check_model_availability_error(self, mock_ollama_client_class):
        """Test model availability check handles errors gracefully."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        mock_client_instance.list.side_effect = Exception("Service unavailable")
        
        client = OllamaClient()
        
        assert client.check_model_availability("tinyllama") is False
    
    @patch('oniks.llm.client.ollama.Client')
    def test_list_available_models_success(self, mock_ollama_client_class):
        """Test successful listing of available models."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        mock_response = {
            'models': [
                {'name': 'tinyllama:latest'},
                {'name': 'llama2:latest'},
                {'name': 'codellama:latest'},
            ]
        }
        mock_client_instance.list.return_value = mock_response
        
        client = OllamaClient()
        models = client.list_available_models()
        
        assert len(models) == 3
        assert 'tinyllama:latest' in models
        assert 'llama2:latest' in models
        assert 'codellama:latest' in models
    
    @patch('oniks.llm.client.ollama.Client')
    def test_list_available_models_error(self, mock_ollama_client_class):
        """Test listing models raises error on service failure."""
        mock_client_instance = Mock()
        mock_ollama_client_class.return_value = mock_client_instance
        
        mock_client_instance.list.side_effect = Exception("Service unavailable")
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError, match="Unable to retrieve model list"):
            client.list_available_models()