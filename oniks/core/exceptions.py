"""Custom exceptions for the ONIKS NeuralNet framework.

This module defines custom exception classes used throughout the framework
to provide clear error handling and reporting.
"""

from datetime import datetime
from typing import Optional, Dict, Any


class LLMUnavailableError(Exception):
    """Critical exception raised when LLM service is unavailable or fails.
    
    This exception is raised when the system cannot proceed without LLM
    functionality. It includes detailed error information for debugging
    and correlation purposes.
    
    Attributes:
        message: The error message describing the failure.
        original_error: The original exception that caused this error.
        request_details: Dictionary containing request information.
        timestamp: When the error occurred.
        correlation_id: Unique identifier for error correlation.
    """
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        request_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the LLMUnavailableError.
        
        Args:
            message: Descriptive error message.
            original_error: The original exception that caused this error.
            request_details: Dictionary with request context information.
            correlation_id: Unique identifier for error correlation.
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.request_details = request_details or {}
        self.timestamp = datetime.now().isoformat()
        self.correlation_id = correlation_id
    
    def __str__(self) -> str:
        """Return a detailed string representation of the error."""
        error_parts = [f"LLM Unavailable: {self.message}"]
        
        if self.correlation_id:
            error_parts.append(f"Correlation ID: {self.correlation_id}")
        
        error_parts.append(f"Timestamp: {self.timestamp}")
        
        if self.original_error:
            error_parts.append(f"Original Error: {type(self.original_error).__name__}: {self.original_error}")
        
        if self.request_details:
            error_parts.append(f"Request Details: {self.request_details}")
        
        return " | ".join(error_parts)
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete error context as a dictionary.
        
        Returns:
            Dictionary containing all error context information.
        """
        return {
            "error_type": "LLMUnavailableError",
            "message": self.message,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "original_error": {
                "type": type(self.original_error).__name__ if self.original_error else None,
                "message": str(self.original_error) if self.original_error else None
            },
            "request_details": self.request_details
        }


class PlanningTimeoutError(Exception):
    """Critical exception raised when planning cycle exceeds timeout limit.
    
    This exception is raised when the PlannerAgent cannot generate and successfully
    parse a plan within the specified timeout period. It prevents infinite hangs
    and provides clear timeout information for debugging.
    
    Attributes:
        message: The error message describing the timeout failure.
        timeout_seconds: The timeout limit that was exceeded.
        elapsed_seconds: How long the operation actually took.
        correlation_id: Unique identifier for error correlation.
        request_details: Dictionary containing request information.
        timestamp: When the timeout occurred.
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        elapsed_seconds: float,
        correlation_id: Optional[str] = None,
        request_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the PlanningTimeoutError.
        
        Args:
            message: Descriptive error message.
            timeout_seconds: The timeout limit that was exceeded.
            elapsed_seconds: How long the operation actually took.
            correlation_id: Unique identifier for error correlation.
            request_details: Dictionary with request context information.
        """
        super().__init__(message)
        self.message = message
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.correlation_id = correlation_id
        self.request_details = request_details or {}
        self.timestamp = datetime.now().isoformat()
    
    def __str__(self) -> str:
        """Return a detailed string representation of the timeout error."""
        error_parts = [f"Planning Timeout: {self.message}"]
        
        if self.correlation_id:
            error_parts.append(f"Correlation ID: {self.correlation_id}")
        
        error_parts.append(f"Timeout Limit: {self.timeout_seconds}s")
        error_parts.append(f"Elapsed Time: {self.elapsed_seconds:.2f}s")
        error_parts.append(f"Timestamp: {self.timestamp}")
        
        if self.request_details:
            error_parts.append(f"Request Details: {self.request_details}")
        
        return " | ".join(error_parts)
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete timeout error context as a dictionary.
        
        Returns:
            Dictionary containing all timeout error context information.
        """
        return {
            "error_type": "PlanningTimeoutError",
            "message": self.message,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "request_details": self.request_details
        }