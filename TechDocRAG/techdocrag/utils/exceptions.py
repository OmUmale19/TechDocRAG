"""
Custom exceptions for TechDocRAG system.
Provides clear error hierarchies for different failure modes.
"""


class TechDocRAGError(Exception):
    """Base exception class for TechDocRAG system."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        
    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            'error': self.__class__.__name__,
            'message': str(self),
            'error_code': self.error_code,
            'details': self.details
        }


class ConfigurationError(TechDocRAGError):
    """Raised when configuration is invalid or missing."""
    pass


class ProcessingError(TechDocRAGError):
    """Base class for document processing errors."""
    pass


class OCRError(ProcessingError):
    """Raised when OCR processing fails."""
    pass


class EmbeddingError(ProcessingError):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(TechDocRAGError):
    """Raised when retrieval operations fail."""
    pass


class ReasoningError(TechDocRAGError):
    """Raised when reasoning engine fails."""
    pass


class CalculationError(TechDocRAGError):
    """Raised when calculation operations fail."""
    pass


class VectorStoreError(TechDocRAGError):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(TechDocRAGError):
    """Raised when retrieval operations fail."""
    pass


class KeywordSearchError(RetrievalError):
    """Raised when keyword search operations fail."""
    pass


class EmbeddingError(TechDocRAGError):
    """Raised when embedding generation fails."""
    pass


class ConfidenceError(TechDocRAGError):
    """Raised when confidence calculation fails."""
    pass


class ValidationError(TechDocRAGError):
    """Raised when input validation fails."""
    pass


class SecurityError(TechDocRAGError):
    """Raised when security checks fail."""
    pass


class StorageError(TechDocRAGError):
    """Raised when storage operations fail."""
    pass


class APIError(TechDocRAGError):
    """Raised when external API calls fail."""
    pass


class LLMError(TechDocRAGError):
    """Raised when LLM operations fail."""
    pass


class TimeoutError(TechDocRAGError):
    """Raised when operations timeout."""
    pass


class InsufficientDataError(TechDocRAGError):
    """Raised when insufficient data is available for processing."""
    pass


class UnsupportedFormatError(ProcessingError):
    """Raised when document format is not supported."""
    pass


class PermissionError(SecurityError):
    """Raised when access permissions are insufficient."""
    pass
