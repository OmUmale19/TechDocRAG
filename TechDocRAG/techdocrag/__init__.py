# TechDocRAG Core Package
__version__ = "1.0.0"
__author__ = "TechDocRAG Team"
__description__ = "Intelligent Document Parsing and Reasoning System"

from .core.document_processor import DocumentProcessor
from .core.query_engine import QueryEngine
from .core.config import Config
from .core.types import DocumentResult, QueryResponse, ConfidenceScore

__all__ = [
    'DocumentProcessor',
    'QueryEngine', 
    'Config',
    'DocumentResult',
    'QueryResponse',
    'ConfidenceScore'
]
