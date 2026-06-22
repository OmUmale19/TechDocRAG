"""
Core type definitions for TechDocRAG system.
Provides type safety and clear interfaces across all components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import datetime
from pathlib import Path


class DocumentType(Enum):
    """Supported document types for specialized processing."""
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    RESUME = "resume"
    LEGAL_CONTRACT = "legal_contract"
    MEDICAL_RECORD = "medical_record"
    GENERAL = "general"


class ConfidenceLevel(Enum):
    """Confidence level categories for interpretability."""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # <0.5


@dataclass
class BoundingBox:
    """Bounding box coordinates for extracted elements."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 1


@dataclass
class ExtractedField:
    """Structured field extracted from document."""
    name: str
    value: Any
    confidence: float
    bbox: Optional[BoundingBox] = None
    extraction_method: str = "hybrid"  # rule_based, semantic, hybrid


@dataclass
class Document:
    """Base document class for input to the system."""
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: Optional[DocumentType] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class DocumentChunk:
    """Semantic chunk with metadata for retrieval."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    page_number: int = 1
    chunk_type: str = "text"  # text, table, image, header
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Alias for id to maintain compatibility."""
        return self.id


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    processing_time: float
    ocr_time: float
    embedding_time: float
    total_pages: int
    total_chunks: int
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Complete document processing result."""
    doc_id: str
    file_path: Path
    doc_type: DocumentType
    extracted_fields: List[ExtractedField]
    chunks: List[DocumentChunk]
    raw_text: str
    metadata: Dict[str, Any]
    metrics: ProcessingMetrics
    processed_at: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class ConfidenceScore:
    """Multi-layered confidence scoring."""
    overall: float
    retrieval: float
    reasoning: float
    calculation: float
    source_quality: float
    
    @property
    def level(self) -> ConfidenceLevel:
        if self.overall >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.overall >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


@dataclass
class SourceAttribution:
    """Source attribution for explainability."""
    doc_id: str
    chunk_id: str
    relevance_score: float
    text_snippet: str
    page_number: int
    bbox: Optional[BoundingBox] = None


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain."""
    step_id: str
    operation: str  # retrieve, extract, calculate, synthesize
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    sources: List[SourceAttribution] = field(default_factory=list)


@dataclass
class QueryResponse:
    """Complete query response with explainability."""
    query: str
    answer: str
    confidence: ConfidenceScore
    sources: List[SourceAttribution]
    reasoning_chain: List[ReasoningStep]
    calculations: List[Dict[str, Any]] = field(default_factory=list)
    response_time: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class MultiDocumentContext:
    """Context for multi-document reasoning."""
    doc_ids: List[str]
    relationships: Dict[str, List[str]]  # doc_id -> related_doc_ids
    timeline: List[Tuple[str, datetime.datetime]]  # doc_id, timestamp
    conflict_resolution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalculationResult:
    """Result of numeric calculation."""
    operation: str
    inputs: Dict[str, float]
    result: float
    formula: str
    confidence: float
    validation_checks: List[str] = field(default_factory=list)


@dataclass
class PIIMask:
    """PII masking information."""
    original_text: str
    masked_text: str
    entity_type: str  # SSN, EMAIL, PHONE, etc.
    confidence: float
    bbox: Optional[BoundingBox] = None


@dataclass
class SecurityContext:
    """Security and privacy context."""
    pii_masks: List[PIIMask]
    sensitivity_level: str  # PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
    access_controls: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
