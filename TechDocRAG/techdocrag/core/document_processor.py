"""
Core Document Processor for TechDocRAG system.
Orchestrates the entire document processing pipeline from raw input to structured knowledge.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from ..core.types import (
    DocumentResult, DocumentType, ExtractedField, DocumentChunk, 
    ProcessingMetrics, BoundingBox
)
from ..core.config import get_config
from ..utils.logging import get_logger, performance_log, audit_logger
from ..utils.helpers import generate_id, safe_filename, hash_content
from ..utils.validators import validate_file_type, validate_file_size
from ..utils.exceptions import ProcessingError, OCRError, ValidationError

from ..processing.ocr_engine import OCREngine
from ..processing.text_processor import TextProcessor
from ..processing.layout_analyzer import LayoutAnalyzer
from ..processing.field_extractor import FieldExtractor
from ..processing.embedding_generator import EmbeddingGenerator


logger = get_logger(__name__)


class DocumentProcessor:
    """
    Main document processing orchestrator.
    
    Coordinates OCR, text processing, field extraction, and embedding generation
    to transform raw documents into structured, searchable knowledge.
    """
    
    def __init__(self, config=None):
        """Initialize document processor with configuration."""
        self.config = config or get_config()
        
        # Initialize processing components
        self.ocr_engine = OCREngine(self.config.ocr)
        self.text_processor = TextProcessor(self.config.processing)
        self.layout_analyzer = LayoutAnalyzer(self.config.processing)
        self.field_extractor = FieldExtractor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config.embedding)
        
        logger.info("DocumentProcessor initialized with all components")
    
    @performance_log("document_processing", track_memory=True)
    async def process(
        self, 
        file_path: Union[str, Path],
        doc_type: Union[str, DocumentType] = DocumentType.GENERAL,
        metadata: Dict[str, Any] = None,
        user_id: str = None
    ) -> DocumentResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            doc_type: Type of document for specialized processing
            metadata: Additional metadata about the document
            user_id: User ID for audit logging
            
        Returns:
            DocumentResult with all extracted information
            
        Raises:
            ProcessingError: If processing fails at any stage
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        # Generate unique document ID
        doc_id = generate_id("doc")
        
        try:
            logger.info(f"Starting document processing: {file_path}", 
                       extra={'doc_id': doc_id, 'user_id': user_id})
            
            # Audit log
            audit_logger.log_document_access(
                doc_id=doc_id,
                action="process",
                user_id=user_id,
                details={'file_path': str(file_path), 'doc_type': str(doc_type)}
            )
            
            # Validate input
            await self._validate_input(file_path, doc_type)
            
            # Convert doc_type to enum if string
            if isinstance(doc_type, str):
                doc_type = DocumentType(doc_type.lower())
            
            # Step 1: OCR and raw text extraction
            ocr_start = datetime.now()
            raw_text, ocr_metadata = await self.ocr_engine.extract_text(file_path)
            ocr_time = (datetime.now() - ocr_start).total_seconds()
            
            logger.info(f"OCR completed: {len(raw_text)} characters extracted",
                       extra={'doc_id': doc_id})
            
            # Step 2: Layout analysis for structure understanding
            layout_elements = await self.layout_analyzer.analyze(file_path, ocr_metadata)
            
            # Step 3: Text processing and chunking
            chunks = await self.text_processor.process_and_chunk(
                raw_text, layout_elements, doc_type
            )
            
            logger.info(f"Text processing completed: {len(chunks)} chunks created",
                       extra={'doc_id': doc_id})
            
            # Step 4: Field extraction using hybrid approach
            extracted_fields = await self.field_extractor.extract_fields(
                raw_text, chunks, doc_type, layout_elements
            )
            
            logger.info(f"Field extraction completed: {len(extracted_fields)} fields extracted",
                       extra={'doc_id': doc_id})
            
            # Step 5: Generate embeddings for semantic search
            embedding_start = datetime.now()
            chunks_with_embeddings = await self.embedding_generator.generate_embeddings(chunks)
            embedding_time = (datetime.now() - embedding_start).total_seconds()
            
            logger.info(f"Embedding generation completed for {len(chunks_with_embeddings)} chunks",
                       extra={'doc_id': doc_id})
            
            # Step 6: Compile processing metrics
            total_time = (datetime.now() - start_time).total_seconds()
            metrics = ProcessingMetrics(
                processing_time=total_time,
                ocr_time=ocr_time,
                embedding_time=embedding_time,
                total_pages=ocr_metadata.get('total_pages', 1),
                total_chunks=len(chunks_with_embeddings),
                confidence_scores={
                    'ocr_confidence': ocr_metadata.get('confidence', 0.0),
                    'field_extraction_confidence': self._calculate_field_confidence(extracted_fields),
                    'overall_confidence': self._calculate_overall_confidence(
                        ocr_metadata.get('confidence', 0.0),
                        extracted_fields
                    )
                }
            )
            
            # Step 7: Compile complete document result
            document_result = DocumentResult(
                doc_id=doc_id,
                file_path=file_path,
                doc_type=doc_type,
                extracted_fields=extracted_fields,
                chunks=chunks_with_embeddings,
                raw_text=raw_text,
                metadata={
                    **(metadata or {}),
                    **ocr_metadata,
                    'processing_timestamp': start_time.isoformat(),
                    'file_size': file_path.stat().st_size,
                    'file_hash': hash_content(file_path.read_bytes()),
                },
                metrics=metrics
            )
            
            logger.info(f"Document processing completed successfully",
                       extra={
                           'doc_id': doc_id,
                           'processing_time': total_time,
                           'fields_extracted': len(extracted_fields),
                           'chunks_created': len(chunks_with_embeddings)
                       })
            
            return document_result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}", 
                        extra={'doc_id': doc_id, 'error': str(e)})
            raise ProcessingError(
                f"Failed to process document {file_path}: {str(e)}",
                error_code="PROCESSING_FAILED",
                details={'doc_id': doc_id, 'file_path': str(file_path)}
            ) from e
    
    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        doc_types: List[Union[str, DocumentType]] = None,
        batch_size: int = None,
        user_id: str = None
    ) -> List[DocumentResult]:
        """
        Process multiple documents in parallel batches.
        
        Args:
            file_paths: List of file paths to process
            doc_types: Document types (one per file or single type for all)
            batch_size: Number of documents to process concurrently
            user_id: User ID for audit logging
            
        Returns:
            List of DocumentResult objects
        """
        batch_size = batch_size or self.config.performance.batch_size
        
        # Prepare document types
        if doc_types is None:
            doc_types = [DocumentType.GENERAL] * len(file_paths)
        elif len(doc_types) == 1:
            doc_types = doc_types * len(file_paths)
        
        if len(doc_types) != len(file_paths):
            raise ValidationError(
                "Number of document types must match number of files",
                error_code="BATCH_SIZE_MISMATCH"
            )
        
        logger.info(f"Starting batch processing: {len(file_paths)} documents",
                   extra={'batch_size': batch_size, 'user_id': user_id})
        
        results = []
        
        # Process in batches to control resource usage
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            batch_types = doc_types[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} documents")
            
            # Process batch concurrently
            batch_tasks = [
                self.process(file_path, doc_type, user_id=user_id)
                for file_path, doc_type in zip(batch_files, batch_types)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {batch_files[j]}: {str(result)}")
                    # Create error result
                    error_result = self._create_error_result(batch_files[j], result)
                    results.append(error_result)
                else:
                    results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} results",
                   extra={'successful': sum(1 for r in results if not hasattr(r, 'error'))})
        
        return results
    
    async def _validate_input(self, file_path: Path, doc_type: DocumentType):
        """Validate input file and parameters."""
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        # Validate file type and size
        validate_file_type(file_path)
        validate_file_size(file_path)
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first KB
        except Exception as e:
            raise ValidationError(f"Cannot read file: {file_path}") from e
    
    def _calculate_field_confidence(self, fields: List[ExtractedField]) -> float:
        """Calculate average confidence across extracted fields."""
        if not fields:
            return 0.0
        
        total_confidence = sum(field.confidence for field in fields)
        return total_confidence / len(fields)
    
    def _calculate_overall_confidence(self, ocr_confidence: float, fields: List[ExtractedField]) -> float:
        """Calculate overall processing confidence."""
        field_confidence = self._calculate_field_confidence(fields)
        
        # Weighted average: OCR 40%, Fields 60%
        return 0.4 * ocr_confidence + 0.6 * field_confidence
    
    def _create_error_result(self, file_path: Path, error: Exception) -> DocumentResult:
        """Create a DocumentResult for failed processing."""
        return DocumentResult(
            doc_id=generate_id("error"),
            file_path=file_path,
            doc_type=DocumentType.GENERAL,
            extracted_fields=[],
            chunks=[],
            raw_text="",
            metadata={
                'error': str(error),
                'error_type': type(error).__name__,
                'processing_failed': True
            },
            metrics=ProcessingMetrics(
                processing_time=0.0,
                ocr_time=0.0,
                embedding_time=0.0,
                total_pages=0,
                total_chunks=0
            )
        )
    
    async def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats by category."""
        return {
            'documents': ['.pdf', '.docx', '.doc', '.txt'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'],
            'all': ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and health metrics."""
        return {
            'components_initialized': {
                'ocr_engine': self.ocr_engine is not None,
                'text_processor': self.text_processor is not None,
                'layout_analyzer': self.layout_analyzer is not None,
                'field_extractor': self.field_extractor is not None,
                'embedding_generator': self.embedding_generator is not None,
            },
            'configuration': {
                'ocr_engine': self.config.ocr.primary_engine,
                'embedding_model': self.config.embedding.model_name,
                'chunk_size': self.config.processing.chunk_size,
                'batch_size': self.config.performance.batch_size,
            },
            'capabilities': {
                'supported_formats': await self.get_supported_formats(),
                'document_types': [doc_type.value for doc_type in DocumentType],
                'max_file_size_mb': 100,  # Based on configuration
                'concurrent_processing': self.config.performance.enable_async_processing,
            }
        }
