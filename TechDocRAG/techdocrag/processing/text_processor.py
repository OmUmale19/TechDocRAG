"""
Advanced text processing and semantic chunking system.
Handles preprocessing, cleaning, and intelligent chunking for optimal retrieval.
"""

import re
import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.types import DocumentChunk, DocumentType, BoundingBox
from ..core.config import ProcessingConfig
from ..utils.logging import get_logger, performance_log
from ..utils.helpers import generate_id, clean_text, chunk_list
from ..utils.exceptions import ProcessingError

logger = get_logger(__name__)


@dataclass
class TextSection:
    """Represents a logical section of text."""
    text: str
    section_type: str  # header, paragraph, table, list, footer
    importance: float  # 0-1 relevance score
    page_number: int = 1
    bbox: Optional[BoundingBox] = None


class TextProcessor:
    """
    Intelligent text processor with semantic chunking capabilities.
    
    Key Innovations:
    1. Layout-aware chunking: Respects document structure
    2. Semantic coherence: Maintains topic boundaries  
    3. Context preservation: Overlapping chunks with metadata
    4. Domain adaptation: Specialized processing per document type
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic processing")
            self.nlp = None
        
        # Document type specific processors
        self.type_processors = {
            DocumentType.INVOICE: self._process_invoice,
            DocumentType.RESUME: self._process_resume,
            DocumentType.LEGAL_CONTRACT: self._process_legal,
            DocumentType.MEDICAL_RECORD: self._process_medical,
            DocumentType.PURCHASE_ORDER: self._process_purchase_order,
        }
        
        logger.info("TextProcessor initialized")
    
    @performance_log("text_processing")
    async def process_and_chunk(
        self,
        raw_text: str,
        layout_elements: List[Dict] = None,
        doc_type: DocumentType = DocumentType.GENERAL
    ) -> List[DocumentChunk]:
        """
        Process raw text and create semantic chunks.
        
        Args:
            raw_text: Raw extracted text
            layout_elements: Layout analysis results
            doc_type: Document type for specialized processing
            
        Returns:
            List of processed DocumentChunk objects
        """
        logger.info(f"Processing text: {len(raw_text)} characters, type: {doc_type}")
        
        try:
            # Step 1: Clean and normalize text
            cleaned_text = self._clean_text(raw_text)
            
            # Step 2: Identify text sections using layout information
            sections = self._identify_sections(cleaned_text, layout_elements, doc_type)
            
            # Step 3: Apply document-type specific processing
            if doc_type in self.type_processors:
                sections = await self.type_processors[doc_type](sections)
            
            # Step 4: Create semantic chunks
            chunks = self._create_semantic_chunks(sections, doc_type)
            
            # Step 5: Add metadata and finalize
            final_chunks = self._finalize_chunks(chunks, doc_type)
            
            logger.info(f"Created {len(final_chunks)} chunks from {len(sections)} sections")
            return final_chunks
            
        except Exception as e:
            raise ProcessingError(f"Text processing failed: {str(e)}") from e
    
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning and normalization."""
        # Basic cleaning
        cleaned = clean_text(text)
        
        # Remove OCR artifacts
        cleaned = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]{}"\'/\$%&*+=<>]', ' ', cleaned)
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l to I
            r'\b0\b': 'O',  # zero to O in words
            r'(?<=\d),(?=\d{3}(?:\D|$))': '',  # Remove thousands separators
            r'(?<=\w)- (?=\w)': '',  # Remove hyphen line breaks
        }
        
        for pattern, replacement in ocr_fixes.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Normalize whitespace while preserving structure
        lines = cleaned.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = re.sub(r'\s+', ' ', line).strip()
            if line:  # Keep non-empty lines
                normalized_lines.append(line)
            elif normalized_lines and normalized_lines[-1]:  # Add single empty line
                normalized_lines.append('')
        
        return '\n'.join(normalized_lines)
    
    def _identify_sections(
        self, 
        text: str, 
        layout_elements: List[Dict] = None, 
        doc_type: DocumentType = DocumentType.GENERAL
    ) -> List[TextSection]:
        """Identify logical sections in the text."""
        sections = []
        
        if layout_elements:
            # Use layout analysis results
            sections = self._sections_from_layout(text, layout_elements)
        else:
            # Fallback to heuristic section detection
            sections = self._sections_from_heuristics(text, doc_type)
        
        return sections
    
    def _sections_from_layout(self, text: str, layout_elements: List[Dict]) -> List[TextSection]:
        """Create sections based on layout analysis."""
        sections = []
        current_page = 1
        
        for element in layout_elements:
            element_type = element.get('type', 'paragraph')
            element_text = element.get('text', '')
            bbox = element.get('bbox')
            page = element.get('page', current_page)
            confidence = element.get('confidence', 0.5)
            
            if element_text.strip():
                section = TextSection(
                    text=element_text,
                    section_type=element_type,
                    importance=confidence,
                    page_number=page,
                    bbox=bbox
                )
                sections.append(section)
        
        return sections
    
    def _sections_from_heuristics(self, text: str, doc_type: DocumentType) -> List[TextSection]:
        """Create sections using text-based heuristics."""
        sections = []
        lines = text.split('\n')
        current_section = []
        current_type = 'paragraph'
        page_number = 1
        
        for line in lines:
            if not line.strip():
                if current_section:
                    # End current section
                    sections.append(TextSection(
                        text='\n'.join(current_section),
                        section_type=current_type,
                        importance=self._calculate_importance(current_section, doc_type),
                        page_number=page_number
                    ))
                    current_section = []
                continue
            
            # Detect section type
            detected_type = self._detect_line_type(line, doc_type)
            
            if detected_type != current_type and current_section:
                # Section type changed, save current section
                sections.append(TextSection(
                    text='\n'.join(current_section),
                    section_type=current_type,
                    importance=self._calculate_importance(current_section, doc_type),
                    page_number=page_number
                ))
                current_section = []
            
            current_section.append(line)
            current_type = detected_type
        
        # Add final section
        if current_section:
            sections.append(TextSection(
                text='\n'.join(current_section),
                section_type=current_type,
                importance=self._calculate_importance(current_section, doc_type),
                page_number=page_number
            ))
        
        return sections
    
    def _detect_line_type(self, line: str, doc_type: DocumentType) -> str:
        """Detect the type of a text line."""
        line = line.strip()
        
        # Header patterns
        if re.match(r'^[A-Z\s]+$', line) and len(line) < 100:
            return 'header'
        
        # List items
        if re.match(r'^\s*[•\-\*\d+\.]\s+', line):
            return 'list'
        
        # Table-like content
        if '\t' in line or re.search(r'\s{3,}', line):
            return 'table'
        
        # Document type specific patterns
        if doc_type == DocumentType.INVOICE:
            if re.search(r'(invoice|bill|amount|total|due)', line.lower()):
                return 'invoice_field'
        elif doc_type == DocumentType.RESUME:
            if re.search(r'(experience|education|skills|work)', line.lower()):
                return 'resume_section'
        
        return 'paragraph'
    
    def _calculate_importance(self, section_lines: List[str], doc_type: DocumentType) -> float:
        """Calculate importance score for a section."""
        if not section_lines:
            return 0.0
        
        text = ' '.join(section_lines).lower()
        importance = 0.5  # Base importance
        
        # Length-based scoring
        if len(text) > 100:
            importance += 0.2
        
        # Keyword-based scoring by document type
        if doc_type == DocumentType.INVOICE:
            keywords = ['total', 'amount', 'due', 'invoice', 'payment', 'tax']
        elif doc_type == DocumentType.RESUME:
            keywords = ['experience', 'education', 'skills', 'work', 'university']
        elif doc_type == DocumentType.LEGAL_CONTRACT:
            keywords = ['agreement', 'party', 'terms', 'condition', 'clause']
        else:
            keywords = ['important', 'summary', 'conclusion', 'key', 'main']
        
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        importance += min(0.3, keyword_count * 0.1)
        
        return min(1.0, importance)
    
    def _create_semantic_chunks(self, sections: List[TextSection], doc_type: DocumentType) -> List[DocumentChunk]:
        """Create semantic chunks from text sections."""
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(section, doc_type)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_section(self, section: TextSection, doc_type: DocumentType) -> List[DocumentChunk]:
        """Chunk a single section into optimal sizes."""
        text = section.text
        
        # If section is small enough, keep as single chunk
        if len(text) <= self.config.chunk_size:
            return [DocumentChunk(
                id=generate_id("chunk"),
                content=text,
                page_number=section.page_number,
                chunk_type=section.section_type,
                bbox=section.bbox,
                metadata={
                    'importance': section.importance,
                    'section_type': section.section_type,
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
            )]
        
        # Split large sections intelligently
        chunks = []
        
        if self.nlp and section.section_type == 'paragraph':
            # Use spaCy for semantic sentence boundary detection
            chunks = self._semantic_chunking(text, section)
        else:
            # Use sliding window approach
            chunks = self._sliding_window_chunking(text, section)
        
        return chunks
    
    def _semantic_chunking(self, text: str, section: TextSection) -> List[DocumentChunk]:
        """Create semantically coherent chunks using NLP."""
        chunks = []
        doc = self.nlp(text)
        
        sentences = [sent.text for sent in doc.sents]
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    id=generate_id("chunk"),
                    content=chunk_text,
                    page_number=section.page_number,
                    chunk_type=section.section_type,
                    bbox=section.bbox,
                    metadata={
                        'importance': section.importance,
                        'section_type': section.section_type,
                        'sentence_count': len(current_chunk),
                        'semantic_chunk': True
                    }
                ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                id=generate_id("chunk"),
                content=chunk_text,
                page_number=section.page_number,
                chunk_type=section.section_type,
                bbox=section.bbox,
                metadata={
                    'importance': section.importance,
                    'section_type': section.section_type,
                    'sentence_count': len(current_chunk),
                    'semantic_chunk': True
                }
            ))
        
        return chunks
    
    def _sliding_window_chunking(self, text: str, section: TextSection) -> List[DocumentChunk]:
        """Create chunks using sliding window approach."""
        chunks = []
        words = text.split()
        
        # Calculate words per chunk based on average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 5
        words_per_chunk = int(self.config.chunk_size / avg_word_length)
        overlap_words = int(self.config.chunk_overlap / avg_word_length)
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            if not chunk_words:
                break
                
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(DocumentChunk(
                id=generate_id("chunk"),
                content=chunk_text,
                page_number=section.page_number,
                chunk_type=section.section_type,
                bbox=section.bbox,
                metadata={
                    'importance': section.importance,
                    'section_type': section.section_type,
                    'word_count': len(chunk_words),
                    'sliding_window': True,
                    'chunk_index': len(chunks)
                }
            ))
        
        return chunks
    
    def _finalize_chunks(self, chunks: List[DocumentChunk], doc_type: DocumentType) -> List[DocumentChunk]:
        """Add final metadata and validation to chunks."""
        finalized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Add position metadata
            chunk.metadata.update({
                'chunk_position': i,
                'total_chunks': len(chunks),
                'doc_type': doc_type.value,
                'min_chunk_size': len(chunk.content) >= self.config.min_chunk_size
            })
            
            # Skip chunks that are too small (unless they're important)
            if (len(chunk.content) < self.config.min_chunk_size and 
                chunk.metadata.get('importance', 0) < 0.8):
                continue
            
            finalized_chunks.append(chunk)
        
        return finalized_chunks
    
    # Document type specific processors
    async def _process_invoice(self, sections: List[TextSection]) -> List[TextSection]:
        """Invoice-specific processing."""
        # Boost importance of financial sections
        for section in sections:
            if re.search(r'(total|amount|due|tax|subtotal)', section.text.lower()):
                section.importance = min(1.0, section.importance + 0.3)
        return sections
    
    async def _process_resume(self, sections: List[TextSection]) -> List[TextSection]:
        """Resume-specific processing."""
        # Boost importance of key sections
        for section in sections:
            if re.search(r'(experience|education|skills|summary)', section.text.lower()):
                section.importance = min(1.0, section.importance + 0.3)
        return sections
    
    async def _process_legal(self, sections: List[TextSection]) -> List[TextSection]:
        """Legal document-specific processing."""
        # Boost importance of legal terms
        for section in sections:
            if re.search(r'(clause|term|condition|agreement|party)', section.text.lower()):
                section.importance = min(1.0, section.importance + 0.2)
        return sections
    
    async def _process_medical(self, sections: List[TextSection]) -> List[TextSection]:
        """Medical document-specific processing."""
        # Boost importance of medical information
        for section in sections:
            if re.search(r'(diagnosis|treatment|medication|patient|doctor)', section.text.lower()):
                section.importance = min(1.0, section.importance + 0.3)
        return sections
    
    async def _process_purchase_order(self, sections: List[TextSection]) -> List[TextSection]:
        """Purchase order-specific processing."""
        # Boost importance of order details
        for section in sections:
            if re.search(r'(order|quantity|price|item|delivery)', section.text.lower()):
                section.importance = min(1.0, section.importance + 0.3)
        return sections
    
    async def chunk_text(self, text: str, doc_id: str, doc_type: str) -> List[DocumentChunk]:
        """Simple text chunking method for testing."""
        try:
            # Convert string doc_type to DocumentType enum
            doc_type_enum = DocumentType.OTHER
            if doc_type.lower() in ['invoice', 'resume', 'legal', 'medical', 'purchase_order']:
                doc_type_enum = DocumentType[doc_type.upper()]
            
            # Use the main processing method with minimal layout info
            chunks = await self.process_and_chunk(text, doc_type_enum, doc_id=doc_id)
            return chunks
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {e}")
            # Return a simple chunk as fallback
            return [DocumentChunk(
                id=f"{doc_id}_chunk_0",
                content=text[:self.config.chunk_size],
                metadata={"doc_id": doc_id, "chunk_index": 0}
            )]
