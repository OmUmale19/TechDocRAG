"""
Advanced field extraction using hybrid rule-based and semantic approaches.
Extracts structured fields from documents with high accuracy and confidence scoring.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.types import ExtractedField, DocumentType, DocumentChunk, BoundingBox
from ..core.config import Config
from ..utils.logging import get_logger, performance_log
from ..utils.helpers import extract_numbers, extract_dates, is_valid_email, is_valid_phone
from ..utils.exceptions import ProcessingError

logger = get_logger(__name__)


@dataclass
class ExtractionRule:
    """Rule for extracting a specific field."""
    field_name: str
    patterns: List[str]
    data_type: str  # text, number, date, email, phone
    confidence_boost: float = 0.0
    validation_func: Optional[callable] = None


class FieldExtractor:
    """
    Hybrid field extraction system combining rule-based and semantic approaches.
    
    Architecture:
    1. Rule-based extraction for structured fields (dates, amounts, IDs)
    2. Semantic extraction for unstructured content using LLM
    3. Confidence scoring and validation
    4. Domain-specific extraction rules
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize extraction rules for different document types
        self.extraction_rules = self._initialize_extraction_rules()
        
        # LLM client for semantic extraction
        self.llm_client = self._initialize_llm()
        
        logger.info("FieldExtractor initialized with rule-based and semantic capabilities")
    
    @performance_log("field_extraction")
    async def extract_fields(
        self,
        raw_text: str,
        chunks: List[DocumentChunk],
        doc_type: DocumentType,
        layout_elements: List[Dict] = None
    ) -> List[ExtractedField]:
        """
        Extract structured fields from document using hybrid approach.
        
        Args:
            raw_text: Full document text
            chunks: Processed document chunks
            doc_type: Document type for specialized extraction
            layout_elements: Layout analysis results
            
        Returns:
            List of extracted fields with confidence scores
        """
        logger.info(f"Extracting fields for document type: {doc_type}")
        
        try:
            extracted_fields = []
            
            # Step 1: Rule-based extraction for structured fields
            rule_fields = await self._rule_based_extraction(raw_text, doc_type, layout_elements)
            extracted_fields.extend(rule_fields)
            
            # Step 2: Semantic extraction for complex fields
            semantic_fields = await self._semantic_extraction(chunks, doc_type)
            extracted_fields.extend(semantic_fields)
            
            # Step 3: Layout-based extraction using position information
            if layout_elements:
                layout_fields = await self._layout_based_extraction(layout_elements, doc_type)
                extracted_fields.extend(layout_fields)
            
            # Step 4: Validate and deduplicate fields
            validated_fields = self._validate_and_deduplicate(extracted_fields)
            
            # Step 5: Calculate final confidence scores
            final_fields = self._calculate_confidence_scores(validated_fields, doc_type)
            
            logger.info(f"Extracted {len(final_fields)} fields with average confidence: "
                       f"{sum(f.confidence for f in final_fields) / len(final_fields) if final_fields else 0:.2f}")
            
            return final_fields
            
        except Exception as e:
            logger.error(f"Field extraction failed: {str(e)}")
            raise ProcessingError(f"Field extraction failed: {str(e)}") from e
    
    async def _rule_based_extraction(
        self, 
        text: str, 
        doc_type: DocumentType,
        layout_elements: List[Dict] = None
    ) -> List[ExtractedField]:
        """Extract fields using predefined rules and patterns."""
        fields = []
        
        # Get rules for this document type
        rules = self.extraction_rules.get(doc_type, [])
        rules.extend(self.extraction_rules.get('common', []))
        
        for rule in rules:
            field_values = self._apply_extraction_rule(text, rule)
            
            for value, confidence, position in field_values:
                field = ExtractedField(
                    name=rule.field_name,
                    value=value,
                    confidence=confidence + rule.confidence_boost,
                    extraction_method="rule_based"
                )
                
                # Add position information if available
                if position and layout_elements:
                    field.bbox = self._find_bbox_for_position(position, layout_elements)
                
                fields.append(field)
        
        return fields
    
    def _apply_extraction_rule(self, text: str, rule: ExtractionRule) -> List[Tuple[Any, float, Optional[int]]]:
        """Apply a single extraction rule to text."""
        results = []
        
        for pattern in rule.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                raw_value = match.group(1) if match.groups() else match.group(0)
                
                # Process value based on data type
                processed_value, confidence = self._process_field_value(raw_value, rule.data_type)
                
                if processed_value is not None:
                    # Validate if validation function provided
                    if rule.validation_func and not rule.validation_func(processed_value):
                        continue
                    
                    results.append((processed_value, confidence, match.start()))
        
        return results
    
    def _process_field_value(self, raw_value: str, data_type: str) -> Tuple[Any, float]:
        """Process and validate field value based on data type."""
        raw_value = raw_value.strip()
        
        if data_type == 'text':
            return raw_value, 0.9
        
        elif data_type == 'number':
            try:
                # Remove common formatting
                clean_value = re.sub(r'[,$\s]', '', raw_value)
                if '.' in clean_value:
                    value = float(clean_value)
                else:
                    value = int(clean_value)
                return value, 0.95
            except ValueError:
                return None, 0.0
        
        elif data_type == 'date':
            parsed_date = self._parse_date(raw_value)
            if parsed_date:
                return parsed_date.isoformat(), 0.9
            return None, 0.0
        
        elif data_type == 'email':
            if is_valid_email(raw_value):
                return raw_value.lower(), 0.95
            return None, 0.0
        
        elif data_type == 'phone':
            if is_valid_phone(raw_value):
                # Normalize phone number
                digits = re.sub(r'\D', '', raw_value)
                return digits, 0.9
            return None, 0.0
        
        else:
            return raw_value, 0.7
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using multiple formats."""
        date_formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d',
            '%m-%d-%Y', '%d-%m-%Y', '%B %d, %Y',
            '%d %B %Y', '%b %d, %Y', '%d %b %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    async def _semantic_extraction(self, chunks: List[DocumentChunk], doc_type: DocumentType) -> List[ExtractedField]:
        """Extract fields using semantic understanding with LLM."""
        if not self.llm_client:
            logger.warning("LLM not available for semantic extraction")
            return []
        
        fields = []
        
        try:
            # Prepare context for LLM
            context = self._prepare_semantic_context(chunks, doc_type)
            
            # Get extraction prompt for document type
            prompt = self._get_extraction_prompt(doc_type, context)
            
            # Query LLM for field extraction
            response = await self._query_llm_for_extraction(prompt)
            
            # Parse LLM response into structured fields
            semantic_fields = self._parse_llm_response(response, doc_type)
            fields.extend(semantic_fields)
            
        except Exception as e:
            logger.warning(f"Semantic extraction failed: {str(e)}")
        
        return fields
    
    def _prepare_semantic_context(self, chunks: List[DocumentChunk], doc_type: DocumentType) -> str:
        """Prepare context for semantic extraction."""
        # Select most relevant chunks for context
        relevant_chunks = sorted(chunks, key=lambda x: x.metadata.get('importance', 0), reverse=True)[:5]
        
        context_parts = [f"Chunk {i+1}: {chunk.content}" for i, chunk in enumerate(relevant_chunks)]
        
        return "\n\n".join(context_parts)
    
    def _get_extraction_prompt(self, doc_type: DocumentType, context: str) -> str:
        """Generate extraction prompt for specific document type."""
        base_prompt = f"""
        Analyze the following {doc_type.value} document and extract structured information.
        
        Document content:
        {context}
        
        Extract the following information in JSON format:
        """
        
        if doc_type == DocumentType.INVOICE:
            prompt = base_prompt + """
            {
                "invoice_number": "string",
                "invoice_date": "YYYY-MM-DD",
                "due_date": "YYYY-MM-DD",
                "vendor_name": "string",
                "customer_name": "string",
                "total_amount": "number",
                "tax_amount": "number",
                "subtotal": "number",
                "currency": "string",
                "payment_terms": "string"
            }
            """
        
        elif doc_type == DocumentType.RESUME:
            prompt = base_prompt + """
            {
                "candidate_name": "string",
                "email": "string",
                "phone": "string",
                "years_experience": "number",
                "education_level": "string",
                "skills": ["list of skills"],
                "current_position": "string",
                "companies": ["list of companies"]
            }
            """
        
        elif doc_type == DocumentType.LEGAL_CONTRACT:
            prompt = base_prompt + """
            {
                "contract_type": "string",
                "parties": ["list of parties"],
                "effective_date": "YYYY-MM-DD",
                "expiration_date": "YYYY-MM-DD",
                "contract_value": "number",
                "jurisdiction": "string",
                "key_terms": ["list of key terms"]
            }
            """
        
        else:
            prompt = base_prompt + """
            {
                "document_title": "string",
                "date": "YYYY-MM-DD",
                "author": "string",
                "key_entities": ["list of entities"],
                "important_numbers": ["list of numbers"],
                "contact_info": ["list of contact information"]
            }
            """
        
        prompt += "\n\nReturn only valid JSON. If information is not available, use null."
        
        return prompt
    
    async def _query_llm_for_extraction(self, prompt: str) -> str:
        """Query LLM for field extraction."""
        try:
            if self.config.llm.provider == "openai":
                import openai
                
                client = openai.OpenAI(api_key=self.config.llm.api_key)
                response = client.chat.completions.create(
                    model=self.config.llm.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert document analyzer. Extract structured information accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            
            else:
                logger.warning(f"LLM provider {self.config.llm.provider} not supported")
                return "{}"
                
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return "{}"
    
    def _parse_llm_response(self, response: str, doc_type: DocumentType) -> List[ExtractedField]:
        """Parse LLM response into structured fields."""
        fields = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return fields
            
            data = json.loads(json_match.group(0))
            
            for field_name, value in data.items():
                if value is not None and value != "":
                    field = ExtractedField(
                        name=field_name,
                        value=value,
                        confidence=0.8,  # Base confidence for LLM extraction
                        extraction_method="semantic"
                    )
                    fields.append(field)
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {str(e)}")
        
        return fields
    
    async def _layout_based_extraction(self, layout_elements: List[Dict], doc_type: DocumentType) -> List[ExtractedField]:
        """Extract fields using layout and position information."""
        fields = []
        
        for element in layout_elements:
            element_type = element.get('type', '')
            text = element.get('text', '')
            
            if element_type == 'table' and text:
                # Extract table data
                table_fields = self._extract_table_fields(text, doc_type)
                fields.extend(table_fields)
            
            elif element_type == 'header' and text:
                # Extract header information
                header_fields = self._extract_header_fields(text, doc_type)
                fields.extend(header_fields)
        
        return fields
    
    def _extract_table_fields(self, table_text: str, doc_type: DocumentType) -> List[ExtractedField]:
        """Extract structured data from table text."""
        fields = []
        
        try:
            rows = table_text.split('\n')
            if len(rows) < 2:
                return fields
            
            # Assume first row is header
            headers = [h.strip() for h in rows[0].split('\t')]
            
            for row_text in rows[1:]:
                cells = [c.strip() for c in row_text.split('\t')]
                
                # Map cells to headers
                for i, (header, cell) in enumerate(zip(headers, cells)):
                    if cell and header:
                        # Determine field type based on content
                        if re.match(r'^\d+\.?\d*$', cell):
                            value = float(cell) if '.' in cell else int(cell)
                            data_type = 'number'
                        else:
                            value = cell
                            data_type = 'text'
                        
                        field = ExtractedField(
                            name=f"table_{header.lower().replace(' ', '_')}",
                            value=value,
                            confidence=0.85,
                            extraction_method="layout_table"
                        )
                        fields.append(field)
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
        
        return fields
    
    def _extract_header_fields(self, header_text: str, doc_type: DocumentType) -> List[ExtractedField]:
        """Extract information from header elements."""
        fields = []
        
        # Common header patterns
        patterns = {
            'title': r'^([A-Z][A-Za-z\s]+)$',
            'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            'reference': r'(REF|Ref|Reference):\s*([A-Za-z0-9-]+)',
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, header_text)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                
                field = ExtractedField(
                    name=f"header_{field_name}",
                    value=value.strip(),
                    confidence=0.8,
                    extraction_method="layout_header"
                )
                fields.append(field)
        
        return fields
    
    def _validate_and_deduplicate(self, fields: List[ExtractedField]) -> List[ExtractedField]:
        """Validate and remove duplicate fields."""
        validated_fields = []
        seen_fields = {}
        
        for field in fields:
            # Create unique key for deduplication
            key = f"{field.name}_{str(field.value)}"
            
            if key in seen_fields:
                # Keep field with higher confidence
                existing_field = seen_fields[key]
                if field.confidence > existing_field.confidence:
                    seen_fields[key] = field
            else:
                seen_fields[key] = field
        
        # Additional validation
        for field in seen_fields.values():
            if self._validate_field(field):
                validated_fields.append(field)
        
        return validated_fields
    
    def _validate_field(self, field: ExtractedField) -> bool:
        """Validate individual field."""
        # Check confidence threshold
        if field.confidence < 0.3:
            return False
        
        # Check value is not empty
        if field.value is None or (isinstance(field.value, str) and not field.value.strip()):
            return False
        
        # Type-specific validation
        if 'email' in field.name.lower():
            return is_valid_email(str(field.value))
        
        if 'phone' in field.name.lower():
            return is_valid_phone(str(field.value))
        
        if 'date' in field.name.lower():
            try:
                datetime.fromisoformat(str(field.value))
                return True
            except:
                return False
        
        return True
    
    def _calculate_confidence_scores(self, fields: List[ExtractedField], doc_type: DocumentType) -> List[ExtractedField]:
        """Calculate final confidence scores with contextual adjustments."""
        for field in fields:
            # Base confidence from extraction method
            base_confidence = field.confidence
            
            # Boost confidence for document-type specific fields
            if self._is_key_field_for_type(field.name, doc_type):
                base_confidence = min(1.0, base_confidence + 0.1)
            
            # Reduce confidence for generic fields
            if 'header_' in field.name or 'table_' in field.name:
                base_confidence = max(0.5, base_confidence - 0.1)
            
            # Cross-validation boost if same field extracted by multiple methods
            method_count = len([f for f in fields if f.name == field.name])
            if method_count > 1:
                base_confidence = min(1.0, base_confidence + 0.05)
            
            field.confidence = base_confidence
        
        return fields
    
    def _is_key_field_for_type(self, field_name: str, doc_type: DocumentType) -> bool:
        """Check if field is key for document type."""
        key_fields = {
            DocumentType.INVOICE: ['invoice_number', 'total_amount', 'due_date', 'vendor_name'],
            DocumentType.RESUME: ['candidate_name', 'email', 'years_experience', 'skills'],
            DocumentType.LEGAL_CONTRACT: ['contract_type', 'parties', 'effective_date'],
        }
        
        return field_name in key_fields.get(doc_type, [])
    
    def _find_bbox_for_position(self, position: int, layout_elements: List[Dict]) -> Optional[BoundingBox]:
        """Find bounding box for text position."""
        # This is a simplified implementation
        # In practice, would need more sophisticated position mapping
        for element in layout_elements:
            if element.get('bbox') and isinstance(element['bbox'], BoundingBox):
                return element['bbox']
        return None
    
    def _initialize_extraction_rules(self) -> Dict[str, List[ExtractionRule]]:
        """Initialize extraction rules for different document types."""
        rules = {
            'common': [
                ExtractionRule(
                    field_name='email',
                    patterns=[r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'],
                    data_type='email',
                    validation_func=is_valid_email
                ),
                ExtractionRule(
                    field_name='phone',
                    patterns=[r'\b(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'],
                    data_type='phone',
                    validation_func=is_valid_phone
                ),
                ExtractionRule(
                    field_name='date',
                    patterns=[
                        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                        r'\b(\d{4}-\d{1,2}-\d{1,2})\b'
                    ],
                    data_type='date'
                ),
            ],
            
            DocumentType.INVOICE: [
                ExtractionRule(
                    field_name='invoice_number',
                    patterns=[
                        r'Invoice\s*#?\s*:?\s*([A-Za-z0-9-]+)',
                        r'Invoice\s*Number\s*:?\s*([A-Za-z0-9-]+)'
                    ],
                    data_type='text',
                    confidence_boost=0.2
                ),
                ExtractionRule(
                    field_name='total_amount',
                    patterns=[
                        r'Total\s*:?\s*\$?\s*([0-9,]+\.?\d*)',
                        r'Amount\s*Due\s*:?\s*\$?\s*([0-9,]+\.?\d*)'
                    ],
                    data_type='number',
                    confidence_boost=0.3
                ),
                ExtractionRule(
                    field_name='due_date',
                    patterns=[r'Due\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'],
                    data_type='date',
                    confidence_boost=0.2
                ),
            ],
            
            DocumentType.RESUME: [
                ExtractionRule(
                    field_name='years_experience',
                    patterns=[r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience'],
                    data_type='number',
                    confidence_boost=0.2
                ),
                ExtractionRule(
                    field_name='education_degree',
                    patterns=[
                        r'(Bachelor|Master|PhD|B\.?A\.?|M\.?A\.?|B\.?S\.?|M\.?S\.?)',
                        r'(Associates?|Doctorate)'
                    ],
                    data_type='text',
                    confidence_boost=0.2
                ),
            ],
            
            DocumentType.LEGAL_CONTRACT: [
                ExtractionRule(
                    field_name='effective_date',
                    patterns=[r'Effective\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'],
                    data_type='date',
                    confidence_boost=0.2
                ),
                ExtractionRule(
                    field_name='contract_value',
                    patterns=[r'Contract\s*Value\s*:?\s*\$?\s*([0-9,]+\.?\d*)'],
                    data_type='number',
                    confidence_boost=0.2
                ),
            ]
        }
        
        return rules
    
    def _initialize_llm(self):
        """Initialize LLM client for semantic extraction."""
        try:
            if self.config.llm.provider in ["openai", "gemini", "anthropic"]:
                logger.info(f"LLM configured: {self.config.llm.provider}")
                return True  # LLM available
            else:
                logger.info("Using rule-based extraction (LLM optional for field extraction)")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {str(e)}")
            return None
