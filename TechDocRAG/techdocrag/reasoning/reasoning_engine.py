"""
Advanced reasoning engine for TechDocRAG.
Implements multi-step reasoning chains with rule-based and LLM-based inference.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from ..core.types import (
    DocumentResult, DocumentChunk, QueryResponse, ConfidenceScore,
    ReasoningStep, CalculationResult
)
from ..core.config import Config
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import ReasoningError
from .calculation_engine import CalculationEngine
from .confidence_calculator import ConfidenceCalculator

logger = get_logger(__name__)


class ReasoningEngine:
    """
    Multi-modal reasoning engine combining rule-based and neural approaches.
    
    Reasoning Pipeline:
    1. Query Analysis - Determine reasoning requirements
    2. Rule-based Reasoning - Apply domain-specific rules
    3. LLM Reasoning - Use language models for complex inference
    4. Calculation Engine - Handle numerical computations
    5. Multi-document Synthesis - Combine evidence across documents
    6. Confidence Assessment - Calculate reasoning confidence
    
    Features:
    - Chain-of-thought reasoning
    - Multi-document evidence synthesis
    - Numerical computation and validation
    - Explainable reasoning steps
    - Domain-specific rule sets
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize sub-components
        self.calculation_engine = CalculationEngine(config.calculation)
        self.confidence_calculator = ConfidenceCalculator(config.confidence)
        
        # LLM client for reasoning
        self.llm_client = self._initialize_llm_client()
        
        # Domain-specific reasoning rules
        self.reasoning_rules = self._load_reasoning_rules()
        
        # Reasoning statistics
        self.stats = {
            'total_reasoning_requests': 0,
            'rule_based_resolutions': 0,
            'llm_reasoning_requests': 0,
            'calculation_requests': 0,
            'multi_document_syntheses': 0
        }
        
        logger.info("ReasoningEngine initialized with rule-based and LLM capabilities")
    
    def _initialize_llm_client(self):
        """Initialize LLM client for reasoning."""
        try:
            if self.config.llm.provider.lower() == 'openai':
                import openai
                return openai.OpenAI(api_key=self.config.llm.openai.api_key)
            elif self.config.llm.provider.lower() == 'anthropic':
                import anthropic
                return anthropic.Anthropic(api_key=self.config.llm.anthropic.api_key)
            elif self.config.llm.provider.lower() == 'gemini':
                # Gemini is handled separately in AnswerSynthesizer/QueryEngine
                logger.info("Gemini LLM configured (handled by AnswerSynthesizer)")
                return None
            else:
                logger.warning(f"Unsupported LLM provider: {self.config.llm.provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            return None
    
    def _load_reasoning_rules(self) -> Dict[str, List[Dict]]:
        """Load domain-specific reasoning rules."""
        return {
            'invoice': [
                {
                    'name': 'total_calculation',
                    'pattern': r'(?:total|sum|amount due)',
                    'action': 'calculate_invoice_total',
                    'priority': 1
                },
                {
                    'name': 'tax_validation',
                    'pattern': r'(?:tax|vat|gst)',
                    'action': 'validate_tax_calculation',
                    'priority': 2
                },
                {
                    'name': 'due_date_reasoning',
                    'pattern': r'(?:due date|payment|deadline)',
                    'action': 'analyze_payment_terms',
                    'priority': 1
                }
            ],
            'resume': [
                {
                    'name': 'experience_calculation',
                    'pattern': r'(?:years?\s+of\s+experience|experience)',
                    'action': 'calculate_total_experience',
                    'priority': 1
                },
                {
                    'name': 'skill_matching',
                    'pattern': r'(?:skills?|competenc|abilit)',
                    'action': 'match_skills_to_requirements',
                    'priority': 2
                },
                {
                    'name': 'education_analysis',
                    'pattern': r'(?:education|degree|qualification)',
                    'action': 'analyze_educational_background',
                    'priority': 2
                }
            ],
            'legal': [
                {
                    'name': 'clause_interpretation',
                    'pattern': r'(?:clause|section|article)',
                    'action': 'interpret_legal_clause',
                    'priority': 1
                },
                {
                    'name': 'obligation_analysis',
                    'pattern': r'(?:obligation|responsibility|duty)',
                    'action': 'analyze_obligations',
                    'priority': 1
                },
                {
                    'name': 'term_extraction',
                    'pattern': r'(?:term|condition|requirement)',
                    'action': 'extract_contract_terms',
                    'priority': 2
                }
            ]
        }
    
    @performance_log("reasoning_process")
    async def reason(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        document_context: List[DocumentResult] = None
    ) -> QueryResponse:
        """
        Execute multi-step reasoning process.
        
        Args:
            query: User query requiring reasoning
            retrieved_chunks: Relevant document chunks
            document_context: Full document context if available
            
        Returns:
            QueryResponse with reasoning results
        """
        self.stats['total_reasoning_requests'] += 1
        
        try:
            logger.info(f"Starting reasoning process for query: {query[:50]}...")
            
            # Step 1: Analyze query to determine reasoning strategy
            reasoning_strategy = await self._analyze_query_requirements(query, retrieved_chunks)
            
            # Step 2: Execute reasoning chain
            reasoning_steps = []
            
            # Try rule-based reasoning first
            rule_result = await self._apply_rule_based_reasoning(
                query, retrieved_chunks, reasoning_strategy
            )
            
            if rule_result['success']:
                reasoning_steps.extend(rule_result['steps'])
                self.stats['rule_based_resolutions'] += 1
                
                # Validate with calculation engine if needed
                if reasoning_strategy.get('requires_calculation'):
                    calc_result = await self._handle_calculations(
                        query, reasoning_steps, retrieved_chunks
                    )
                    reasoning_steps.extend(calc_result['steps'])
            else:
                # Fall back to LLM reasoning
                llm_result = await self._apply_llm_reasoning(
                    query, retrieved_chunks, reasoning_strategy
                )
                reasoning_steps.extend(llm_result['steps'])
                self.stats['llm_reasoning_requests'] += 1
            
            # Step 3: Multi-document synthesis if needed
            if len(set(chunk.get('doc_id') for chunk in retrieved_chunks)) > 1:
                synthesis_result = await self._synthesize_multi_document_evidence(
                    query, reasoning_steps, retrieved_chunks
                )
                reasoning_steps.extend(synthesis_result['steps'])
                self.stats['multi_document_syntheses'] += 1
            
            # Step 4: Generate final answer
            final_answer = await self._generate_final_answer(
                query, reasoning_steps, retrieved_chunks
            )
            
            # Step 5: Calculate confidence
            confidence = await self.confidence_calculator.calculate_reasoning_confidence(
                query, reasoning_steps, retrieved_chunks
            )
            
            # Step 6: Prepare response
            response = QueryResponse(
                query=query,
                answer=final_answer,
                confidence_score=confidence,
                reasoning_steps=reasoning_steps,
                source_documents=[chunk.get('doc_id') for chunk in retrieved_chunks],
                retrieved_chunks=retrieved_chunks,
                processing_time=0.0,  # Will be set by caller
                metadata={
                    'reasoning_strategy': reasoning_strategy,
                    'rule_based_success': rule_result['success'],
                    'multi_document_synthesis': len(set(chunk.get('doc_id') for chunk in retrieved_chunks)) > 1
                }
            )
            
            logger.info(f"Reasoning completed with confidence: {confidence.overall_confidence:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Reasoning process failed: {str(e)}")
            raise ReasoningError(f"Reasoning failed: {str(e)}") from e
    
    async def _analyze_query_requirements(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze query to determine reasoning requirements."""
        
        # Detect query patterns
        requires_calculation = bool(re.search(
            r'\b(?:total|sum|calculate|add|subtract|multiply|divide|how much|how many|percentage)\b',
            query.lower()
        ))
        
        requires_comparison = bool(re.search(
            r'\b(?:compare|vs|versus|difference|better|higher|lower|more|less)\b',
            query.lower()
        ))
        
        requires_temporal_reasoning = bool(re.search(
            r'\b(?:before|after|during|since|until|when|date|time|period)\b',
            query.lower()
        ))
        
        requires_aggregation = bool(re.search(
            r'\b(?:all|every|each|list|summarize|overview|combined)\b',
            query.lower()
        ))
        
        # Determine document types
        doc_types = set()
        for chunk in chunks:
            if chunk.get('doc_type'):
                doc_types.add(chunk['doc_type'])
        
        # Determine reasoning complexity
        complexity = 'simple'
        if requires_calculation or requires_comparison:
            complexity = 'moderate'
        if requires_temporal_reasoning and requires_aggregation:
            complexity = 'complex'
        
        return {
            'requires_calculation': requires_calculation,
            'requires_comparison': requires_comparison,
            'requires_temporal_reasoning': requires_temporal_reasoning,
            'requires_aggregation': requires_aggregation,
            'document_types': list(doc_types),
            'complexity': complexity,
            'multi_document': len(set(chunk.get('doc_id') for chunk in chunks)) > 1
        }
    
    async def _apply_rule_based_reasoning(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply domain-specific reasoning rules."""
        
        reasoning_steps = []
        success = False
        
        try:
            # Get relevant rules based on document types
            applicable_rules = []
            for doc_type in strategy.get('document_types', []):
                if doc_type in self.reasoning_rules:
                    applicable_rules.extend(self.reasoning_rules[doc_type])
            
            # Sort rules by priority
            applicable_rules.sort(key=lambda x: x.get('priority', 999))
            
            # Apply rules
            for rule in applicable_rules:
                if re.search(rule['pattern'], query.lower()):
                    logger.debug(f"Applying rule: {rule['name']}")
                    
                    rule_result = await self._execute_rule(rule, query, chunks)
                    
                    if rule_result['success']:
                        reasoning_steps.append(ReasoningStep(
                            step_type='rule_application',
                            description=f"Applied rule: {rule['name']}",
                            input_data={'rule': rule, 'query': query},
                            output_data=rule_result['output'],
                            confidence=rule_result.get('confidence', 0.8),
                            sources=[chunk.get('doc_id') for chunk in chunks]
                        ))
                        success = True
                        break  # Use first successful rule
            
        except Exception as e:
            logger.warning(f"Rule-based reasoning failed: {str(e)}")
        
        return {
            'success': success,
            'steps': reasoning_steps
        }
    
    async def _execute_rule(
        self,
        rule: Dict[str, Any],
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a specific reasoning rule."""
        
        action = rule['action']
        
        try:
            if action == 'calculate_invoice_total':
                return await self._calculate_invoice_total(chunks)
            elif action == 'validate_tax_calculation':
                return await self._validate_tax_calculation(chunks)
            elif action == 'analyze_payment_terms':
                return await self._analyze_payment_terms(chunks)
            elif action == 'calculate_total_experience':
                return await self._calculate_total_experience(chunks)
            elif action == 'match_skills_to_requirements':
                return await self._match_skills_to_requirements(query, chunks)
            elif action == 'analyze_educational_background':
                return await self._analyze_educational_background(chunks)
            elif action == 'interpret_legal_clause':
                return await self._interpret_legal_clause(query, chunks)
            elif action == 'analyze_obligations':
                return await self._analyze_obligations(chunks)
            elif action == 'extract_contract_terms':
                return await self._extract_contract_terms(chunks)
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _calculate_invoice_total(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate invoice total from line items."""
        try:
            # Extract numerical values and line items
            line_items = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for line item patterns
                line_item_patterns = [
                    r'(\d+(?:\.\d{2})?)\s*(?:each|per|@)',  # Unit prices
                    r'quantity\s*:?\s*(\d+)',                 # Quantities
                    r'total\s*:?\s*\$?(\d+(?:\.\d{2})?)',    # Totals
                ]
                
                for pattern in line_item_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        try:
                            value = float(match)
                            line_items.append(value)
                        except ValueError:
                            continue
            
            if not line_items:
                return {'success': False, 'error': 'No numerical values found'}
            
            # Calculate total
            calculated_total = sum(line_items)
            
            return {
                'success': True,
                'output': {
                    'calculated_total': calculated_total,
                    'line_items': line_items,
                    'calculation_method': 'sum_of_line_items'
                },
                'confidence': 0.9 if len(line_items) > 2 else 0.7
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _validate_tax_calculation(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate tax calculations in invoice."""
        try:
            # Extract subtotal, tax rate, and tax amount
            subtotal = None
            tax_rate = None
            tax_amount = None
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for subtotal
                subtotal_match = re.search(r'subtotal\s*:?\s*\$?(\d+(?:\.\d{2})?)', content, re.IGNORECASE)
                if subtotal_match:
                    subtotal = float(subtotal_match.group(1))
                
                # Look for tax rate
                tax_rate_match = re.search(r'tax\s*(?:rate)?\s*:?\s*(\d+(?:\.\d+)?)\s*%', content, re.IGNORECASE)
                if tax_rate_match:
                    tax_rate = float(tax_rate_match.group(1)) / 100
                
                # Look for tax amount
                tax_amount_match = re.search(r'tax\s*(?:amount)?\s*:?\s*\$?(\d+(?:\.\d{2})?)', content, re.IGNORECASE)
                if tax_amount_match:
                    tax_amount = float(tax_amount_match.group(1))
            
            if subtotal and tax_rate:
                calculated_tax = subtotal * tax_rate
                
                validation_result = {
                    'subtotal': subtotal,
                    'tax_rate': tax_rate,
                    'calculated_tax': calculated_tax,
                    'stated_tax': tax_amount,
                    'validation_passed': abs(calculated_tax - (tax_amount or 0)) < 0.01 if tax_amount else None
                }
                
                return {
                    'success': True,
                    'output': validation_result,
                    'confidence': 0.9
                }
            
            return {'success': False, 'error': 'Insufficient tax information found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _analyze_payment_terms(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze payment terms and due dates."""
        try:
            payment_terms = {}
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for due date patterns
                due_date_patterns = [
                    r'due\s+(?:date\s*:?\s*)?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                    r'payment\s+due\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                    r'net\s+(\d+)\s+days?'
                ]
                
                for pattern in due_date_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        payment_terms['due_date_info'] = match.group(1)
                        break
                
                # Look for payment methods
                payment_method_patterns = [
                    r'payment\s+method\s*:?\s*([^\n]+)',
                    r'pay\s+by\s*:?\s*([^\n]+)',
                ]
                
                for pattern in payment_method_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        payment_terms['payment_method'] = match.group(1).strip()
                        break
            
            if payment_terms:
                return {
                    'success': True,
                    'output': payment_terms,
                    'confidence': 0.8
                }
            
            return {'success': False, 'error': 'No payment terms found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _calculate_total_experience(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total years of experience from resume."""
        try:
            experience_periods = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for experience patterns
                experience_patterns = [
                    r'(\d+)\s+years?\s+(?:of\s+)?experience',
                    r'(\d{4})\s*[-–]\s*(\d{4}|\w+)',  # Date ranges
                    r'(\d{1,2})/(\d{4})\s*[-–]\s*(\d{1,2})/(\d{4})',  # Month/Year ranges
                ]
                
                for pattern in experience_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 1:  # Direct years mentioned
                            try:
                                years = int(match[0])
                                experience_periods.append(years)
                            except ValueError:
                                continue
                        elif len(match) == 2:  # Year range
                            try:
                                start_year = int(match[0])
                                end_year = int(match[1]) if match[1].isdigit() else datetime.now().year
                                years = end_year - start_year
                                experience_periods.append(years)
                            except ValueError:
                                continue
            
            if experience_periods:
                total_experience = sum(experience_periods)
                
                return {
                    'success': True,
                    'output': {
                        'total_years': total_experience,
                        'experience_periods': experience_periods,
                        'calculation_method': 'sum_of_periods'
                    },
                    'confidence': 0.8
                }
            
            return {'success': False, 'error': 'No experience information found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _match_skills_to_requirements(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Match skills mentioned in documents to query requirements."""
        try:
            # Extract skills from query (requirements)
            query_skills = set()
            
            # Common skill patterns in queries
            skill_indicators = ['skill', 'experience', 'knowledge', 'proficient', 'familiar']
            
            for indicator in skill_indicators:
                if indicator in query.lower():
                    # Extract potential skills after indicators
                    pattern = rf'{indicator}\s+(?:in\s+|with\s+)?([^.!?]+)'
                    matches = re.findall(pattern, query.lower())
                    for match in matches:
                        query_skills.update(match.strip().split(','))
            
            # Extract skills from documents
            document_skills = set()
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Common skill patterns in resumes
                skill_patterns = [
                    r'skills?\s*:?\s*([^\n]+)',
                    r'technologies?\s*:?\s*([^\n]+)',
                    r'programming\s+languages?\s*:?\s*([^\n]+)',
                    r'experience\s+with\s*:?\s*([^\n]+)',
                ]
                
                for pattern in skill_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        skills = [skill.strip() for skill in match.split(',')]
                        document_skills.update(skills)
            
            # Find matches
            matched_skills = []
            for query_skill in query_skills:
                for doc_skill in document_skills:
                    if query_skill.lower().strip() in doc_skill.lower() or doc_skill.lower().strip() in query_skill.lower():
                        matched_skills.append({
                            'required': query_skill.strip(),
                            'found': doc_skill.strip()
                        })
            
            if matched_skills or document_skills:
                return {
                    'success': True,
                    'output': {
                        'matched_skills': matched_skills,
                        'all_document_skills': list(document_skills),
                        'required_skills': list(query_skills),
                        'match_percentage': len(matched_skills) / max(len(query_skills), 1) * 100
                    },
                    'confidence': 0.7
                }
            
            return {'success': False, 'error': 'No skills found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _analyze_educational_background(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze educational background from resume."""
        try:
            education_info = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for education patterns
                education_patterns = [
                    r'(bachelor|master|phd|doctorate)\s+(?:of\s+|in\s+)?([^\n,]+)',
                    r'(b\.?\s*[as]\.?|m\.?\s*[as]\.?|ph\.?\s*d\.?)\s+(?:in\s+)?([^\n,]+)',
                    r'degree\s+in\s+([^\n,]+)',
                    r'graduated\s+(?:from\s+)?([^\n,]+)',
                ]
                
                for pattern in education_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 2:
                            education_info.append({
                                'degree_type': match[0].strip(),
                                'field_of_study': match[1].strip()
                            })
                        else:
                            education_info.append({
                                'education': match[0].strip()
                            })
            
            if education_info:
                return {
                    'success': True,
                    'output': {
                        'education': education_info,
                        'highest_degree': education_info[0] if education_info else None
                    },
                    'confidence': 0.8
                }
            
            return {'success': False, 'error': 'No education information found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _interpret_legal_clause(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Interpret legal clauses (simplified implementation)."""
        try:
            # This would typically use specialized legal NLP models
            # For now, provide basic clause identification
            
            clauses_found = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for clause patterns
                clause_patterns = [
                    r'(clause\s+\d+[a-z]?)\s*:?\s*([^\n]+)',
                    r'(section\s+\d+(?:\.\d+)?)\s*:?\s*([^\n]+)',
                    r'(article\s+\d+)\s*:?\s*([^\n]+)',
                ]
                
                for pattern in clause_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        clauses_found.append({
                            'clause_id': match[0],
                            'content': match[1].strip()
                        })
            
            if clauses_found:
                return {
                    'success': True,
                    'output': {
                        'clauses': clauses_found,
                        'interpretation': 'Basic clause identification completed'
                    },
                    'confidence': 0.6  # Lower confidence for simplified implementation
                }
            
            return {'success': False, 'error': 'No clauses found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _analyze_obligations(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze obligations and responsibilities in legal documents."""
        try:
            obligations = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for obligation patterns
                obligation_patterns = [
                    r'(shall|must|will|agrees? to)\s+([^\n.]+)',
                    r'(responsible for|obligation to|duty to)\s+([^\n.]+)',
                    r'(party\s+[AB]?)\s+(shall|must|will)\s+([^\n.]+)',
                ]
                
                for pattern in obligation_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 2:
                            obligations.append({
                                'obligation_type': match[0],
                                'description': match[1].strip()
                            })
                        elif len(match) == 3:
                            obligations.append({
                                'party': match[0],
                                'obligation_type': match[1],
                                'description': match[2].strip()
                            })
            
            if obligations:
                return {
                    'success': True,
                    'output': {
                        'obligations': obligations,
                        'total_obligations': len(obligations)
                    },
                    'confidence': 0.7
                }
            
            return {'success': False, 'error': 'No obligations found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _extract_contract_terms(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key contract terms and conditions."""
        try:
            terms = {}
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for common contract terms
                term_patterns = [
                    (r'term\s*:?\s*(\d+)\s+(years?|months?|days?)', 'contract_duration'),
                    (r'effective\s+date\s*:?\s*([^\n]+)', 'effective_date'),
                    (r'termination\s*:?\s*([^\n]+)', 'termination_clause'),
                    (r'governing\s+law\s*:?\s*([^\n]+)', 'governing_law'),
                    (r'jurisdiction\s*:?\s*([^\n]+)', 'jurisdiction'),
                ]
                
                for pattern, term_type in term_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        terms[term_type] = match.group(1).strip()
            
            if terms:
                return {
                    'success': True,
                    'output': {
                        'extracted_terms': terms,
                        'terms_count': len(terms)
                    },
                    'confidence': 0.8
                }
            
            return {'success': False, 'error': 'No contract terms found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _apply_llm_reasoning(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply LLM-based reasoning when rules fail."""
        
        reasoning_steps = []
        
        try:
            if not self.llm_client:
                return {
                    'success': False,
                    'steps': [],
                    'error': 'LLM client not available'
                }
            
            # Prepare context for LLM
            context = self._prepare_llm_context(query, chunks, strategy)
            
            # Generate reasoning prompt
            prompt = self._create_reasoning_prompt(query, context, strategy)
            
            # Call LLM
            if self.config.llm.provider.lower() == 'openai':
                response = await self._call_openai_reasoning(prompt)
            else:
                response = await self._call_generic_llm_reasoning(prompt)
            
            # Parse LLM response
            parsed_response = self._parse_llm_reasoning_response(response)
            
            # Convert to reasoning steps
            reasoning_steps.append(ReasoningStep(
                step_type='llm_reasoning',
                description='Applied language model reasoning',
                input_data={'query': query, 'context': context},
                output_data=parsed_response,
                confidence=parsed_response.get('confidence', 0.7),
                sources=[chunk.get('doc_id') for chunk in chunks]
            ))
            
            return {
                'success': True,
                'steps': reasoning_steps
            }
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {str(e)}")
            return {
                'success': False,
                'steps': reasoning_steps,
                'error': str(e)
            }
    
    def _prepare_llm_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> str:
        """Prepare context for LLM reasoning."""
        
        context_parts = []
        
        # Add document information
        doc_types = strategy.get('document_types', [])
        if doc_types:
            context_parts.append(f"Document types: {', '.join(doc_types)}")
        
        # Add relevant chunks
        context_parts.append("Relevant content:")
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            content = chunk.get('content', '')[:500]  # Limit content length
            doc_id = chunk.get('doc_id', 'unknown')
            context_parts.append(f"[Document {i+1} - {doc_id}]: {content}...")
        
        return '\n\n'.join(context_parts)
    
    def _create_reasoning_prompt(
        self,
        query: str,
        context: str,
        strategy: Dict[str, Any]
    ) -> str:
        """Create reasoning prompt for LLM."""
        
        complexity = strategy.get('complexity', 'simple')
        
        prompt = f"""You are an AI assistant specialized in document analysis and reasoning.

Query: {query}

Context:
{context}

Analysis Requirements:
- Complexity level: {complexity}
- Requires calculation: {strategy.get('requires_calculation', False)}
- Requires comparison: {strategy.get('requires_comparison', False)}
- Multi-document analysis: {strategy.get('multi_document', False)}

Please provide a comprehensive analysis that:
1. Directly answers the query based on the provided context
2. Shows your reasoning steps clearly
3. Identifies the specific information sources used
4. Provides confidence level for your conclusion
5. Highlights any limitations or assumptions

Format your response as JSON with the following structure:
{{
    "answer": "Your direct answer to the query",
    "reasoning_steps": [
        {{
            "step": 1,
            "description": "What you did in this step",
            "evidence": "Specific evidence used",
            "conclusion": "What you concluded"
        }}
    ],
    "confidence": 0.85,
    "sources_used": ["doc_id1", "doc_id2"],
    "limitations": "Any limitations or assumptions"
}}

Response:"""
        
        return prompt
    
    async def _call_openai_reasoning(self, prompt: str) -> str:
        """Call OpenAI for reasoning."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.openai.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI reasoning call failed: {str(e)}")
            raise
    
    async def _call_generic_llm_reasoning(self, prompt: str) -> str:
        """Call generic LLM for reasoning."""
        # Placeholder for other LLM providers
        # Would implement Anthropic, local models, etc.
        
        logger.warning("Generic LLM reasoning not implemented")
        return json.dumps({
            "answer": "Unable to process with current LLM configuration",
            "reasoning_steps": [],
            "confidence": 0.0,
            "limitations": "LLM provider not fully supported"
        })
    
    def _parse_llm_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM reasoning response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback to simple parsing
            return {
                'answer': response,
                'reasoning_steps': [{'description': 'LLM reasoning completed'}],
                'confidence': 0.6,
                'limitations': 'Response format not standardized'
            }
            
        except json.JSONDecodeError:
            return {
                'answer': response,
                'reasoning_steps': [{'description': 'LLM reasoning completed'}],
                'confidence': 0.5,
                'limitations': 'Could not parse structured response'
            }
    
    async def _handle_calculations(
        self,
        query: str,
        reasoning_steps: List[ReasoningStep],
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle numerical calculations using calculation engine."""
        
        calculation_steps = []
        
        try:
            self.stats['calculation_requests'] += 1
            
            # Extract numerical expressions from query and reasoning steps
            expressions = []
            
            # From query
            numbers_in_query = re.findall(r'\d+(?:\.\d+)?', query)
            if numbers_in_query:
                expressions.extend(numbers_in_query)
            
            # From reasoning steps
            for step in reasoning_steps:
                if isinstance(step.output_data, dict):
                    for key, value in step.output_data.items():
                        if isinstance(value, (int, float)):
                            expressions.append(str(value))
                        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                            expressions.extend([str(x) for x in value])
            
            # Use calculation engine
            for expr in expressions[:5]:  # Limit calculations
                try:
                    calc_result = await self.calculation_engine.calculate(expr)
                    
                    calculation_steps.append(ReasoningStep(
                        step_type='calculation',
                        description=f'Calculated: {expr}',
                        input_data={'expression': expr},
                        output_data=calc_result.__dict__,
                        confidence=calc_result.confidence,
                        sources=[]
                    ))
                    
                except Exception as e:
                    logger.warning(f"Calculation failed for {expr}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Calculation handling failed: {str(e)}")
        
        return {
            'success': len(calculation_steps) > 0,
            'steps': calculation_steps
        }
    
    async def _synthesize_multi_document_evidence(
        self,
        query: str,
        reasoning_steps: List[ReasoningStep],
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Advanced multi-document evidence synthesis with conflict detection.
        
        Features:
        - Cross-document consensus analysis
        - Conflict detection and resolution
        - Evidence strength weighting
        - Document reliability scoring
        """
        
        synthesis_steps = []
        
        try:
            # Group chunks by document
            doc_groups = defaultdict(list)
            for chunk in chunks:
                doc_id = chunk.get('doc_id', 'unknown')
                doc_groups[doc_id].append(chunk)
            
            if len(doc_groups) < 2:
                return {'success': False, 'steps': [], 'consensus': None}
            
            logger.info(f"Synthesizing evidence from {len(doc_groups)} documents")
            
            # Analyze each document's contribution
            doc_analyses = {}
            for doc_id, doc_chunks in doc_groups.items():
                doc_content = ' '.join([chunk.get('content', '') for chunk in doc_chunks])
                
                # Calculate relevance score
                query_terms = set(query.lower().split())
                doc_terms = set(doc_content.lower().split())
                relevance = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
                
                # Calculate document quality score
                avg_chunk_score = sum(chunk.get('score', 0) for chunk in doc_chunks) / len(doc_chunks)
                
                # Extract key information for consensus analysis
                extracted_info = self._extract_key_information(doc_content, query)
                
                doc_analyses[doc_id] = {
                    'relevance_score': relevance,
                    'quality_score': avg_chunk_score,
                    'chunk_count': len(doc_chunks),
                    'content_length': len(doc_content),
                    'extracted_info': extracted_info
                }
            
            # Perform cross-document consensus analysis
            consensus_result = self._analyze_cross_document_consensus(doc_analyses, query)
            
            # Detect conflicts
            conflicts = self._detect_information_conflicts(doc_analyses)
            
            # Calculate synthesis confidence based on consensus and conflicts
            synthesis_confidence = self._calculate_synthesis_confidence(
                consensus_result, conflicts, doc_analyses
            )
            
            # Create synthesis step
            synthesis_steps.append(ReasoningStep(
                step_type='multi_document_synthesis',
                description=f'Synthesized evidence from {len(doc_groups)} documents with {consensus_result["consensus_level"]} consensus',
                input_data={
                    'documents': list(doc_groups.keys()),
                    'query': query
                },
                output_data={
                    'document_analyses': doc_analyses,
                    'consensus_result': consensus_result,
                    'detected_conflicts': conflicts,
                    'synthesis_method': 'advanced_consensus_based',
                    'evidence_strength': consensus_result.get('strength', 'moderate')
                },
                confidence=synthesis_confidence,
                sources=list(doc_groups.keys())
            ))
            
            # Add conflict resolution step if conflicts detected
            if conflicts:
                resolution_step = await self._resolve_conflicts(conflicts, doc_analyses, query)
                synthesis_steps.append(resolution_step)
            
            logger.info(f"Multi-document synthesis completed: {consensus_result['consensus_level']} consensus, {len(conflicts)} conflicts")
            
        except Exception as e:
            logger.error(f"Multi-document synthesis failed: {str(e)}")
        
        return {
            'success': len(synthesis_steps) > 0,
            'steps': synthesis_steps,
            'consensus': consensus_result if synthesis_steps else None
        }
    
    def _extract_key_information(self, content: str, query: str) -> Dict[str, Any]:
        """Extract key information from document content relevant to query."""
        extracted = {
            'numbers': [],
            'dates': [],
            'entities': [],
            'key_terms': []
        }
        
        # Extract numbers (including currency)
        number_pattern = r'[₹$€£¥]\s*[\d,]+\.?\d*|\d+\.?\d*'
        extracted['numbers'] = re.findall(number_pattern, content)
        
        # Extract dates
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
        extracted['dates'] = re.findall(date_pattern, content, re.IGNORECASE)
        
        # Extract capitalized terms (potential entities)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        extracted['entities'] = re.findall(entity_pattern, content)[:10]  # Limit to top 10
        
        # Extract query-relevant terms
        query_words = query.lower().split()
        for word in query_words:
            if len(word) > 3 and word in content.lower():
                extracted['key_terms'].append(word)
        
        return extracted
    
    def _analyze_cross_document_consensus(
        self,
        doc_analyses: Dict[str, Dict],
        query: str
    ) -> Dict[str, Any]:
        """
        Analyze consensus across multiple documents.
        
        Returns consensus level (high/medium/low) and supporting evidence.
        """
        
        # Compare extracted information across documents
        all_numbers = []
        all_dates = []
        all_entities = []
        
        for doc_id, analysis in doc_analyses.items():
            extracted = analysis.get('extracted_info', {})
            all_numbers.extend(extracted.get('numbers', []))
            all_dates.extend(extracted.get('dates', []))
            all_entities.extend(extracted.get('entities', []))
        
        # Calculate overlap metrics
        doc_count = len(doc_analyses)
        
        # Number consensus (how many docs have similar numbers)
        number_consensus = self._calculate_value_consensus(all_numbers, doc_count)
        
        # Date consensus
        date_consensus = self._calculate_value_consensus(all_dates, doc_count)
        
        # Entity overlap (common entities across documents)
        entity_overlap = len(set(all_entities)) / max(len(all_entities), 1) if all_entities else 0
        
        # Overall consensus score
        consensus_score = (number_consensus + date_consensus + entity_overlap) / 3
        
        # Determine consensus level
        if consensus_score >= 0.7:
            consensus_level = "high"
            strength = "strong"
        elif consensus_score >= 0.4:
            consensus_level = "medium"
            strength = "moderate"
        else:
            consensus_level = "low"
            strength = "weak"
        
        return {
            'consensus_level': consensus_level,
            'consensus_score': consensus_score,
            'strength': strength,
            'number_agreement': number_consensus,
            'date_agreement': date_consensus,
            'entity_overlap': entity_overlap,
            'supporting_documents': doc_count
        }
    
    def _calculate_value_consensus(self, values: List[str], doc_count: int) -> float:
        """Calculate consensus score for a list of values (numbers/dates)."""
        if not values:
            return 0.0
        
        from collections import Counter
        value_counts = Counter(values)
        
        # Most common value and its frequency
        if value_counts:
            most_common_count = value_counts.most_common(1)[0][1]
            return most_common_count / doc_count
        
        return 0.0
    
    def _detect_information_conflicts(
        self,
        doc_analyses: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts or contradictions between documents.
        
        Returns list of detected conflicts with severity ratings.
        """
        conflicts = []
        
        # Compare numbers across documents
        doc_numbers = {}
        for doc_id, analysis in doc_analyses.items():
            numbers = analysis.get('extracted_info', {}).get('numbers', [])
            if numbers:
                doc_numbers[doc_id] = numbers
        
        # Check for conflicting numbers
        if len(doc_numbers) >= 2:
            doc_ids = list(doc_numbers.keys())
            for i, doc_id1 in enumerate(doc_ids):
                for doc_id2 in doc_ids[i+1:]:
                    nums1 = set(doc_numbers[doc_id1])
                    nums2 = set(doc_numbers[doc_id2])
                    
                    # If documents have completely different numbers, it's a potential conflict
                    if nums1 and nums2 and not nums1.intersection(nums2):
                        conflicts.append({
                            'type': 'number_mismatch',
                            'severity': 'medium',
                            'documents': [doc_id1, doc_id2],
                            'description': f'Documents contain different numerical values',
                            'details': {
                                doc_id1: list(nums1)[:5],
                                doc_id2: list(nums2)[:5]
                            }
                        })
        
        # Compare dates
        doc_dates = {}
        for doc_id, analysis in doc_analyses.items():
            dates = analysis.get('extracted_info', {}).get('dates', [])
            if dates:
                doc_dates[doc_id] = dates
        
        if len(doc_dates) >= 2:
            date_sets = [set(dates) for dates in doc_dates.values()]
            if len(set().union(*date_sets)) > len(doc_dates):
                conflicts.append({
                    'type': 'date_inconsistency',
                    'severity': 'low',
                    'documents': list(doc_dates.keys()),
                    'description': 'Documents reference different dates',
                    'details': {doc_id: dates for doc_id, dates in doc_dates.items()}
                })
        
        return conflicts
    
    def _calculate_synthesis_confidence(
        self,
        consensus_result: Dict[str, Any],
        conflicts: List[Dict],
        doc_analyses: Dict[str, Dict]
    ) -> float:
        """
        Calculate confidence score for multi-document synthesis.
        
        Factors:
        - Consensus level (higher = more confidence)
        - Number of conflicts (more conflicts = less confidence)
        - Document quality scores
        - Number of supporting documents
        """
        
        # Base confidence from consensus
        consensus_score = consensus_result.get('consensus_score', 0.5)
        
        # Penalty for conflicts
        conflict_penalty = len(conflicts) * 0.1
        
        # Boost for document quality
        avg_quality = sum(doc.get('quality_score', 0.5) for doc in doc_analyses.values()) / len(doc_analyses)
        quality_boost = (avg_quality - 0.5) * 0.2
        
        # Boost for multiple supporting documents
        doc_count_boost = min(len(doc_analyses) * 0.05, 0.15)
        
        # Calculate final confidence
        confidence = consensus_score + quality_boost + doc_count_boost - conflict_penalty
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    async def _resolve_conflicts(
        self,
        conflicts: List[Dict],
        doc_analyses: Dict[str, Dict],
        query: str
    ) -> ReasoningStep:
        """
        Attempt to resolve detected conflicts.
        
        Strategy:
        - Prioritize higher quality documents
        - Use majority consensus
        - Flag unresolvable conflicts
        """
        
        resolutions = []
        
        for conflict in conflicts:
            conflict_type = conflict['type']
            severity = conflict['severity']
            
            # Determine resolution strategy based on conflict type
            if conflict_type == 'number_mismatch':
                # Use document with highest quality score
                docs_involved = conflict['documents']
                best_doc = max(docs_involved, key=lambda d: doc_analyses[d].get('quality_score', 0))
                
                resolutions.append({
                    'conflict_type': conflict_type,
                    'resolution_method': 'quality_based_selection',
                    'selected_source': best_doc,
                    'reasoning': f'Selected {best_doc} based on higher quality score'
                })
            
            elif conflict_type == 'date_inconsistency':
                # Flag as informational, usually not critical
                resolutions.append({
                    'conflict_type': conflict_type,
                    'resolution_method': 'informational_flag',
                    'reasoning': 'Multiple dates found across documents - context-dependent'
                })
        
        return ReasoningStep(
            step_type='conflict_resolution',
            description=f'Resolved {len(resolutions)} conflicts across documents',
            input_data={'conflicts': conflicts},
            output_data={'resolutions': resolutions},
            confidence=0.7,
            sources=list(doc_analyses.keys())
        )
    
    async def _generate_final_answer(
        self,
        query: str,
        reasoning_steps: List[ReasoningStep],
        chunks: List[Dict[str, Any]]
    ) -> str:
        """Generate final answer based on reasoning steps."""
        
        try:
            # Collect answers from reasoning steps
            step_answers = []
            
            for step in reasoning_steps:
                if step.step_type == 'llm_reasoning' and isinstance(step.output_data, dict):
                    answer = step.output_data.get('answer')
                    if answer:
                        step_answers.append(answer)
                elif step.step_type == 'rule_application' and isinstance(step.output_data, dict):
                    # Convert rule output to answer format
                    rule_output = step.output_data
                    if 'calculated_total' in rule_output:
                        step_answers.append(f"The calculated total is {rule_output['calculated_total']}")
                    elif 'total_years' in rule_output:
                        step_answers.append(f"Total experience: {rule_output['total_years']} years")
                    elif 'matched_skills' in rule_output:
                        step_answers.append(f"Found {len(rule_output['matched_skills'])} matching skills")
            
            # If we have specific answers, use the most confident one
            if step_answers:
                # For now, use the first answer (could be improved with confidence ranking)
                return step_answers[0]
            
            # Fallback: generate answer based on most relevant chunk
            if chunks:
                most_relevant_chunk = max(chunks, key=lambda x: x.get('score', 0))
                content = most_relevant_chunk.get('content', '')
                
                # Simple answer extraction (first sentence that might contain answer)
                sentences = content.split('.')
                for sentence in sentences[:3]:
                    if any(word in sentence.lower() for word in query.lower().split()):
                        return sentence.strip()
                
                # If no good sentence found, return summary
                return f"Based on the document analysis: {content[:200]}..."
            
            return "Unable to generate a definitive answer based on the available information."
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "An error occurred while generating the final answer."
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        return {
            'reasoning_stats': self.stats.copy(),
            'calculation_engine_stats': await self.calculation_engine.get_stats(),
            'confidence_calculator_stats': await self.confidence_calculator.get_stats()
        }
