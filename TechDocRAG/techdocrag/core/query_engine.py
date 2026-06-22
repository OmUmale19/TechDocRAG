"""
Intelligent Query Engine for TechDocRAG system.
Orchestrates hybrid retrieval, reasoning, and response generation with explainability.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..core.types import (
    QueryResponse, ConfidenceScore, SourceAttribution, ReasoningStep,
    DocumentResult, MultiDocumentContext, CalculationResult
)
from ..core.config import get_config
from ..utils.logging import get_logger, performance_log, audit_logger, set_request_context
from ..utils.helpers import generate_id, weighted_average
from ..utils.validators import validate_query
from ..utils.exceptions import RetrievalError, ReasoningError

from ..retrieval.hybrid_retriever import HybridRetriever
from ..reasoning.reasoning_engine import ReasoningEngine
from ..reasoning.confidence_calculator import ConfidenceCalculator
from ..reasoning.answer_synthesizer import AnswerSynthesizer
from ..llm.gemini_client import GeminiClient

logger = get_logger(__name__)


class QueryEngine:
    """
    Advanced query processing engine with hybrid retrieval and explainable reasoning.
    
    Key Capabilities:
    1. Hybrid retrieval (semantic + keyword + fusion)
    2. Multi-document reasoning and synthesis
    3. Integrated calculation engine
    4. Explainable AI with confidence scoring
    5. Multi-step reasoning chains
    6. Context-aware response generation
    """
    
    def __init__(self, config=None):
        """Initialize query engine with all components."""
        self.config = config or get_config()
        
        # Initialize core components
        self.retriever = HybridRetriever(self.config)
        self.reasoning_engine = ReasoningEngine(self.config)
        self.confidence_calculator = ConfidenceCalculator(self.config.confidence)
        
        # Initialize LLM and Answer Synthesizer (Phase 3)
        self.gemini_client = None
        self.answer_synthesizer = None
        if self.config.llm.enable_synthesis:
            try:
                self.gemini_client = GeminiClient(
                    api_key=self.config.llm.api_key,
                    model_name=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                self.answer_synthesizer = AnswerSynthesizer(
                    gemini_client=self.gemini_client,
                    enable_synthesis=True
                )
                logger.info("LLM-powered answer synthesis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM synthesis: {str(e)}. Falling back to template mode.")
                self.answer_synthesizer = AnswerSynthesizer(enable_synthesis=False)
        else:
            self.answer_synthesizer = AnswerSynthesizer(enable_synthesis=False)
            logger.info("LLM synthesis disabled - using template answers")
        
        # Query processing state
        self.document_store = {}  # In-memory document store
        self.active_sessions = {}  # Active query sessions
        
        logger.info("QueryEngine initialized with hybrid retrieval and reasoning")
    
    @performance_log("query_processing")
    async def query(
        self,
        query: str,
        doc_ids: List[str] = None,
        max_docs: int = 5,
        user_id: str = None,
        session_id: str = None
    ) -> QueryResponse:
        """
        Process query and generate comprehensive response.
        
        Args:
            query: User query string
            doc_ids: Optional list of specific document IDs to search
            max_docs: Maximum number of documents to consider
            user_id: User ID for audit logging
            session_id: Session ID for context tracking
            
        Returns:
            QueryResponse with answer, confidence, sources, and reasoning
        """
        start_time = datetime.now()
        request_id = generate_id("query")
        
        # Set request context for logging
        set_request_context(request_id, user_id)
        
        logger.info(f"Processing query: {query[:100]}...", 
                   extra={'request_id': request_id, 'user_id': user_id})
        
        try:
            # Step 1: Validate and preprocess query
            validated_query = validate_query(query)
            
            # Audit log
            audit_logger.log_document_access(
                doc_id="query_processing",
                action="query",
                user_id=user_id,
                details={'query': validated_query, 'doc_ids': doc_ids}
            )
            
            # Step 2: Retrieve relevant documents and passages
            retrieval_results = await self.retriever.retrieve(
                query=validated_query,
                doc_ids=doc_ids,
                top_k=max_docs * 3  # Retrieve more for better filtering
            )
            
            logger.info(f"Retrieved {len(retrieval_results)} relevant passages")
            
            # Step 3: Calculate retrieval confidence
            retrieval_confidence = self._calculate_retrieval_confidence(retrieval_results)
            
            # Step 4: Use Answer Synthesizer for LLM-powered response (Phase 3)
            if self.answer_synthesizer:
                response = await self.answer_synthesizer.synthesize_answer(
                    question=validated_query,
                    retrieved_chunks=retrieval_results,
                    retrieval_confidence=retrieval_confidence
                )
            else:
                # Fallback to old reasoning chain method
                multi_doc_context = self._create_multi_document_context(retrieval_results, doc_ids)
                reasoning_chain = await self._execute_reasoning_chain(
                    validated_query, retrieval_results, multi_doc_context
                )
                answer = await self._generate_answer(validated_query, reasoning_chain)
                confidence = await self._calculate_confidence(
                    validated_query, retrieval_results, reasoning_chain, answer
                )
                sources = self._create_source_attributions(retrieval_results, reasoning_chain)
                calculations = self._extract_calculations(reasoning_chain)
                
                response = QueryResponse(
                    query=validated_query,
                    answer=answer,
                    confidence=confidence,
                    sources=sources,
                    reasoning_steps=reasoning_chain,
                    processing_time=0.0
                )
            
            # Update response time
            response_time = (datetime.now() - start_time).total_seconds()
            response.processing_time = response_time
            
            logger.info(f"Query processed successfully in {response_time:.2f}s with confidence {confidence.overall:.2f}",
                       extra={'request_id': request_id})
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", 
                        extra={'request_id': request_id, 'query': query[:100]})
            
            # Return error response
            return self._create_error_response(validated_query if 'validated_query' in locals() else query, str(e))
    
    async def add_document(self, document_result: DocumentResult):
        """Add processed document to the query engine."""
        try:
            # Store document
            self.document_store[document_result.doc_id] = document_result
            
            # Add to retrieval index
            await self.retriever.add_document(document_result)
            
            logger.info(f"Added document to query engine: {document_result.doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to add document {document_result.doc_id}: {str(e)}")
            raise RetrievalError(f"Failed to add document: {str(e)}") from e
    
    async def add_documents_batch(self, documents: List[DocumentResult]):
        """Add multiple documents in batch."""
        try:
            for doc in documents:
                self.document_store[doc.doc_id] = doc
            
            await self.retriever.add_documents_batch(documents)
            
            logger.info(f"Added {len(documents)} documents to query engine")
            
        except Exception as e:
            logger.error(f"Failed to add documents batch: {str(e)}")
            raise RetrievalError(f"Failed to add documents: {str(e)}") from e
    
    async def _execute_reasoning_chain(
        self,
        query: str,
        retrieval_results: List[Dict],
        multi_doc_context: Optional[MultiDocumentContext]
    ) -> List[ReasoningStep]:
        """Execute multi-step reasoning chain."""
        reasoning_chain = []
        
        try:
            # Step 1: Initial retrieval step
            retrieval_step = ReasoningStep(
                step_id=generate_id("step"),
                operation="retrieve",
                input_data={'query': query},
                output_data={'results_count': len(retrieval_results)},
                confidence=self._calculate_retrieval_confidence(retrieval_results),
                sources=[self._create_source_attribution(result) for result in retrieval_results[:5]]
            )
            reasoning_chain.append(retrieval_step)
            
            # Step 2: Information extraction and synthesis
            synthesis_result = await self.reasoning_engine.synthesize_information(
                query, retrieval_results
            )
            
            synthesis_step = ReasoningStep(
                step_id=generate_id("step"),
                operation="synthesize",
                input_data={'retrieval_results': len(retrieval_results)},
                output_data=synthesis_result,
                confidence=synthesis_result.get('confidence', 0.7)
            )
            reasoning_chain.append(synthesis_step)
            
            # Step 3: Multi-document reasoning if applicable
            if multi_doc_context and len(multi_doc_context.doc_ids) > 1:
                multi_doc_result = await self.reasoning_engine.multi_document_reasoning(
                    query, retrieval_results, multi_doc_context
                )
                
                multi_doc_step = ReasoningStep(
                    step_id=generate_id("step"),
                    operation="multi_document_reasoning",
                    input_data={'doc_count': len(multi_doc_context.doc_ids)},
                    output_data=multi_doc_result,
                    confidence=multi_doc_result.get('confidence', 0.6)
                )
                reasoning_chain.append(multi_doc_step)
            
            # Step 4: Calculation if needed
            if self._requires_calculation(query, synthesis_result):
                calculation_result = await self.reasoning_engine.perform_calculations(
                    query, synthesis_result
                )
                
                calculation_step = ReasoningStep(
                    step_id=generate_id("step"),
                    operation="calculate",
                    input_data={'calculation_type': calculation_result.get('operation', 'unknown')},
                    output_data=calculation_result,
                    confidence=calculation_result.get('confidence', 0.8)
                )
                reasoning_chain.append(calculation_step)
            
            # Step 5: Final reasoning and answer generation
            final_reasoning = await self.reasoning_engine.generate_final_reasoning(
                query, reasoning_chain
            )
            
            final_step = ReasoningStep(
                step_id=generate_id("step"),
                operation="final_reasoning",
                input_data={'previous_steps': len(reasoning_chain)},
                output_data=final_reasoning,
                confidence=final_reasoning.get('confidence', 0.7)
            )
            reasoning_chain.append(final_step)
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Reasoning chain execution failed: {str(e)}")
            raise ReasoningError(f"Reasoning failed: {str(e)}") from e
    
    async def _generate_answer(self, query: str, reasoning_chain: List[ReasoningStep]) -> str:
        """Generate final answer from reasoning chain."""
        try:
            # Extract key information from reasoning chain
            synthesis_data = {}
            calculations = []
            
            for step in reasoning_chain:
                if step.operation == "synthesize":
                    synthesis_data = step.output_data
                elif step.operation == "calculate":
                    calculations.append(step.output_data)
                elif step.operation == "final_reasoning":
                    # Use final reasoning as primary answer source
                    if 'answer' in step.output_data:
                        return step.output_data['answer']
            
            # Fallback answer generation
            answer_parts = []
            
            if 'key_information' in synthesis_data:
                answer_parts.append(synthesis_data['key_information'])
            
            if calculations:
                calc_summary = "; ".join([f"{calc.get('operation', 'Calculation')}: {calc.get('result', 'N/A')}" 
                                         for calc in calculations])
                answer_parts.append(f"Calculations: {calc_summary}")
            
            if answer_parts:
                return " ".join(answer_parts)
            else:
                return "I was unable to find sufficient information to answer your question."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"
    
    async def _calculate_confidence(
        self,
        query: str,
        retrieval_results: List[Dict],
        reasoning_chain: List[ReasoningStep],
        answer: str
    ) -> ConfidenceScore:
        """Calculate comprehensive confidence score."""
        try:
            # Retrieval confidence
            retrieval_confidence = self._calculate_retrieval_confidence(retrieval_results)
            
            # Reasoning confidence (average of all reasoning steps)
            reasoning_confidences = [step.confidence for step in reasoning_chain]
            reasoning_confidence = weighted_average(reasoning_confidences, [1.0] * len(reasoning_confidences))
            
            # Calculation confidence (if any calculations were performed)
            calculation_steps = [step for step in reasoning_chain if step.operation == "calculate"]
            calculation_confidence = 1.0  # Default if no calculations
            if calculation_steps:
                calc_confidences = [step.confidence for step in calculation_steps]
                calculation_confidence = weighted_average(calc_confidences, [1.0] * len(calc_confidences))
            
            # Source quality confidence
            source_quality = self._calculate_source_quality(retrieval_results)
            
            # Overall confidence (weighted combination)
            overall_confidence = weighted_average(
                [retrieval_confidence, reasoning_confidence, calculation_confidence, source_quality],
                [0.3, 0.4, 0.2, 0.1]  # Weights for different components
            )
            
            return ConfidenceScore(
                overall=overall_confidence,
                retrieval=retrieval_confidence,
                reasoning=reasoning_confidence,
                calculation=calculation_confidence,
                source_quality=source_quality
            )
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return ConfidenceScore(
                overall=0.5, retrieval=0.5, reasoning=0.5, 
                calculation=0.5, source_quality=0.5
            )
    
    def _create_multi_document_context(
        self, 
        retrieval_results: List[Dict], 
        doc_ids: List[str] = None
    ) -> Optional[MultiDocumentContext]:
        """Create multi-document context if multiple documents are involved."""
        # Extract unique document IDs from retrieval results
        result_doc_ids = list(set(result.get('doc_id') for result in retrieval_results if result.get('doc_id')))
        
        if len(result_doc_ids) <= 1:
            return None
        
        # Create relationships (simplified - could be enhanced with document similarity)
        relationships = {doc_id: [other_id for other_id in result_doc_ids if other_id != doc_id] 
                        for doc_id in result_doc_ids}
        
        # Create timeline (if documents have timestamps)
        timeline = []
        for doc_id in result_doc_ids:
            if doc_id in self.document_store:
                doc = self.document_store[doc_id]
                timeline.append((doc_id, doc.processed_at))
        
        return MultiDocumentContext(
            doc_ids=result_doc_ids,
            relationships=relationships,
            timeline=sorted(timeline, key=lambda x: x[1])
        )
    
    def _requires_calculation(self, query: str, synthesis_data: Dict) -> bool:
        """Determine if query requires mathematical calculations."""
        calculation_keywords = [
            'calculate', 'compute', 'sum', 'total', 'add', 'subtract',
            'multiply', 'divide', 'percentage', 'percent', 'ratio',
            'average', 'mean', 'difference', 'amount due', 'penalty'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in calculation_keywords)
    
    def _calculate_retrieval_confidence(self, retrieval_results: List[Dict]) -> float:
        """Calculate confidence based on retrieval results quality."""
        if not retrieval_results:
            return 0.0
        
        # Average retrieval scores
        scores = [result.get('score', 0.0) for result in retrieval_results]
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence if top results have high scores
        top_3_scores = sorted(scores, reverse=True)[:3]
        if top_3_scores and top_3_scores[0] > 0.8:
            avg_score = min(1.0, avg_score + 0.1)
        
        return avg_score
    
    def _calculate_source_quality(self, retrieval_results: List[Dict]) -> float:
        """Calculate source quality based on document metadata."""
        if not retrieval_results:
            return 0.0
        
        quality_scores = []
        
        for result in retrieval_results:
            doc_id = result.get('doc_id')
            base_quality = 0.7  # Base quality score
            
            # Boost quality for documents with high processing confidence
            if doc_id in self.document_store:
                doc = self.document_store[doc_id]
                processing_confidence = doc.metrics.confidence_scores.get('overall_confidence', 0.7)
                base_quality = min(1.0, base_quality + processing_confidence * 0.2)
            
            # Boost quality for exact matches
            if result.get('score', 0) > 0.9:
                base_quality = min(1.0, base_quality + 0.1)
            
            quality_scores.append(base_quality)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _create_source_attributions(
        self, 
        retrieval_results: List[Dict], 
        reasoning_chain: List[ReasoningStep]
    ) -> List[SourceAttribution]:
        """Create source attributions for transparency."""
        sources = []
        
        # Add sources from retrieval results
        for result in retrieval_results[:5]:  # Top 5 sources
            source = self._create_source_attribution(result)
            sources.append(source)
        
        # Add sources from reasoning chain
        for step in reasoning_chain:
            sources.extend(step.sources)
        
        # Deduplicate sources
        seen_sources = set()
        unique_sources = []
        
        for source in sources:
            source_key = f"{source.doc_id}_{source.chunk_id}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)
        
        return unique_sources[:10]  # Limit to top 10 sources
    
    def _create_source_attribution(self, result: Dict) -> SourceAttribution:
        """Create source attribution from retrieval result."""
        return SourceAttribution(
            doc_id=result.get('doc_id', 'unknown'),
            chunk_id=result.get('chunk_id', 'unknown'),
            relevance_score=result.get('score', 0.0),
            text_snippet=result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', ''),
            page_number=result.get('page_number', 1),
            bbox=result.get('bbox')
        )
    
    def _extract_calculations(self, reasoning_chain: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """Extract calculation results from reasoning chain."""
        calculations = []
        
        for step in reasoning_chain:
            if step.operation == "calculate":
                calculations.append(step.output_data)
        
        return calculations
    
    def _create_error_response(self, query: str, error_message: str) -> QueryResponse:
        """Create error response for failed queries."""
        return QueryResponse(
            query=query,
            answer=f"I encountered an error while processing your query: {error_message}",
            confidence=ConfidenceScore(
                overall=0.0, retrieval=0.0, reasoning=0.0, 
                calculation=0.0, source_quality=0.0
            ),
            sources=[],
            reasoning_chain=[],
            calculations=[],
            response_time=0.0
        )
    
    async def process_query(self, query: str, max_results: int = 5, **kwargs) -> QueryResponse:
        """
        Alias for query method to maintain compatibility with tests.
        
        Args:
            query: User query string
            max_results: Maximum number of results to return (mapped to max_docs)
            **kwargs: Additional keyword arguments passed to query method
            
        Returns:
            QueryResponse with answer, confidence, sources, and reasoning
        """
        return await self.query(query, max_docs=max_results, **kwargs)
    
    async def get_document_summary(self, doc_id: str) -> Dict[str, Any]:
        """Get summary of a specific document."""
        if doc_id not in self.document_store:
            raise RetrievalError(f"Document not found: {doc_id}")
        
        doc = self.document_store[doc_id]
        
        return {
            'doc_id': doc.doc_id,
            'doc_type': doc.doc_type.value,
            'file_path': str(doc.file_path),
            'total_chunks': len(doc.chunks),
            'total_fields': len(doc.extracted_fields),
            'processing_confidence': doc.metrics.confidence_scores.get('overall_confidence', 0.0),
            'processed_at': doc.processed_at.isoformat(),
            'key_fields': [{'name': f.name, 'value': f.value, 'confidence': f.confidence} 
                          for f in doc.extracted_fields[:10]]
        }
    
    async def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to query."""
        try:
            results = await self.retriever.retrieve(query=query, top_k=top_k * 2)
            
            # Group by document and calculate document-level scores
            doc_scores = {}
            for result in results:
                doc_id = result.get('doc_id')
                score = result.get('score', 0.0)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] = max(doc_scores[doc_id], score)
                else:
                    doc_scores[doc_id] = score
            
            # Sort by score and return top documents
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            similar_docs = []
            for doc_id, score in sorted_docs:
                if doc_id in self.document_store:
                    doc = self.document_store[doc_id]
                    similar_docs.append({
                        'doc_id': doc_id,
                        'similarity_score': score,
                        'doc_type': doc.doc_type.value,
                        'file_path': str(doc.file_path),
                        'processing_confidence': doc.metrics.confidence_scores.get('overall_confidence', 0.0)
                    })
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Similar document search failed: {str(e)}")
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health metrics."""
        total_docs = len(self.document_store)
        total_chunks = sum(len(doc.chunks) for doc in self.document_store.values())
        
        # Calculate average processing confidence
        if total_docs > 0:
            avg_confidence = sum(
                doc.metrics.confidence_scores.get('overall_confidence', 0.0) 
                for doc in self.document_store.values()
            ) / total_docs
        else:
            avg_confidence = 0.0
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'average_processing_confidence': avg_confidence,
            'retrieval_system': await self.retriever.get_stats(),
            'reasoning_system': await self.reasoning_engine.get_stats(),
            'confidence_calculator': await self.confidence_calculator.get_stats()
        }
