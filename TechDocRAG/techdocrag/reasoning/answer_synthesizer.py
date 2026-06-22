"""
Answer Synthesizer for TechDocRAG
Transforms retrieved document chunks into natural language answers using LLM.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.types import QueryResponse, ConfidenceScore, SourceAttribution, ReasoningStep
from ..llm.gemini_client import GeminiClient
from ..utils.logging import get_logger
from ..utils.exceptions import LLMError

logger = get_logger(__name__)


class AnswerSynthesizer:
    """
    Synthesizes natural language answers from retrieved document chunks.
    
    Key Features:
    - Transforms raw chunks into coherent answers
    - Generates proper citations
    - Provides reasoning explanations
    - Calculates LLM-aware confidence scores
    """
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        enable_synthesis: bool = True
    ):
        """
        Initialize Answer Synthesizer.
        
        Args:
            gemini_client: Gemini LLM client (will create default if not provided)
            enable_synthesis: Enable LLM synthesis (if False, returns template answers)
        """
        self.gemini_client = gemini_client
        self.enable_synthesis = enable_synthesis
        
        # Statistics
        self.stats = {
            'total_syntheses': 0,
            'llm_syntheses': 0,
            'template_answers': 0,
            'avg_synthesis_time': 0.0
        }
        
        logger.info(f"AnswerSynthesizer initialized (LLM enabled: {enable_synthesis})")
    
    async def synthesize_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        retrieval_confidence: float = 0.0
    ) -> QueryResponse:
        """
        Synthesize natural language answer from retrieved chunks.
        
        Args:
            question: User's question
            retrieved_chunks: List of retrieved document chunks with metadata
            retrieval_confidence: Confidence from retrieval system
        
        Returns:
            QueryResponse with synthesized answer, citations, and reasoning
        """
        start_time = datetime.now()
        self.stats['total_syntheses'] += 1
        
        try:
            if not retrieved_chunks:
                return self._create_no_answer_response(question)
            
            # Check if LLM synthesis is enabled and available
            if self.enable_synthesis and self.gemini_client:
                response = await self._synthesize_with_llm(
                    question, retrieved_chunks, retrieval_confidence
                )
                self.stats['llm_syntheses'] += 1
            else:
                response = self._create_template_response(
                    question, retrieved_chunks, retrieval_confidence
                )
                self.stats['template_answers'] += 1
            
            # Update statistics
            synthesis_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_synthesis_time(synthesis_time)
            
            logger.info(
                f"Answer synthesized in {synthesis_time:.2f}s",
                extra={
                    'question': question[:100],
                    'used_llm': self.enable_synthesis,
                    'confidence': response.confidence.overall
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {str(e)}", extra={'question': question[:100]})
            # Fallback to template response on error
            return self._create_template_response(
                question, retrieved_chunks, retrieval_confidence
            )
    
    async def _synthesize_with_llm(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        retrieval_confidence: float
    ) -> QueryResponse:
        """Synthesize answer using Gemini LLM."""
        
        # Extract context and sources from chunks
        context = [chunk.get('content', '') for chunk in chunks[:5]]  # Top 5 chunks
        sources = [
            {
                'doc_id': chunk.get('doc_id', 'unknown'),
                'title': chunk.get('metadata', {}).get('title', 'Unknown Document'),
                'chunk_id': chunk.get('chunk_id', ''),
                'score': chunk.get('score', 0.0)
            }
            for chunk in chunks[:5]
        ]
        
        # Generate answer using Gemini
        llm_result = await self.gemini_client.generate_answer(
            question=question,
            context=context,
            sources=sources
        )
        
        # Build QueryResponse
        response = QueryResponse(
            query=question,
            answer=llm_result['answer'],
            confidence=self._build_confidence_score(
                llm_confidence=llm_result['confidence'],
                retrieval_confidence=retrieval_confidence,
                num_sources=len(chunks)
            ),
            sources=self._build_source_attributions(chunks, llm_result.get('citations', [])),
            reasoning_chain=[
                ReasoningStep(
                    step_id="step_1",
                    operation="retrieve",
                    input_data={'query': question},
                    output_data={'chunks_found': len(chunks)},
                    confidence=retrieval_confidence,
                    sources=self._build_source_attributions(chunks, [])
                ),
                ReasoningStep(
                    step_id="step_2",
                    operation="synthesize",
                    input_data={'chunks': len(context)},
                    output_data={'answer_generated': True},
                    confidence=llm_result['confidence'],
                    sources=[]
                )
            ],
            response_time=0.0,  # Will be set by caller
            calculations=[]
        )
        
        return response
    
    def _create_template_response(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        retrieval_confidence: float
    ) -> QueryResponse:
        """Create template-based response (fallback when LLM not available)."""
        
        # Build simple answer from top chunk
        top_chunk = chunks[0] if chunks else {}
        answer = f"{top_chunk.get('content', 'No relevant information found.')[:500]}..."
        
        response = QueryResponse(
            query=question,
            answer=answer,
            confidence=ConfidenceScore(
                overall=retrieval_confidence,
                retrieval=retrieval_confidence,
                reasoning=0.0,  # No reasoning without LLM
                calculation=0.0,  # No calculations
                source_quality=min(100, len(chunks) * 20)
            ),
            sources=self._build_source_attributions(chunks, []),
            reasoning_chain=[
                ReasoningStep(
                    step_id="step_1",
                    operation="retrieve",
                    input_data={'query': question},
                    output_data={'chunks_found': len(chunks), 'mode': 'template'},
                    confidence=retrieval_confidence,
                    sources=self._build_source_attributions(chunks, [])
                )
            ],
            response_time=0.0,
            calculations=[]
        )
        
        return response
    
    def _create_no_answer_response(self, question: str) -> QueryResponse:
        """Create response when no relevant chunks found."""
        
        return QueryResponse(
            query=question,
            answer="I couldn't find relevant information to answer this question in the available documents.",
            confidence=ConfidenceScore(
                overall=0.0,
                retrieval=0.0,
                reasoning=0.0,
                calculation=0.0,
                source_quality=0.0
            ),
            sources=[],
            reasoning_chain=[
                ReasoningStep(
                    step_id="step_1",
                    operation="retrieve",
                    input_data={'query': question},
                    output_data={'chunks_found': 0},
                    confidence=0.0,
                    sources=[]
                )
            ],
            response_time=0.0,
            calculations=[]
        )
    
    def _build_confidence_score(
        self,
        llm_confidence: float,
        retrieval_confidence: float,
        num_sources: int
    ) -> ConfidenceScore:
        """Build comprehensive confidence score."""
        
        # Calculate source quality based on number and diversity
        source_quality = min(100, num_sources * 15)
        
        # Overall confidence is weighted average
        overall = (
            llm_confidence * 0.5 +  # 50% from LLM confidence
            retrieval_confidence * 0.3 +  # 30% from retrieval
            source_quality * 0.2  # 20% from source quality
        )
        
        return ConfidenceScore(
            overall=round(overall, 1),
            retrieval=round(retrieval_confidence, 1),
            reasoning=round(llm_confidence, 1),
            calculation=0.0,  # No calculations in this version
            source_quality=round(source_quality, 1)
        )
    
    def _build_source_attributions(
        self,
        chunks: List[Dict[str, Any]],
        citations: List[str]
    ) -> List[SourceAttribution]:
        """Build source attribution list."""
        
        attributions = []
        
        for i, chunk in enumerate(chunks[:5]):  # Top 5 sources
            metadata = chunk.get('metadata', {})
            
            attribution = SourceAttribution(
                doc_id=chunk.get('doc_id', f'doc_{i}'),
                chunk_id=chunk.get('chunk_id', f'chunk_{i}'),
                relevance_score=chunk.get('score', 0.0),
                text_snippet=chunk.get('content', '')[:200],
                page_number=metadata.get('page_number', 1)
            )
            
            attributions.append(attribution)
        
        return attributions
    
    def _update_avg_synthesis_time(self, new_time: float):
        """Update running average of synthesis times."""
        total = self.stats['total_syntheses']
        current_avg = self.stats['avg_synthesis_time']
        self.stats['avg_synthesis_time'] = (current_avg * (total - 1) + new_time) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        return {
            **self.stats,
            'llm_usage_rate': (
                self.stats['llm_syntheses'] / self.stats['total_syntheses'] * 100
                if self.stats['total_syntheses'] > 0 else 0
            )
        }
