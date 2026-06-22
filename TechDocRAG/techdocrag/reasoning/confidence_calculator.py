"""
Confidence calculation system for TechDocRAG.
Provides comprehensive confidence scoring for reasoning and retrieval.
"""

import asyncio
import math
from typing import List, Dict, Any, Optional
from statistics import mean, stdev

from ..core.types import ConfidenceScore, ReasoningStep
from ..core.config import ConfidenceConfig
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import ConfidenceError

logger = get_logger(__name__)


class ConfidenceCalculator:
    """
    Advanced confidence calculation system.
    
    Confidence Factors:
    1. Source Reliability - Quality and consistency of source documents
    2. Retrieval Quality - Relevance and ranking of retrieved content
    3. Reasoning Coherence - Logical consistency of reasoning steps
    4. Evidence Strength - Amount and quality of supporting evidence
    5. Calculation Accuracy - Precision of numerical computations
    6. Multi-document Consensus - Agreement across multiple sources
    
    Features:
    - Multi-factor confidence scoring
    - Uncertainty quantification
    - Evidence-based weighting
    - Calibrated confidence levels
    """
    
    def __init__(self, config: ConfidenceConfig):
        self.config = config
        
        # Confidence weights for different factors
        self.factor_weights = {
            'source_reliability': config.source_weight,
            'retrieval_quality': config.retrieval_weight,
            'reasoning_coherence': config.reasoning_weight,
            'evidence_strength': config.evidence_weight,
            'calculation_accuracy': config.calculation_weight,
            'multi_document_consensus': config.consensus_weight
        }
        
        # Confidence calibration parameters
        self.calibration_params = {
            'base_confidence': 0.5,
            'evidence_boost': 0.3,
            'consensus_boost': 0.2,
            'uncertainty_penalty': 0.1
        }
        
        # Statistics
        self.stats = {
            'total_calculations': 0,
            'avg_confidence': 0.0,
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0      # < 0.5
            }
        }
        
        logger.info("ConfidenceCalculator initialized")
    
    @performance_log("confidence_calculation")
    async def calculate_overall_confidence(
        self,
        factors: Dict[str, float],
        evidence: List[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score from multiple factors.
        
        Args:
            factors: Dictionary of confidence factors (0.0 to 1.0)
            evidence: Supporting evidence for confidence calculation
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        self.stats['total_calculations'] += 1
        
        try:
            # Normalize and validate factors
            normalized_factors = self._normalize_factors(factors)
            
            # Calculate weighted confidence
            weighted_confidence = self._calculate_weighted_confidence(normalized_factors)
            
            # Apply evidence boost
            evidence_boost = self._calculate_evidence_boost(evidence) if evidence else 0.0
            
            # Apply consensus boost if multiple sources
            consensus_boost = self._calculate_consensus_boost(evidence) if evidence else 0.0
            
            # Calculate uncertainty penalty
            uncertainty_penalty = self._calculate_uncertainty_penalty(normalized_factors)
            
            # Combine all factors
            raw_confidence = (
                weighted_confidence + 
                evidence_boost + 
                consensus_boost - 
                uncertainty_penalty
            )
            
            # Clamp to valid range
            overall_confidence = max(0.0, min(1.0, raw_confidence))
            
            # Create confidence score object
            confidence_score = ConfidenceScore(
                overall_confidence=overall_confidence,
                factor_scores=normalized_factors,
                evidence_count=len(evidence) if evidence else 0,
                source_diversity=self._calculate_source_diversity(evidence) if evidence else 0.0,
                uncertainty_level=self._calculate_uncertainty_level(normalized_factors)
            )
            
            # Update statistics
            self._update_statistics(overall_confidence)
            
            logger.debug(f"Confidence calculated: {overall_confidence:.3f}")
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            
            # Return default low confidence
            return ConfidenceScore(
                overall_confidence=0.1,
                factor_scores={'error': 0.1},
                evidence_count=0,
                source_diversity=0.0,
                uncertainty_level=1.0
            )
    
    async def calculate_retrieval_confidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: str
    ) -> float:
        """Calculate confidence based on retrieval quality."""
        try:
            if not retrieved_chunks:
                return 0.0
            
            factors = []
            
            # Factor 1: Score distribution
            scores = [chunk.get('score', 0.0) for chunk in retrieved_chunks]
            if scores:
                max_score = max(scores)
                avg_score = mean(scores)
                score_factor = (max_score + avg_score) / 2
                factors.append(score_factor)
            
            # Factor 2: Content relevance
            query_terms = set(query.lower().split())
            relevance_scores = []
            
            for chunk in retrieved_chunks:
                content = chunk.get('content', '').lower()
                content_terms = set(content.split())
                
                if query_terms:
                    overlap = len(query_terms.intersection(content_terms))
                    relevance = overlap / len(query_terms)
                    relevance_scores.append(relevance)
            
            if relevance_scores:
                avg_relevance = mean(relevance_scores)
                factors.append(avg_relevance)
            
            # Factor 3: Result consistency
            if len(scores) > 1:
                score_stdev = stdev(scores)
                consistency = 1.0 - min(score_stdev, 1.0)  # Lower stdev = higher consistency
                factors.append(consistency)
            
            # Factor 4: Source diversity
            doc_ids = set(chunk.get('doc_id') for chunk in retrieved_chunks)
            diversity = min(len(doc_ids) / max(1, len(retrieved_chunks)), 1.0)
            factors.append(diversity)
            
            # Calculate overall retrieval confidence
            return mean(factors) if factors else 0.0
            
        except Exception as e:
            logger.warning(f"Retrieval confidence calculation failed: {str(e)}")
            return 0.3  # Default moderate confidence
    
    async def calculate_reasoning_confidence(
        self,
        query: str,
        reasoning_steps: List[ReasoningStep],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> ConfidenceScore:
        """Calculate confidence for reasoning process."""
        try:
            factors = {}
            
            # Factor 1: Reasoning step confidence
            if reasoning_steps:
                step_confidences = [step.confidence for step in reasoning_steps]
                factors['reasoning_coherence'] = mean(step_confidences)
            else:
                factors['reasoning_coherence'] = 0.0
            
            # Factor 2: Retrieval confidence
            factors['retrieval_quality'] = await self.calculate_retrieval_confidence(
                retrieved_chunks, query
            )
            
            # Factor 3: Evidence strength
            factors['evidence_strength'] = self._calculate_evidence_strength(reasoning_steps)
            
            # Factor 4: Source reliability
            factors['source_reliability'] = self._calculate_source_reliability(retrieved_chunks)
            
            # Factor 5: Multi-document consensus (if applicable)
            doc_ids = set(chunk.get('doc_id') for chunk in retrieved_chunks)
            if len(doc_ids) > 1:
                factors['multi_document_consensus'] = self._calculate_consensus_score(retrieved_chunks)
            else:
                factors['multi_document_consensus'] = 0.5  # Neutral for single document
            
            # Prepare evidence for overall calculation
            evidence = [
                {
                    'type': 'reasoning_step',
                    'confidence': step.confidence,
                    'sources': step.sources
                }
                for step in reasoning_steps
            ]
            
            return await self.calculate_overall_confidence(factors, evidence)
            
        except Exception as e:
            logger.error(f"Reasoning confidence calculation failed: {str(e)}")
            return ConfidenceScore(
                overall_confidence=0.2,
                factor_scores={'error': 0.2},
                evidence_count=0,
                source_diversity=0.0,
                uncertainty_level=0.8
            )
    
    def _normalize_factors(self, factors: Dict[str, float]) -> Dict[str, float]:
        """Normalize confidence factors to [0, 1] range."""
        normalized = {}
        
        for factor, value in factors.items():
            if isinstance(value, (int, float)):
                normalized[factor] = max(0.0, min(1.0, float(value)))
            else:
                normalized[factor] = 0.0
        
        return normalized
    
    def _calculate_weighted_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate weighted confidence from factors."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            weight = self.factor_weights.get(factor, 0.1)  # Default weight
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / max(total_weight, 0.1)
    
    def _calculate_evidence_boost(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate confidence boost from evidence strength."""
        if not evidence:
            return 0.0
        
        # More evidence generally increases confidence
        evidence_count_factor = min(len(evidence) / 5.0, 1.0)  # Normalize to 5 pieces of evidence
        
        # Quality of evidence
        quality_scores = []
        for item in evidence:
            confidence = item.get('confidence', 0.5)
            quality_scores.append(confidence)
        
        avg_quality = mean(quality_scores) if quality_scores else 0.5
        
        # Combined evidence boost
        evidence_boost = (evidence_count_factor * avg_quality) * self.calibration_params['evidence_boost']
        
        return evidence_boost
    
    def _calculate_consensus_boost(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate confidence boost from multi-source consensus."""
        if not evidence:
            return 0.0
        
        # Extract sources from evidence
        sources = set()
        for item in evidence:
            item_sources = item.get('sources', [])
            if isinstance(item_sources, list):
                sources.update(item_sources)
            elif isinstance(item_sources, str):
                sources.add(item_sources)
        
        # More diverse sources = higher consensus boost
        source_diversity = min(len(sources) / 3.0, 1.0)  # Normalize to 3 sources
        
        consensus_boost = source_diversity * self.calibration_params['consensus_boost']
        
        return consensus_boost
    
    def _calculate_uncertainty_penalty(self, factors: Dict[str, float]) -> float:
        """Calculate penalty for high uncertainty."""
        if not factors:
            return self.calibration_params['uncertainty_penalty']
        
        # Calculate variance in factor scores
        factor_values = list(factors.values())
        if len(factor_values) > 1:
            factor_variance = stdev(factor_values) ** 2
            uncertainty_penalty = factor_variance * self.calibration_params['uncertainty_penalty']
        else:
            uncertainty_penalty = 0.0
        
        return uncertainty_penalty
    
    def _calculate_source_diversity(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate diversity of information sources."""
        if not evidence:
            return 0.0
        
        sources = set()
        for item in evidence:
            item_sources = item.get('sources', [])
            if isinstance(item_sources, list):
                sources.update(item_sources)
            elif isinstance(item_sources, str):
                sources.add(item_sources)
        
        # Diversity score based on unique sources
        max_expected_sources = 5  # Reasonable maximum
        diversity = min(len(sources) / max_expected_sources, 1.0)
        
        return diversity
    
    def _calculate_uncertainty_level(self, factors: Dict[str, float]) -> float:
        """Calculate overall uncertainty level."""
        if not factors:
            return 1.0
        
        factor_values = list(factors.values())
        
        # Uncertainty based on:
        # 1. Low average confidence
        avg_confidence = mean(factor_values)
        confidence_uncertainty = 1.0 - avg_confidence
        
        # 2. High variance in factors
        if len(factor_values) > 1:
            variance_uncertainty = stdev(factor_values)
        else:
            variance_uncertainty = 0.0
        
        # Combine uncertainties
        total_uncertainty = (confidence_uncertainty + variance_uncertainty) / 2
        
        return min(total_uncertainty, 1.0)
    
    def _calculate_evidence_strength(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate strength of evidence from reasoning steps."""
        if not reasoning_steps:
            return 0.0
        
        strength_factors = []
        
        for step in reasoning_steps:
            # Factor 1: Step confidence
            step_confidence = step.confidence
            
            # Factor 2: Source count
            source_count = len(step.sources) if step.sources else 0
            source_factor = min(source_count / 2.0, 1.0)  # Normalize to 2 sources
            
            # Factor 3: Output data richness
            output_richness = 0.5  # Default
            if isinstance(step.output_data, dict):
                output_richness = min(len(step.output_data) / 5.0, 1.0)  # Normalize to 5 fields
            
            # Combine factors for this step
            step_strength = (step_confidence + source_factor + output_richness) / 3
            strength_factors.append(step_strength)
        
        return mean(strength_factors)
    
    def _calculate_source_reliability(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Calculate reliability of information sources."""
        if not retrieved_chunks:
            return 0.0
        
        reliability_factors = []
        
        for chunk in retrieved_chunks:
            # Factor 1: Confidence score from extraction
            extraction_confidence = chunk.get('confidence_score', 0.5)
            
            # Factor 2: Content length (longer content often more reliable)
            content_length = len(chunk.get('content', ''))
            length_factor = min(content_length / 500.0, 1.0)  # Normalize to 500 chars
            
            # Factor 3: Document type reliability (some types more reliable)
            doc_type = chunk.get('doc_type', 'unknown')
            type_reliability = {
                'invoice': 0.9,
                'receipt': 0.8,
                'contract': 0.9,
                'resume': 0.7,
                'report': 0.8,
                'unknown': 0.5
            }.get(doc_type, 0.5)
            
            # Combine factors
            chunk_reliability = (extraction_confidence + length_factor + type_reliability) / 3
            reliability_factors.append(chunk_reliability)
        
        return mean(reliability_factors)
    
    def _calculate_consensus_score(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Calculate consensus across multiple documents."""
        if len(retrieved_chunks) < 2:
            return 0.5  # Neutral for single document
        
        # Group chunks by document
        doc_groups = {}
        for chunk in retrieved_chunks:
            doc_id = chunk.get('doc_id', 'unknown')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(chunk)
        
        if len(doc_groups) < 2:
            return 0.5  # Neutral for single document
        
        # Calculate consensus based on score consistency across documents
        doc_scores = []
        for doc_id, chunks in doc_groups.items():
            doc_avg_score = mean([chunk.get('score', 0.0) for chunk in chunks])
            doc_scores.append(doc_avg_score)
        
        # Consensus = 1 - variance in document scores
        if len(doc_scores) > 1:
            score_variance = stdev(doc_scores) ** 2
            consensus = 1.0 - min(score_variance, 1.0)
        else:
            consensus = 0.5
        
        return consensus
    
    def _update_statistics(self, confidence: float):
        """Update confidence calculation statistics."""
        # Update running average
        total = self.stats['total_calculations']
        current_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = ((current_avg * (total - 1)) + confidence) / total
        
        # Update distribution
        if confidence > 0.8:
            self.stats['confidence_distribution']['high'] += 1
        elif confidence > 0.5:
            self.stats['confidence_distribution']['medium'] += 1
        else:
            self.stats['confidence_distribution']['low'] += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get confidence calculator statistics."""
        total = self.stats['total_calculations']
        dist = self.stats['confidence_distribution']
        
        distribution_percentages = {}
        if total > 0:
            distribution_percentages = {
                'high': (dist['high'] / total) * 100,
                'medium': (dist['medium'] / total) * 100,
                'low': (dist['low'] / total) * 100
            }
        
        return {
            'total_calculations': total,
            'average_confidence': self.stats['avg_confidence'],
            'confidence_distribution': dist.copy(),
            'distribution_percentages': distribution_percentages,
            'factor_weights': self.factor_weights.copy(),
            'calibration_params': self.calibration_params.copy()
        }
