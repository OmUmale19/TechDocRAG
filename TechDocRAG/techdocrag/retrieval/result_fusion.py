"""
Result fusion algorithms for combining semantic and keyword search results.
Implements Reciprocal Rank Fusion (RRF) and other state-of-the-art fusion techniques.
"""

import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..core.config import RetrievalConfig
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class ResultFusion:
    """
    Advanced result fusion for combining multiple retrieval systems.
    
    Fusion Algorithms:
    1. Reciprocal Rank Fusion (RRF) - State-of-the-art rank fusion
    2. Weighted Score Combination - Simple score-based fusion
    3. Borda Count - Positional voting system
    4. CombSUM/CombMNZ - Classical IR fusion methods
    5. Adaptive Fusion - Context-aware weight adjustment
    
    Features:
    - Multiple fusion strategies
    - Dynamic weight adjustment
    - Duplicate handling
    - Score normalization
    - Performance optimization
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Fusion parameters
        self.rrf_k = config.rrf_k  # RRF parameter (typically 60)
        self.semantic_weight = config.semantic_weight  # Weight for semantic results
        self.keyword_weight = config.keyword_weight    # Weight for keyword results
        
        # Fusion statistics
        self.fusion_stats = {
            'total_fusions': 0,
            'avg_semantic_results': 0.0,
            'avg_keyword_results': 0.0,
            'avg_final_results': 0.0,
            'fusion_improvements': 0
        }
        
        logger.info(f"ResultFusion initialized with algorithm: {config.fusion_algorithm}")
    
    @performance_log("result_fusion")
    async def fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int = 10,
        algorithm: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and keyword search results.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            top_k: Number of final results to return
            algorithm: Fusion algorithm to use
            
        Returns:
            Fused and ranked results
        """
        self.fusion_stats['total_fusions'] += 1
        self.fusion_stats['avg_semantic_results'] = self._update_average(
            self.fusion_stats['avg_semantic_results'], 
            len(semantic_results),
            self.fusion_stats['total_fusions']
        )
        self.fusion_stats['avg_keyword_results'] = self._update_average(
            self.fusion_stats['avg_keyword_results'], 
            len(keyword_results),
            self.fusion_stats['total_fusions']
        )
        
        try:
            # Use configured algorithm if not specified
            if algorithm is None:
                algorithm = self.config.fusion_algorithm
            
            logger.debug(f"Fusing results: {len(semantic_results)} semantic + {len(keyword_results)} keyword")
            
            # Handle edge cases
            if not semantic_results and not keyword_results:
                return []
            
            if not semantic_results:
                return await self._post_process_results(keyword_results[:top_k])
            
            if not keyword_results:
                return await self._post_process_results(semantic_results[:top_k])
            
            # Apply fusion algorithm
            if algorithm.lower() == 'rrf':
                fused_results = await self._reciprocal_rank_fusion(
                    semantic_results, keyword_results, top_k
                )
            elif algorithm.lower() == 'weighted_score':
                fused_results = await self._weighted_score_fusion(
                    semantic_results, keyword_results, top_k
                )
            elif algorithm.lower() == 'borda_count':
                fused_results = await self._borda_count_fusion(
                    semantic_results, keyword_results, top_k
                )
            elif algorithm.lower() == 'combsum':
                fused_results = await self._combsum_fusion(
                    semantic_results, keyword_results, top_k
                )
            elif algorithm.lower() == 'adaptive':
                fused_results = await self._adaptive_fusion(
                    semantic_results, keyword_results, top_k
                )
            else:
                logger.warning(f"Unknown fusion algorithm: {algorithm}, using RRF")
                fused_results = await self._reciprocal_rank_fusion(
                    semantic_results, keyword_results, top_k
                )
            
            # Post-process results
            final_results = await self._post_process_results(fused_results)
            
            self.fusion_stats['avg_final_results'] = self._update_average(
                self.fusion_stats['avg_final_results'], 
                len(final_results),
                self.fusion_stats['total_fusions']
            )
            
            logger.debug(f"Fusion complete: {len(final_results)} final results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {str(e)}")
            # Fallback to simple concatenation
            all_results = semantic_results + keyword_results
            return await self._post_process_results(all_results[:top_k])
    
    async def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        RRF Score = 1 / (k + rank_semantic) + 1 / (k + rank_keyword)
        where k is typically 60.
        """
        # Create document scores dictionary
        doc_scores = defaultdict(lambda: {'rrf_score': 0.0, 'sources': set(), 'metadata': {}})
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_key = self._get_document_key(result)
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            
            doc_scores[doc_key]['rrf_score'] += rrf_contribution * self.semantic_weight
            doc_scores[doc_key]['sources'].add('semantic')
            doc_scores[doc_key]['metadata'] = result
            
            # Store original semantic score
            if 'semantic_score' not in doc_scores[doc_key]:
                doc_scores[doc_key]['semantic_score'] = result.get('score', 0.0)
                doc_scores[doc_key]['semantic_rank'] = rank
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_key = self._get_document_key(result)
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            
            doc_scores[doc_key]['rrf_score'] += rrf_contribution * self.keyword_weight
            doc_scores[doc_key]['sources'].add('keyword')
            
            # Update metadata if not from semantic search
            if 'semantic' not in doc_scores[doc_key]['sources']:
                doc_scores[doc_key]['metadata'] = result
            
            # Store original keyword score
            doc_scores[doc_key]['keyword_score'] = result.get('score', 0.0)
            doc_scores[doc_key]['keyword_rank'] = rank
        
        # Convert to list and sort by RRF score
        fused_results = []
        for doc_key, score_data in doc_scores.items():
            result = score_data['metadata'].copy()
            result['score'] = score_data['rrf_score']
            result['fusion_algorithm'] = 'rrf'
            result['sources'] = list(score_data['sources'])
            
            # Add individual scores for transparency
            if 'semantic_score' in score_data:
                result['semantic_score'] = score_data['semantic_score']
                result['semantic_rank'] = score_data['semantic_rank']
            if 'keyword_score' in score_data:
                result['keyword_score'] = score_data['keyword_score']
                result['keyword_rank'] = score_data['keyword_rank']
            
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        return fused_results[:top_k]
    
    async def _weighted_score_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Weighted score combination fusion.
        Normalizes scores and combines with weights.
        """
        # Normalize scores
        semantic_norm = self._normalize_scores(semantic_results)
        keyword_norm = self._normalize_scores(keyword_results)
        
        # Create document scores dictionary
        doc_scores = defaultdict(lambda: {'combined_score': 0.0, 'sources': set(), 'metadata': {}})
        
        # Process semantic results
        for result in semantic_norm:
            doc_key = self._get_document_key(result)
            
            doc_scores[doc_key]['combined_score'] += result['normalized_score'] * self.semantic_weight
            doc_scores[doc_key]['sources'].add('semantic')
            doc_scores[doc_key]['metadata'] = result
            doc_scores[doc_key]['semantic_score'] = result.get('score', 0.0)
        
        # Process keyword results
        for result in keyword_norm:
            doc_key = self._get_document_key(result)
            
            doc_scores[doc_key]['combined_score'] += result['normalized_score'] * self.keyword_weight
            doc_scores[doc_key]['sources'].add('keyword')
            
            # Update metadata if not from semantic search
            if 'semantic' not in doc_scores[doc_key]['sources']:
                doc_scores[doc_key]['metadata'] = result
            
            doc_scores[doc_key]['keyword_score'] = result.get('score', 0.0)
        
        # Convert to list and sort
        fused_results = []
        for doc_key, score_data in doc_scores.items():
            result = score_data['metadata'].copy()
            result['score'] = score_data['combined_score']
            result['fusion_algorithm'] = 'weighted_score'
            result['sources'] = list(score_data['sources'])
            
            if 'semantic_score' in score_data:
                result['semantic_score'] = score_data['semantic_score']
            if 'keyword_score' in score_data:
                result['keyword_score'] = score_data['keyword_score']
            
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        return fused_results[:top_k]
    
    async def _borda_count_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Borda count fusion based on ranking positions.
        """
        doc_scores = defaultdict(lambda: {'borda_score': 0, 'sources': set(), 'metadata': {}})
        
        # Maximum possible points for normalization
        max_semantic_points = len(semantic_results)
        max_keyword_points = len(keyword_results)
        
        # Process semantic results (higher rank = more points)
        for rank, result in enumerate(semantic_results):
            doc_key = self._get_document_key(result)
            points = (max_semantic_points - rank) * self.semantic_weight
            
            doc_scores[doc_key]['borda_score'] += points
            doc_scores[doc_key]['sources'].add('semantic')
            doc_scores[doc_key]['metadata'] = result
            doc_scores[doc_key]['semantic_rank'] = rank + 1
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_key = self._get_document_key(result)
            points = (max_keyword_points - rank) * self.keyword_weight
            
            doc_scores[doc_key]['borda_score'] += points
            doc_scores[doc_key]['sources'].add('keyword')
            
            if 'semantic' not in doc_scores[doc_key]['sources']:
                doc_scores[doc_key]['metadata'] = result
            
            doc_scores[doc_key]['keyword_rank'] = rank + 1
        
        # Convert to list and sort
        fused_results = []
        for doc_key, score_data in doc_scores.items():
            result = score_data['metadata'].copy()
            result['score'] = score_data['borda_score']
            result['fusion_algorithm'] = 'borda_count'
            result['sources'] = list(score_data['sources'])
            
            if 'semantic_rank' in score_data:
                result['semantic_rank'] = score_data['semantic_rank']
            if 'keyword_rank' in score_data:
                result['keyword_rank'] = score_data['keyword_rank']
            
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        return fused_results[:top_k]
    
    async def _combsum_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        CombSUM fusion - sum of normalized scores.
        """
        # Normalize scores first
        semantic_norm = self._normalize_scores(semantic_results)
        keyword_norm = self._normalize_scores(keyword_results)
        
        doc_scores = defaultdict(lambda: {'combsum_score': 0.0, 'sources': set(), 'metadata': {}})
        
        # Process all results
        all_results = [
            (result, 'semantic', self.semantic_weight) for result in semantic_norm
        ] + [
            (result, 'keyword', self.keyword_weight) for result in keyword_norm
        ]
        
        for result, source, weight in all_results:
            doc_key = self._get_document_key(result)
            
            doc_scores[doc_key]['combsum_score'] += result['normalized_score'] * weight
            doc_scores[doc_key]['sources'].add(source)
            
            # Keep first metadata encountered
            if not doc_scores[doc_key]['metadata']:
                doc_scores[doc_key]['metadata'] = result
        
        # Convert to list and sort
        fused_results = []
        for doc_key, score_data in doc_scores.items():
            result = score_data['metadata'].copy()
            result['score'] = score_data['combsum_score']
            result['fusion_algorithm'] = 'combsum'
            result['sources'] = list(score_data['sources'])
            
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        return fused_results[:top_k]
    
    async def _adaptive_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Adaptive fusion that adjusts weights based on result quality.
        """
        # Analyze result quality to adjust weights
        semantic_quality = self._assess_result_quality(semantic_results)
        keyword_quality = self._assess_result_quality(keyword_results)
        
        # Adjust weights based on quality
        total_quality = semantic_quality + keyword_quality
        if total_quality > 0:
            adaptive_semantic_weight = (semantic_quality / total_quality) * 2
            adaptive_keyword_weight = (keyword_quality / total_quality) * 2
        else:
            adaptive_semantic_weight = self.semantic_weight
            adaptive_keyword_weight = self.keyword_weight
        
        logger.debug(f"Adaptive weights: semantic={adaptive_semantic_weight:.2f}, keyword={adaptive_keyword_weight:.2f}")
        
        # Use RRF with adaptive weights
        doc_scores = defaultdict(lambda: {'adaptive_score': 0.0, 'sources': set(), 'metadata': {}})
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_key = self._get_document_key(result)
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            
            doc_scores[doc_key]['adaptive_score'] += rrf_contribution * adaptive_semantic_weight
            doc_scores[doc_key]['sources'].add('semantic')
            doc_scores[doc_key]['metadata'] = result
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_key = self._get_document_key(result)
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            
            doc_scores[doc_key]['adaptive_score'] += rrf_contribution * adaptive_keyword_weight
            doc_scores[doc_key]['sources'].add('keyword')
            
            if 'semantic' not in doc_scores[doc_key]['sources']:
                doc_scores[doc_key]['metadata'] = result
        
        # Convert to list and sort
        fused_results = []
        for doc_key, score_data in doc_scores.items():
            result = score_data['metadata'].copy()
            result['score'] = score_data['adaptive_score']
            result['fusion_algorithm'] = 'adaptive'
            result['sources'] = list(score_data['sources'])
            result['adaptive_semantic_weight'] = adaptive_semantic_weight
            result['adaptive_keyword_weight'] = adaptive_keyword_weight
            
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        return fused_results[:top_k]
    
    def _get_document_key(self, result: Dict[str, Any]) -> str:
        """Generate unique key for document/chunk."""
        doc_id = result.get('doc_id', '')
        chunk_id = result.get('chunk_id', '')
        return f"{doc_id}_{chunk_id}"
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return results
        
        scores = [result.get('score', 0.0) for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case where all scores are the same
        if max_score == min_score:
            for result in results:
                result['normalized_score'] = 1.0
            return results
        
        # Min-max normalization
        score_range = max_score - min_score
        normalized_results = []
        
        for result in results:
            normalized_result = result.copy()
            original_score = result.get('score', 0.0)
            normalized_score = (original_score - min_score) / score_range
            normalized_result['normalized_score'] = normalized_score
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    def _assess_result_quality(self, results: List[Dict[str, Any]]) -> float:
        """Assess the quality of search results."""
        if not results:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Score distribution (higher variance = better quality)
        scores = [result.get('score', 0.0) for result in results]
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            quality_factors.append(min(variance, 1.0))  # Cap at 1.0
        else:
            quality_factors.append(0.5)  # Neutral quality for single result
        
        # Factor 2: Content length diversity
        content_lengths = [len(result.get('content', '')) for result in results]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            length_quality = min(avg_length / 500.0, 1.0)  # Normalize by expected chunk size
            quality_factors.append(length_quality)
        
        # Factor 3: Result count (more results = potentially better coverage)
        count_quality = min(len(results) / 10.0, 1.0)  # Normalize by expected result count
        quality_factors.append(count_quality)
        
        # Combine quality factors
        return sum(quality_factors) / len(quality_factors)
    
    async def _post_process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process fused results."""
        processed_results = []
        
        for result in results:
            # Ensure required fields are present
            processed_result = {
                'doc_id': result.get('doc_id', ''),
                'chunk_id': result.get('chunk_id', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'search_type': 'hybrid',
                **result  # Include all other fields
            }
            
            # Add fusion metadata
            if 'fusion_algorithm' not in processed_result:
                processed_result['fusion_algorithm'] = self.config.fusion_algorithm
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average."""
        return ((current_avg * (count - 1)) + new_value) / count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        return {
            'algorithm': self.config.fusion_algorithm,
            'parameters': {
                'rrf_k': self.rrf_k,
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight
            },
            'statistics': self.fusion_stats.copy()
        }
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the fusion algorithm."""
        return {
            'name': self.config.fusion_algorithm,
            'description': self._get_algorithm_description(),
            'parameters': {
                'rrf_k': self.rrf_k,
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight
            }
        }
    
    def _get_algorithm_description(self) -> str:
        """Get description of the current fusion algorithm."""
        descriptions = {
            'rrf': 'Reciprocal Rank Fusion - Combines rankings using reciprocal ranks',
            'weighted_score': 'Weighted Score Combination - Combines normalized scores with weights',
            'borda_count': 'Borda Count - Voting system based on ranking positions',
            'combsum': 'CombSUM - Sum of normalized scores from all systems',
            'adaptive': 'Adaptive Fusion - Dynamically adjusts weights based on result quality'
        }
        
        return descriptions.get(
            self.config.fusion_algorithm.lower(), 
            'Unknown fusion algorithm'
        )
