"""
Hybrid retrieval system combining semantic and keyword search.
Implements state-of-the-art fusion techniques for optimal retrieval performance.
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.types import DocumentResult, DocumentChunk
from ..core.config import Config
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import RetrievalError
from ..processing.embedding_generator import EmbeddingGenerator

from .vector_store import VectorStore
from .keyword_searcher import KeywordSearcher
from .result_fusion import ResultFusion

logger = get_logger(__name__)


class HybridRetriever:
    """
    Advanced hybrid retrieval system combining multiple search strategies.
    
    Architecture:
    1. Semantic Search: Dense vector similarity using embeddings
    2. Keyword Search: Sparse retrieval using BM25/TF-IDF
    3. Result Fusion: Intelligent combination using RRF (Reciprocal Rank Fusion)
    4. Query Enhancement: Query expansion and reformulation
    5. Reranking: Optional neural reranking for improved precision
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize retrieval components
        self.vector_store = VectorStore(config.vector_db)
        self.keyword_searcher = KeywordSearcher(config.retrieval)
        self.result_fusion = ResultFusion(config.retrieval)
        self.embedding_generator = EmbeddingGenerator(config.embedding)
        
        # Retrieval statistics
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'semantic_hits': 0,
            'keyword_hits': 0,
            'fusion_improvements': 0
        }
        
        logger.info("HybridRetriever initialized with semantic and keyword search")
    
    @performance_log("hybrid_retrieval")
    async def retrieve(
        self,
        query: str,
        doc_ids: List[str] = None,
        top_k: int = 10,
        enable_query_expansion: bool = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Search query
            doc_ids: Optional filter for specific documents
            top_k: Number of results to return
            enable_query_expansion: Whether to expand query
            rerank: Whether to apply neural reranking
            
        Returns:
            List of retrieval results with scores and metadata
        """
        self.retrieval_stats['total_queries'] += 1
        
        try:
            logger.info(f"Hybrid retrieval for query: {query[:50]}...")
            
            # Step 1: Query preprocessing and expansion
            processed_queries = await self._preprocess_query(
                query, enable_query_expansion
            )
            
            # Step 2: Parallel semantic and keyword search
            semantic_results, keyword_results = await asyncio.gather(
                self._semantic_search(processed_queries, doc_ids, top_k),
                self._keyword_search(processed_queries, doc_ids, top_k)
            )
            
            # Step 3: Fusion of results
            fused_results = await self.result_fusion.fuse_results(
                semantic_results, keyword_results, top_k
            )
            
            # Step 4: Optional reranking
            if rerank and len(fused_results) > 1:
                fused_results = await self._rerank_results(query, fused_results)
            
            # Step 5: Post-processing and metadata enrichment
            final_results = await self._post_process_results(fused_results, query)
            
            # Update statistics
            self.retrieval_stats['semantic_hits'] += len(semantic_results)
            self.retrieval_stats['keyword_hits'] += len(keyword_results)
            
            logger.info(f"Retrieved {len(final_results)} results (semantic: {len(semantic_results)}, keyword: {len(keyword_results)})")
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise RetrievalError(f"Retrieval failed: {str(e)}") from e
    
    async def add_document(self, document: DocumentResult):
        """Add document to both vector and keyword indexes."""
        try:
            # Add to vector store
            await self.vector_store.add_document(document)
            
            # Add to keyword searcher
            await self.keyword_searcher.add_document(document)
            
            logger.debug(f"Added document {document.doc_id} to retrieval indexes")
            
        except Exception as e:
            logger.error(f"Failed to add document {document.doc_id}: {str(e)}")
            raise RetrievalError(f"Failed to add document: {str(e)}") from e
    
    async def add_documents_batch(self, documents: List[DocumentResult]):
        """Add multiple documents in batch for efficiency."""
        try:
            # Batch operations for efficiency
            await asyncio.gather(
                self.vector_store.add_documents_batch(documents),
                self.keyword_searcher.add_documents_batch(documents)
            )
            
            logger.info(f"Added {len(documents)} documents to retrieval indexes")
            
        except Exception as e:
            logger.error(f"Failed to add documents batch: {str(e)}")
            raise RetrievalError(f"Failed to add documents: {str(e)}") from e
    
    async def _preprocess_query(
        self, 
        query: str, 
        enable_expansion: bool = None
    ) -> Dict[str, str]:
        """Preprocess and optionally expand the query."""
        if enable_expansion is None:
            enable_expansion = self.config.retrieval.enable_query_expansion
        
        processed_queries = {'original': query.strip()}
        
        if enable_expansion:
            # Query expansion techniques
            expanded_queries = await self._expand_query(query)
            processed_queries.update(expanded_queries)
        
        return processed_queries
    
    async def _expand_query(self, query: str) -> Dict[str, str]:
        """Expand query using various techniques."""
        expanded = {}
        
        try:
            # Technique 1: Synonym expansion (simplified)
            synonyms = await self._get_synonyms(query)
            if synonyms:
                expanded['with_synonyms'] = f"{query} {' '.join(synonyms)}"
            
            # Technique 2: Hypothetical document generation
            if self.config.retrieval.enable_hypothetical_questions:
                hypothetical = await self._generate_hypothetical_document(query)
                if hypothetical:
                    expanded['hypothetical'] = hypothetical
            
            # Technique 3: Query reformulation (domain-specific)
            reformulated = await self._reformulate_query(query)
            if reformulated:
                expanded['reformulated'] = reformulated
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
        
        return expanded
    
    async def _get_synonyms(self, query: str) -> List[str]:
        """Get synonyms for query terms (simplified implementation)."""
        # This is a simplified implementation
        # In production, you'd use WordNet, domain-specific thesaurus, or LLM
        
        synonym_map = {
            'invoice': ['bill', 'receipt', 'statement'],
            'total': ['sum', 'amount', 'cost'],
            'date': ['time', 'when', 'period'],
            'company': ['business', 'organization', 'firm'],
            'experience': ['background', 'history', 'expertise'],
            'skill': ['ability', 'competency', 'capability']
        }
        
        synonyms = []
        query_words = query.lower().split()
        
        for word in query_words:
            if word in synonym_map:
                synonyms.extend(synonym_map[word])
        
        return synonyms[:3]  # Limit to avoid query bloat
    
    async def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """Generate hypothetical document that would answer the query."""
        # This would typically use an LLM to generate a hypothetical document
        # For now, return a simple template
        
        hypothetical_templates = {
            'what': f"This document contains information about {query.replace('what', '').strip()}",
            'how': f"This document explains how to {query.replace('how', '').strip()}",
            'when': f"This document specifies when {query.replace('when', '').strip()}",
            'where': f"This document indicates where {query.replace('where', '').strip()}",
            'why': f"This document explains why {query.replace('why', '').strip()}"
        }
        
        query_lower = query.lower()
        for question_word, template in hypothetical_templates.items():
            if query_lower.startswith(question_word):
                return template
        
        return None
    
    async def _reformulate_query(self, query: str) -> Optional[str]:
        """Reformulate query for better retrieval (domain-specific)."""
        # Domain-specific query reformulation patterns
        reformulation_patterns = [
            # Invoice-related
            (r'total amount', 'invoice total sum amount due'),
            (r'due date', 'payment due date deadline'),
            (r'vendor', 'supplier company business'),
            
            # Resume-related  
            (r'experience', 'work experience employment history'),
            (r'skills', 'technical skills abilities competencies'),
            (r'education', 'degree university college education'),
            
            # Legal-related
            (r'contract', 'agreement legal document terms'),
            (r'clause', 'contract clause provision term'),
        ]
        
        import re
        reformulated = query
        
        for pattern, replacement in reformulation_patterns:
            reformulated = re.sub(pattern, replacement, reformulated, flags=re.IGNORECASE)
        
        return reformulated if reformulated != query else None
    
    async def _semantic_search(
        self, 
        queries: Dict[str, str], 
        doc_ids: List[str], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        try:
            # Search with original query first
            primary_results = await self.vector_store.search(
                queries['original'], doc_ids, top_k
            )
            
            # If we have expanded queries, search with those too
            all_results = primary_results.copy()
            
            for query_type, expanded_query in queries.items():
                if query_type != 'original':
                    expanded_results = await self.vector_store.search(
                        expanded_query, doc_ids, top_k // 2
                    )
                    # Adjust scores for expanded queries (slightly lower weight)
                    for result in expanded_results:
                        result['score'] *= 0.9
                        result['query_type'] = query_type
                    
                    all_results.extend(expanded_results)
            
            # Deduplicate and re-rank
            unique_results = self._deduplicate_results(all_results)
            
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def _keyword_search(
        self, 
        queries: Dict[str, str], 
        doc_ids: List[str], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25/TF-IDF."""
        try:
            # Search with all query variants
            all_results = []
            
            for query_type, query_text in queries.items():
                results = await self.keyword_searcher.search(
                    query_text, doc_ids, top_k
                )
                
                # Adjust scores for non-original queries
                score_multiplier = 1.0 if query_type == 'original' else 0.8
                for result in results:
                    result['score'] *= score_multiplier
                    result['query_type'] = query_type
                
                all_results.extend(results)
            
            # Deduplicate and re-rank
            unique_results = self._deduplicate_results(all_results)
            
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results, keeping the highest scoring version."""
        seen = {}
        
        for result in results:
            key = f"{result.get('doc_id')}_{result.get('chunk_id')}"
            
            if key not in seen or result['score'] > seen[key]['score']:
                seen[key] = result
        
        return list(seen.values())
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply neural reranking to improve result quality."""
        try:
            # This is a placeholder for neural reranking
            # In production, you'd use a cross-encoder model
            
            # For now, implement a simple relevance boost based on content analysis
            enhanced_results = []
            
            for result in results:
                content = result.get('content', '')
                
                # Simple relevance scoring based on query term frequency
                query_terms = set(query.lower().split())
                content_terms = set(content.lower().split())
                
                # Calculate term overlap
                overlap = len(query_terms.intersection(content_terms))
                overlap_score = overlap / len(query_terms) if query_terms else 0
                
                # Boost score based on term overlap
                boosted_score = result['score'] * (1 + overlap_score * 0.2)
                
                result['rerank_score'] = boosted_score
                result['term_overlap'] = overlap_score
                
                enhanced_results.append(result)
            
            return sorted(enhanced_results, key=lambda x: x['rerank_score'], reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return results
    
    async def _post_process_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Post-process results with additional metadata and snippets."""
        processed_results = []
        
        for result in results:
            # Add query-specific highlighting (simplified)
            content = result.get('content', '')
            highlighted_content = self._highlight_query_terms(content, query)
            
            # Calculate additional relevance signals
            relevance_signals = self._calculate_relevance_signals(result, query)
            
            processed_result = {
                **result,
                'highlighted_content': highlighted_content,
                'relevance_signals': relevance_signals,
                'snippet_length': len(highlighted_content),
                'query_match_strength': relevance_signals.get('query_match_strength', 0.0)
            }
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def _highlight_query_terms(self, content: str, query: str) -> str:
        """Highlight query terms in content (simplified)."""
        import re
        
        query_terms = query.lower().split()
        highlighted = content
        
        for term in query_terms:
            if len(term) > 2:  # Only highlight meaningful terms
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f"**{term}**", highlighted)
        
        return highlighted
    
    def _calculate_relevance_signals(
        self, 
        result: Dict[str, Any], 
        query: str
    ) -> Dict[str, float]:
        """Calculate additional relevance signals."""
        content = result.get('content', '').lower()
        query_lower = query.lower()
        
        # Query term frequency
        query_terms = query_lower.split()
        term_frequencies = []
        
        for term in query_terms:
            if len(term) > 2:
                freq = content.count(term) / max(1, len(content.split()))
                term_frequencies.append(freq)
        
        avg_term_freq = sum(term_frequencies) / max(1, len(term_frequencies))
        
        # Position-based relevance (terms appearing earlier are more relevant)
        first_occurrence = len(content)
        for term in query_terms:
            if term in content:
                pos = content.index(term)
                first_occurrence = min(first_occurrence, pos)
        
        position_score = 1.0 - (first_occurrence / max(1, len(content)))
        
        # Length penalty (very short or very long chunks might be less relevant)
        length_penalty = 1.0
        content_length = len(content)
        if content_length < 50:
            length_penalty = 0.8
        elif content_length > 1000:
            length_penalty = 0.9
        
        return {
            'query_match_strength': avg_term_freq,
            'position_score': position_score,
            'length_penalty': length_penalty,
            'combined_relevance': avg_term_freq * position_score * length_penalty
        }
    
    async def analyze_cross_document_relationships(
        self,
        doc_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze relationships and connections between multiple documents.
        
        Returns:
            Dictionary containing relationship analysis including:
            - Common entities
            - Date overlaps
            - Numerical correlations
            - Topic similarity
        """
        try:
            # Get all chunks for specified documents
            all_chunks = []
            for doc_id in doc_ids:
                # Retrieve chunks for this document
                doc_chunks = await self.vector_store.get_document_chunks(doc_id)
                all_chunks.extend([{**chunk, 'doc_id': doc_id} for chunk in doc_chunks])
            
            if len(all_chunks) < 2:
                return {'error': 'Need at least 2 documents for relationship analysis'}
            
            # Group by document
            doc_groups = {}
            for chunk in all_chunks:
                doc_id = chunk['doc_id']
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(chunk)
            
            # Analyze relationships
            relationships = {
                'document_count': len(doc_groups),
                'common_entities': await self._find_common_entities(doc_groups),
                'date_overlaps': await self._find_date_overlaps(doc_groups),
                'numerical_patterns': await self._find_numerical_patterns(doc_groups),
                'topic_similarity': await self._calculate_topic_similarity(doc_groups),
                'cross_references': await self._detect_cross_references(doc_groups)
            }
            
            logger.info(f"Cross-document analysis completed for {len(doc_groups)} documents")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Cross-document relationship analysis failed: {str(e)}")
            return {'error': str(e)}
    
    async def _find_common_entities(self, doc_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Find entities that appear across multiple documents."""
        import re
        
        # Extract entities from each document (simplified - capitalized words)
        doc_entities = {}
        for doc_id, chunks in doc_groups.items():
            content = ' '.join([chunk.get('content', '') for chunk in chunks])
            
            # Extract capitalized words (potential entities)
            entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))
            doc_entities[doc_id] = entities
        
        # Find common entities
        if len(doc_entities) >= 2:
            all_entities = list(doc_entities.values())
            common = set.intersection(*all_entities) if all_entities else set()
            
            # Find entities appearing in at least 50% of documents
            partial_common = {}
            all_unique_entities = set().union(*all_entities)
            
            for entity in all_unique_entities:
                count = sum(1 for ent_set in all_entities if entity in ent_set)
                if count > 1:
                    partial_common[entity] = count
            
            return {
                'fully_common': list(common),
                'partially_common': partial_common,
                'total_unique_entities': len(all_unique_entities)
            }
        
        return {'fully_common': [], 'partially_common': {}, 'total_unique_entities': 0}
    
    async def _find_date_overlaps(self, doc_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Find date mentions and overlaps across documents."""
        import re
        
        doc_dates = {}
        for doc_id, chunks in doc_groups.items():
            content = ' '.join([chunk.get('content', '') for chunk in chunks])
            
            # Extract dates (various formats)
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
            ]
            
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, content, re.IGNORECASE))
            
            doc_dates[doc_id] = dates
        
        # Analyze overlaps
        all_dates = [date for dates in doc_dates.values() for date in dates]
        common_dates = [date for date in set(all_dates) if all_dates.count(date) > 1]
        
        return {
            'dates_per_document': {doc_id: len(dates) for doc_id, dates in doc_dates.items()},
            'common_dates': common_dates,
            'total_unique_dates': len(set(all_dates))
        }
    
    async def _find_numerical_patterns(self, doc_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Find numerical patterns and correlations across documents."""
        import re
        
        doc_numbers = {}
        for doc_id, chunks in doc_groups.items():
            content = ' '.join([chunk.get('content', '') for chunk in chunks])
            
            # Extract numbers (including currency)
            number_pattern = r'[₹$€£¥]\s*[\d,]+\.?\d*|\d+\.?\d*'
            numbers = re.findall(number_pattern, content)
            
            # Clean and convert to float
            clean_numbers = []
            for num in numbers:
                clean = re.sub(r'[₹$€£¥,\s]', '', num)
                try:
                    clean_numbers.append(float(clean))
                except ValueError:
                    pass
            
            doc_numbers[doc_id] = clean_numbers
        
        # Analyze patterns
        all_numbers = [num for nums in doc_numbers.values() for num in nums]
        
        patterns = {
            'numbers_per_document': {doc_id: len(nums) for doc_id, nums in doc_numbers.items()},
            'total_numbers_found': len(all_numbers)
        }
        
        if all_numbers:
            patterns['statistics'] = {
                'min': min(all_numbers),
                'max': max(all_numbers),
                'avg': sum(all_numbers) / len(all_numbers)
            }
        
        return patterns
    
    async def _calculate_topic_similarity(self, doc_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate topic similarity between documents using embeddings."""
        try:
            # Get document-level embeddings (average of chunk embeddings)
            doc_embeddings = {}
            
            for doc_id, chunks in doc_groups.items():
                # Get embeddings for all chunks
                chunk_embeddings = []
                for chunk in chunks:
                    if 'embedding' in chunk:
                        chunk_embeddings.append(chunk['embedding'])
                
                if chunk_embeddings:
                    # Average embeddings to get document-level embedding
                    import numpy as np
                    doc_embedding = np.mean(chunk_embeddings, axis=0)
                    doc_embeddings[doc_id] = doc_embedding
            
            # Calculate pairwise similarities
            similarities = {}
            doc_ids = list(doc_embeddings.keys())
            
            for i, doc_id1 in enumerate(doc_ids):
                for doc_id2 in doc_ids[i+1:]:
                    if doc_id1 in doc_embeddings and doc_id2 in doc_embeddings:
                        # Cosine similarity
                        import numpy as np
                        emb1 = doc_embeddings[doc_id1]
                        emb2 = doc_embeddings[doc_id2]
                        
                        similarity = np.dot(emb1, emb2) / (
                            np.linalg.norm(emb1) * np.linalg.norm(emb2)
                        )
                        
                        pair_key = f"{doc_id1}↔{doc_id2}"
                        similarities[pair_key] = float(similarity)
            
            # Calculate average similarity
            avg_similarity = sum(similarities.values()) / len(similarities) if similarities else 0.0
            
            return {
                'pairwise_similarities': similarities,
                'average_similarity': avg_similarity,
                'similarity_level': 'high' if avg_similarity > 0.7 else 'medium' if avg_similarity > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.warning(f"Topic similarity calculation failed: {str(e)}")
            return {'error': str(e)}
    
    async def _detect_cross_references(self, doc_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Detect potential cross-references between documents."""
        import re
        
        # Look for reference patterns (invoice numbers, IDs, etc.)
        doc_references = {}
        
        for doc_id, chunks in doc_groups.items():
            content = ' '.join([chunk.get('content', '') for chunk in chunks])
            
            # Extract potential reference IDs
            ref_patterns = [
                r'(?:INV|INVOICE)[#-]?\s*[\w-]+',
                r'(?:REF|REFERENCE)[#-]?\s*[\w-]+',
                r'(?:ID|NUMBER)[#-]?\s*[\w-]+',
                r'[A-Z]{2,}\d{3,}'  # Generic alphanumeric IDs
            ]
            
            references = []
            for pattern in ref_patterns:
                references.extend(re.findall(pattern, content, re.IGNORECASE))
            
            doc_references[doc_id] = set(references)
        
        # Find cross-references
        cross_refs = []
        doc_ids = list(doc_references.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                common_refs = doc_references[doc_id1].intersection(doc_references[doc_id2])
                if common_refs:
                    cross_refs.append({
                        'doc1': doc_id1,
                        'doc2': doc_id2,
                        'common_references': list(common_refs)
                    })
        
        return {
            'cross_reference_count': len(cross_refs),
            'cross_references': cross_refs,
            'has_relationships': len(cross_refs) > 0
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = await self.vector_store.get_stats()
        keyword_stats = await self.keyword_searcher.get_stats()
        
        return {
            'total_queries': self.retrieval_stats['total_queries'],
            'semantic_hits': self.retrieval_stats['semantic_hits'],
            'keyword_hits': self.retrieval_stats['keyword_hits'],
            'vector_store': vector_stats,
            'keyword_searcher': keyword_stats,
            'fusion_algorithm': self.result_fusion.get_algorithm_info()
        }
    
    async def clear_indexes(self):
        """Clear all retrieval indexes."""
        await asyncio.gather(
            self.vector_store.clear(),
            self.keyword_searcher.clear()
        )
        
        logger.info("All retrieval indexes cleared")
    
    async def optimize_indexes(self):
        """Optimize retrieval indexes for better performance."""
        await asyncio.gather(
            self.vector_store.optimize(),
            self.keyword_searcher.optimize()
        )
        
        logger.info("Retrieval indexes optimized")
