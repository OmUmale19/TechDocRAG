"""
Keyword-based search implementation using BM25 and TF-IDF algorithms.
Provides sparse retrieval for exact term matching and traditional IR techniques.
"""

import asyncio
import json
import pickle
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import Counter, defaultdict
import math
import re

from ..core.types import DocumentResult, DocumentChunk
from ..core.config import RetrievalConfig
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class KeywordSearcher:
    """
    Advanced keyword search using BM25 algorithm with optional TF-IDF fallback.
    
    Features:
    - BM25 scoring for state-of-the-art keyword matching
    - TF-IDF as fallback option
    - Document and chunk-level indexing
    - Phrase search support
    - Boolean operators (AND, OR, NOT)
    - Fuzzy matching for typos
    - Custom field boosting
    - Stop word filtering
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # BM25 parameters
        self.k1 = config.bm25_k1  # Term frequency saturation parameter
        self.b = config.bm25_b    # Length normalization parameter
        
        # Search indexes
        self.document_index = {}     # doc_id -> document metadata
        self.inverted_index = defaultdict(set)  # term -> set of (doc_id, chunk_id)
        self.term_frequencies = defaultdict(lambda: defaultdict(int))  # (doc_id, chunk_id) -> {term: count}
        self.document_lengths = {}   # (doc_id, chunk_id) -> token count
        self.chunk_content = {}      # (doc_id, chunk_id) -> content
        
        # Statistics
        self.total_documents = 0
        self.average_document_length = 0.0
        self.vocabulary_size = 0
        
        # Stop words
        self.stop_words = self._load_stop_words()
        
        # Persistence
        self.index_file = Path(config.index_save_path) if config.index_save_path else None
        
        logger.info("KeywordSearcher initialized with BM25 algorithm")
        
        # Load existing index if available
        if self.index_file and self.index_file.exists():
            self._load_index()
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words for filtering."""
        # Basic English stop words
        basic_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'these', 'they', 'them',
            'their', 'there', 'where', 'when', 'what', 'who', 'how', 'why'
        }
        
        # Add domain-specific stop words
        domain_stop_words = {
            'document', 'page', 'file', 'content', 'text', 'data',
            'information', 'details', 'item', 'section', 'part'
        }
        
        return basic_stop_words.union(domain_stop_words)
    
    @performance_log("keyword_indexing")
    async def add_document(self, document: DocumentResult):
        """Add a single document to the keyword index."""
        await self.add_documents_batch([document])
    
    async def add_documents_batch(self, documents: List[DocumentResult]):
        """Add multiple documents to the keyword index."""
        try:
            logger.info(f"Indexing {len(documents)} documents for keyword search")
            
            for document in documents:
                await self._index_document(document)
            
            # Update statistics
            self._update_statistics()
            
            # Save index periodically
            if self.total_documents % 100 == 0:
                await self._save_index()
            
            logger.info(f"Indexed {len(documents)} documents. Total: {self.total_documents}")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise RetrievalError(f"Failed to index documents: {str(e)}") from e
    
    async def _index_document(self, document: DocumentResult):
        """Index a single document."""
        # Store document metadata
        self.document_index[document.doc_id] = {
            'doc_type': document.doc_type,
            'file_path': document.file_path,
            'processed_date': document.processed_at.isoformat() if document.processed_at else None,
            'chunk_count': len(document.chunks)
        }
        
        # Index each chunk
        for chunk in document.chunks:
            await self._index_chunk(document.doc_id, chunk)
    
    async def _index_chunk(self, doc_id: str, chunk: DocumentChunk):
        """Index a single chunk."""
        chunk_key = (doc_id, chunk.id)
        
        # Store chunk content
        self.chunk_content[chunk_key] = chunk.content
        
        # Tokenize and process text
        tokens = self._tokenize(chunk.content)
        
        # Remove stop words if configured
        if self.config.remove_stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Store document length
        self.document_lengths[chunk_key] = len(tokens)
        
        # Build term frequencies
        term_freq = Counter(tokens)
        self.term_frequencies[chunk_key] = dict(term_freq)
        
        # Update inverted index
        for term in term_freq.keys():
            self.inverted_index[term].add(chunk_key)
        
        # Handle field-specific indexing for structured documents
        await self._index_structured_fields(doc_id, chunk)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep numbers and basic punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out very short tokens (unless they're numbers)
        tokens = [token for token in tokens if len(token) >= 2 or token.isdigit()]
        
        # Handle compound words and hyphenated terms
        expanded_tokens = []
        for token in tokens:
            if '-' in token and len(token) > 3:
                # Add both the full term and split terms
                expanded_tokens.append(token)
                expanded_tokens.extend(token.split('-'))
            else:
                expanded_tokens.append(token)
        
        return expanded_tokens
    
    async def _index_structured_fields(self, doc_id: str, chunk: DocumentChunk):
        """Index structured fields with boosting."""
        # This would be enhanced based on extracted fields
        # For now, simple implementation based on chunk type
        
        if chunk.chunk_type in ['title', 'header']:
            # Boost important chunks by adding terms multiple times
            content = chunk.content.lower()
            tokens = self._tokenize(content)
            
            for term in set(tokens):  # Use set to avoid over-boosting
                # Add to inverted index with higher weight (simulated by multiple entries)
                chunk_key = (doc_id, chunk.id)
                for _ in range(self.config.title_boost_factor):
                    self.inverted_index[f"title:{term}"].add(chunk_key)
    
    @performance_log("keyword_search")
    async def search(
        self,
        query: str,
        doc_ids: List[str] = None,
        top_k: int = 10,
        algorithm: str = 'bm25'
    ) -> List[Dict[str, Any]]:
        """
        Search using keyword matching.
        
        Args:
            query: Search query
            doc_ids: Optional filter for specific documents
            top_k: Number of results to return
            algorithm: Scoring algorithm ('bm25' or 'tfidf')
            
        Returns:
            List of search results with scores
        """
        try:
            logger.debug(f"Keyword search: {query[:50]}... (algorithm: {algorithm})")
            
            # Parse query
            query_terms, operators = self._parse_query(query)
            
            if not query_terms:
                return []
            
            # Get candidate documents
            candidates = self._get_candidates(query_terms, operators, doc_ids)
            
            if not candidates:
                return []
            
            # Score candidates
            if algorithm.lower() == 'bm25':
                scored_results = await self._score_bm25(query_terms, candidates)
            else:
                scored_results = await self._score_tfidf(query_terms, candidates)
            
            # Sort by score
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Convert to result format
            results = await self._format_results(scored_results[:top_k])
            
            logger.debug(f"Keyword search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            raise RetrievalError(f"Keyword search failed: {str(e)}") from e
    
    def _parse_query(self, query: str) -> tuple[List[str], Dict[str, Any]]:
        """Parse query and extract terms and operators."""
        # Simple query parsing - can be enhanced with full boolean logic
        
        # Extract phrase queries (quoted terms)
        phrases = re.findall(r'"([^"]*)"', query)
        query_without_phrases = re.sub(r'"[^"]*"', '', query)
        
        # Extract regular terms
        terms = self._tokenize(query_without_phrases)
        
        # Remove stop words if configured
        if self.config.remove_stop_words:
            terms = [term for term in terms if term not in self.stop_words]
        
        # Combine terms and phrases
        all_terms = terms + phrases
        
        # Extract operators (simplified)
        operators = {
            'phrases': phrases,
            'must_include': [],  # Terms that must be present
            'must_exclude': [],  # Terms that must not be present
            'optional': terms    # Optional terms
        }
        
        # Handle simple boolean operators
        if ' AND ' in query.upper():
            operators['must_include'] = all_terms
            operators['optional'] = []
        elif ' NOT ' in query.upper():
            # Simple NOT handling
            parts = query.upper().split(' NOT ')
            if len(parts) > 1:
                excluded = self._tokenize(parts[1])
                operators['must_exclude'] = excluded
        
        return all_terms, operators
    
    def _get_candidates(
        self, 
        query_terms: List[str], 
        operators: Dict[str, Any],
        doc_ids: List[str] = None
    ) -> Set[tuple]:
        """Get candidate documents that match query terms."""
        if not query_terms:
            return set()
        
        # Get documents matching any term
        candidates = set()
        
        for term in query_terms:
            if term in self.inverted_index:
                candidates.update(self.inverted_index[term])
        
        # Handle phrase queries
        for phrase in operators.get('phrases', []):
            phrase_candidates = self._search_phrase(phrase)
            candidates.update(phrase_candidates)
        
        # Apply boolean logic
        if operators.get('must_include'):
            # All must_include terms must be present
            for term in operators['must_include']:
                if term in self.inverted_index:
                    term_docs = self.inverted_index[term]
                    candidates = candidates.intersection(term_docs)
                else:
                    candidates = set()  # Term not found, no results
                    break
        
        if operators.get('must_exclude'):
            # Remove documents containing excluded terms
            for term in operators['must_exclude']:
                if term in self.inverted_index:
                    excluded_docs = self.inverted_index[term]
                    candidates = candidates.difference(excluded_docs)
        
        # Filter by document IDs if specified
        if doc_ids:
            candidates = {(doc_id, chunk_id) for doc_id, chunk_id in candidates if doc_id in doc_ids}
        
        return candidates
    
    def _search_phrase(self, phrase: str) -> Set[tuple]:
        """Search for exact phrase matches."""
        phrase_terms = self._tokenize(phrase)
        if len(phrase_terms) < 2:
            # Single term, treat as regular term search
            if phrase_terms and phrase_terms[0] in self.inverted_index:
                return self.inverted_index[phrase_terms[0]]
            return set()
        
        # Find documents containing all terms
        candidate_docs = None
        for term in phrase_terms:
            if term in self.inverted_index:
                term_docs = self.inverted_index[term]
                if candidate_docs is None:
                    candidate_docs = term_docs.copy()
                else:
                    candidate_docs = candidate_docs.intersection(term_docs)
            else:
                return set()  # One term not found, no phrase matches
        
        # Check for actual phrase occurrence in candidates
        phrase_matches = set()
        for doc_key in candidate_docs:
            content = self.chunk_content.get(doc_key, '')
            if phrase.lower() in content.lower():
                phrase_matches.add(doc_key)
        
        return phrase_matches
    
    async def _score_bm25(
        self, 
        query_terms: List[str], 
        candidates: Set[tuple]
    ) -> List[Dict[str, Any]]:
        """Score candidates using BM25 algorithm."""
        if self.average_document_length == 0:
            self._update_statistics()
        
        scored_results = []
        
        for doc_key in candidates:
            score = 0.0
            doc_id, chunk_id = doc_key
            
            # Get document length
            doc_length = self.document_lengths.get(doc_key, 1)
            
            # Calculate BM25 score for each term
            for term in query_terms:
                if term in self.term_frequencies[doc_key]:
                    # Term frequency in document
                    tf = self.term_frequencies[doc_key][term]
                    
                    # Document frequency (number of documents containing term)
                    df = len(self.inverted_index[term])
                    
                    # Inverse document frequency
                    idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
                    
                    term_score = idf * (numerator / denominator)
                    score += term_score
            
            if score > 0:
                scored_results.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'score': score,
                    'doc_key': doc_key
                })
        
        return scored_results
    
    async def _score_tfidf(
        self, 
        query_terms: List[str], 
        candidates: Set[tuple]
    ) -> List[Dict[str, Any]]:
        """Score candidates using TF-IDF algorithm."""
        scored_results = []
        
        for doc_key in candidates:
            score = 0.0
            doc_id, chunk_id = doc_key
            
            # Get document length for normalization
            doc_length = self.document_lengths.get(doc_key, 1)
            
            for term in query_terms:
                if term in self.term_frequencies[doc_key]:
                    # Term frequency
                    tf = self.term_frequencies[doc_key][term] / doc_length
                    
                    # Inverse document frequency
                    df = len(self.inverted_index[term])
                    idf = math.log(self.total_documents / (df + 1))
                    
                    # TF-IDF score
                    term_score = tf * idf
                    score += term_score
            
            if score > 0:
                scored_results.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'score': score,
                    'doc_key': doc_key
                })
        
        return scored_results
    
    async def _format_results(self, scored_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format results for output."""
        formatted_results = []
        
        for result in scored_results:
            doc_key = result['doc_key']
            doc_id = result['doc_id']
            chunk_id = result['chunk_id']
            
            # Get content
            content = self.chunk_content.get(doc_key, '')
            
            # Get document metadata
            doc_meta = self.document_index.get(doc_id, {})
            
            formatted_result = {
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'content': content,
                'score': result['score'],
                'doc_type': doc_meta.get('doc_type'),
                'file_path': doc_meta.get('file_path'),
                'search_type': 'keyword',
                'chunk_type': None,  # Would need to store this in index
                'page_number': None,  # Would need to store this in index
                'confidence_score': None
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _update_statistics(self):
        """Update search statistics."""
        self.total_documents = len(self.document_lengths)
        
        if self.total_documents > 0:
            total_length = sum(self.document_lengths.values())
            self.average_document_length = total_length / self.total_documents
        else:
            self.average_document_length = 0.0
        
        self.vocabulary_size = len(self.inverted_index)
        
        logger.debug(f"Updated statistics: {self.total_documents} documents, avg length: {self.average_document_length:.1f}")
    
    async def _save_index(self):
        """Save keyword index to disk."""
        if not self.index_file:
            return
        
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            index_data = {
                'document_index': dict(self.document_index),
                'inverted_index': {k: list(v) for k, v in self.inverted_index.items()},
                'term_frequencies': {str(k): v for k, v in self.term_frequencies.items()},
                'document_lengths': {str(k): v for k, v in self.document_lengths.items()},
                'chunk_content': {str(k): v for k, v in self.chunk_content.items()},
                'total_documents': self.total_documents,
                'average_document_length': self.average_document_length,
                'vocabulary_size': self.vocabulary_size
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.debug(f"Keyword index saved to {self.index_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save keyword index: {str(e)}")
    
    def _load_index(self):
        """Load keyword index from disk."""
        try:
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.document_index = index_data['document_index']
            self.inverted_index = defaultdict(set)
            
            # Restore inverted index
            for term, doc_list in index_data['inverted_index'].items():
                self.inverted_index[term] = set(tuple(eval(doc_key)) for doc_key in doc_list)
            
            # Restore other data structures
            self.term_frequencies = defaultdict(lambda: defaultdict(int))
            for doc_key_str, tf_dict in index_data['term_frequencies'].items():
                doc_key = tuple(eval(doc_key_str))
                self.term_frequencies[doc_key] = tf_dict
            
            self.document_lengths = {}
            for doc_key_str, length in index_data['document_lengths'].items():
                doc_key = tuple(eval(doc_key_str))
                self.document_lengths[doc_key] = length
            
            self.chunk_content = {}
            for doc_key_str, content in index_data['chunk_content'].items():
                doc_key = tuple(eval(doc_key_str))
                self.chunk_content[doc_key] = content
            
            self.total_documents = index_data['total_documents']
            self.average_document_length = index_data['average_document_length']
            self.vocabulary_size = index_data['vocabulary_size']
            
            logger.info(f"Keyword index loaded: {self.total_documents} documents, {self.vocabulary_size} terms")
            
        except Exception as e:
            logger.warning(f"Failed to load keyword index: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get keyword search statistics."""
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': self.vocabulary_size,
            'average_document_length': self.average_document_length,
            'algorithm': 'BM25',
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'remove_stop_words': self.config.remove_stop_words
            }
        }
    
    async def clear(self):
        """Clear all keyword indexes."""
        self.document_index.clear()
        self.inverted_index.clear()
        self.term_frequencies.clear()
        self.document_lengths.clear()
        self.chunk_content.clear()
        
        self.total_documents = 0
        self.average_document_length = 0.0
        self.vocabulary_size = 0
        
        logger.info("Keyword search indexes cleared")
    
    async def optimize(self):
        """Optimize keyword search indexes."""
        try:
            # Update statistics
            self._update_statistics()
            
            # Save index
            await self._save_index()
            
            # Could implement index compression or other optimizations here
            
            logger.info("Keyword search indexes optimized")
            
        except Exception as e:
            logger.warning(f"Failed to optimize keyword indexes: {str(e)}")
    
    async def get_term_suggestions(self, partial_term: str, max_suggestions: int = 10) -> List[str]:
        """Get term suggestions for autocomplete."""
        partial_term = partial_term.lower()
        suggestions = []
        
        for term in self.inverted_index.keys():
            if term.startswith(partial_term) and len(term) > len(partial_term):
                suggestions.append(term)
                if len(suggestions) >= max_suggestions:
                    break
        
        # Sort by frequency (documents containing the term)
        suggestions.sort(key=lambda t: len(self.inverted_index[t]), reverse=True)
        
        return suggestions[:max_suggestions]
