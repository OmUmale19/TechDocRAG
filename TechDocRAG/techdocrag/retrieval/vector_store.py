"""
Vector store implementation supporting multiple backends (FAISS, Chroma, Pinecone).
Provides efficient similarity search with embeddings for semantic retrieval.
"""

import asyncio
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from ..core.types import DocumentResult, DocumentChunk
from ..core.config import VectorDBConfig
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import VectorStoreError
from ..processing.embedding_generator import EmbeddingGenerator

logger = get_logger(__name__)


class VectorStore:
    """
    Vector store abstraction supporting multiple backends.
    
    Backends:
    - FAISS: For development and local deployment
    - Chroma: For production with persistence
    - Pinecone: For cloud-scale production
    
    Features:
    - Efficient similarity search
    - Batch operations
    - Metadata filtering
    - Index persistence
    - Automatic optimization
    """
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.backend = None
        self.index = None
        self.metadata_store = {}  # Document metadata
        self.doc_id_to_indices = {}  # Mapping doc_id to vector indices
        self.embedding_generator = None
        
        # Initialize backend
        self._initialize_backend()
        
        logger.info(f"VectorStore initialized with {config.provider} backend")
    
    def _initialize_backend(self):
        """Initialize the appropriate vector store backend."""
        if self.config.provider.lower() == 'faiss':
            self._initialize_faiss()
        elif self.config.provider.lower() == 'chroma':
            self._initialize_chroma()
        elif self.config.provider.lower() == 'pinecone':
            self._initialize_pinecone()
        else:
            raise VectorStoreError(f"Unsupported vector store provider: {self.config.provider}")
    
    def _initialize_faiss(self):
        """Initialize FAISS backend."""
        try:
            import faiss
            
            # Create FAISS index based on configuration
            if self.config.faiss.index_type == 'flat':
                self.index = faiss.IndexFlatIP(self.config.embedding_dimension)
            elif self.config.faiss.index_type == 'ivf':
                # IVF (Inverted File) index for larger datasets
                quantizer = faiss.IndexFlatIP(self.config.embedding_dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, 
                    self.config.embedding_dimension, 
                    self.config.faiss.nlist
                )
            elif self.config.faiss.index_type == 'hnsw':
                # HNSW (Hierarchical Navigable Small World) for fast approximate search
                self.index = faiss.IndexHNSWFlat(
                    self.config.embedding_dimension, 
                    self.config.faiss.m
                )
                self.index.hnsw.efConstruction = self.config.faiss.ef_construction
                self.index.hnsw.efSearch = self.config.faiss.ef_search
            
            # Enable GPU if configured and available
            if self.config.faiss.use_gpu and faiss.get_num_gpus() > 0:
                gpu_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, self.index)
                logger.info("FAISS: GPU acceleration enabled")
            
            self.backend = 'faiss'
            
            # Mark that we need to load the index on first use
            self.faiss_index_loaded = False
            
        except ImportError:
            raise VectorStoreError("FAISS not installed. Run: pip install faiss-cpu")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize FAISS: {str(e)}")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize Chroma client
            if self.config.chroma.persist_directory:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config.chroma.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.chroma_client = chromadb.EphemeralClient()
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.backend = 'chroma'
            
        except ImportError:
            raise VectorStoreError("ChromaDB not installed. Run: pip install chromadb")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone backend."""
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.pinecone.api_key,
                environment=self.config.pinecone.environment
            )
            
            # Connect to index
            self.index = pinecone.Index(self.config.pinecone.index_name)
            self.backend = 'pinecone'
            
        except ImportError:
            raise VectorStoreError("Pinecone not installed. Run: pip install pinecone-client")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    async def _ensure_faiss_loaded(self):
        """Ensure FAISS index is loaded before use."""
        if self.backend == 'faiss' and not self.faiss_index_loaded:
            await self._load_faiss_index()
            self.faiss_index_loaded = True
    
    async def add_document(self, document: DocumentResult):
        """Add a single document to the vector store."""
        try:
            await self.add_documents_batch([document])
            logger.debug(f"Added document {document.doc_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add document {document.doc_id}: {str(e)}")
            raise VectorStoreError(f"Failed to add document: {str(e)}") from e
    
    @performance_log("vector_store_batch_add")
    async def add_documents_batch(self, documents: List[DocumentResult]):
        """Add multiple documents in batch for efficiency."""
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Prepare embeddings and metadata
            all_embeddings = []
            all_metadata = []
            all_ids = []
            
            for document in documents:
                doc_embeddings, doc_metadata, doc_ids = await self._prepare_document_embeddings(document)
                
                all_embeddings.extend(doc_embeddings)
                all_metadata.extend(doc_metadata)
                all_ids.extend(doc_ids)
            
            # Add to appropriate backend
            if self.backend == 'faiss':
                await self._ensure_faiss_loaded()
                await self._add_to_faiss(all_embeddings, all_metadata, all_ids)
            elif self.backend == 'chroma':
                await self._add_to_chroma(all_embeddings, all_metadata, all_ids)
            elif self.backend == 'pinecone':
                await self._add_to_pinecone(all_embeddings, all_metadata, all_ids)
            
            logger.info(f"Successfully added {len(all_embeddings)} vectors to {self.backend}")
            
        except Exception as e:
            logger.error(f"Failed to add documents batch: {str(e)}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}") from e
    
    async def add_chunks(self, chunks: List[DocumentChunk], doc_id: str = "default"):
        """Add document chunks directly to the vector store."""
        try:
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            
            # Prepare embeddings and metadata from chunks
            embeddings = []
            metadata = []
            ids = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue
                    
                embeddings.append(np.array(chunk.embedding))
                metadata.append({
                    "chunk_id": chunk.id,
                    "doc_id": doc_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_number
                })
                ids.append(chunk.id)
            
            if not embeddings:
                logger.warning("No chunks with embeddings found")
                return
            
            # Add to appropriate backend
            if self.backend == 'faiss':
                await self._add_to_faiss(embeddings, metadata, ids)
            elif self.backend == 'chroma':
                await self._add_to_chroma(embeddings, metadata, ids)
            elif self.backend == 'pinecone':
                await self._add_to_pinecone(embeddings, metadata, ids)
            
            logger.info(f"Successfully added {len(embeddings)} chunks to {self.backend}")
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            raise VectorStoreError(f"Failed to add chunks: {str(e)}") from e
    
    async def _prepare_document_embeddings(
        self, 
        document: DocumentResult
    ) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """Prepare embeddings and metadata for a document."""
        embeddings = []
        metadata = []
        ids = []
        
        # Initialize embedding generator if needed
        if self.embedding_generator is None:
            # We'll need to import this properly
            from ..processing.embedding_generator import EmbeddingGenerator
            from ..core.config import Config
            # This is a simplified approach - in production, pass config properly
            self.embedding_generator = EmbeddingGenerator(Config().embedding)
        
        # Process each chunk
        for chunk in document.chunks:
            # Generate embedding for chunk
            embeddings_result = await self.embedding_generator.generate_embeddings([chunk])
            
            if embeddings_result and len(embeddings_result) > 0:
                chunk_with_embedding = embeddings_result[0]
                if chunk_with_embedding.embedding:
                    embeddings.append(chunk_with_embedding.embedding)
                    
                    # Prepare metadata
                    chunk_metadata = {
                        'doc_id': document.doc_id,
                        'chunk_id': chunk.id,
                        'chunk_type': chunk.chunk_type,
                        'page_number': chunk.page_number,
                        'content': chunk.content,
                    'doc_type': document.doc_type,
                    'file_path': document.file_path,
                    'processed_date': document.processed_at.isoformat() if document.processed_at else None
                }
                
                metadata.append(chunk_metadata)
                ids.append(f"{document.doc_id}_{chunk.id}")
        
        # Track document to indices mapping
        start_idx = len(self.metadata_store)
        end_idx = start_idx + len(embeddings)
        self.doc_id_to_indices[document.doc_id] = list(range(start_idx, end_idx))
        
        return embeddings, metadata, ids
    
    async def _add_to_faiss(
        self, 
        embeddings: List[np.ndarray], 
        metadata: List[Dict], 
        ids: List[str]
    ):
        """Add vectors to FAISS index."""
        if not embeddings:
            return
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'ntotal') and self.index.ntotal == 0 and hasattr(self.index, 'train'):
            if len(embeddings) >= self.config.faiss.nlist:
                self.index.train(embedding_matrix)
            else:
                logger.warning("Not enough vectors to train IVF index")
        
        # Add vectors
        start_idx = self.index.ntotal
        self.index.add(embedding_matrix)
        
        # Store metadata
        for i, (metadata_item, vector_id) in enumerate(zip(metadata, ids)):
            self.metadata_store[start_idx + i] = {
                **metadata_item,
                'vector_id': vector_id
            }
        
        # Save index periodically
        if self.index.ntotal % 1000 == 0:
            await self._save_faiss_index()
    
    async def _add_to_chroma(
        self, 
        embeddings: List[np.ndarray], 
        metadata: List[Dict], 
        ids: List[str]
    ):
        """Add vectors to ChromaDB collection."""
        if not embeddings:
            return
        
        # Convert embeddings to list format
        embedding_list = [emb.tolist() for emb in embeddings]
        
        # Extract content for ChromaDB
        documents = [meta['content'] for meta in metadata]
        
        # Clean metadata (remove content to avoid duplication)
        clean_metadata = []
        for meta in metadata:
            clean_meta = {k: v for k, v in meta.items() if k != 'content' and v is not None}
            # Convert all values to strings (ChromaDB requirement)
            clean_meta = {k: str(v) for k, v in clean_meta.items()}
            clean_metadata.append(clean_meta)
        
        # Add to collection
        self.collection.add(
            embeddings=embedding_list,
            documents=documents,
            metadatas=clean_metadata,
            ids=ids
        )
    
    async def _add_to_pinecone(
        self, 
        embeddings: List[np.ndarray], 
        metadata: List[Dict], 
        ids: List[str]
    ):
        """Add vectors to Pinecone index."""
        if not embeddings:
            return
        
        # Prepare vectors for Pinecone
        vectors = []
        for emb, meta, vector_id in zip(embeddings, metadata, ids):
            # Clean metadata (remove large content)
            clean_meta = {k: v for k, v in meta.items() if k != 'content' and v is not None}
            clean_meta = {k: str(v)[:500] for k, v in clean_meta.items()}  # Limit string length
            
            vectors.append({
                'id': vector_id,
                'values': emb.tolist(),
                'metadata': clean_meta
            })
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    @performance_log("vector_search")
    async def search(
        self,
        query: str,
        doc_ids: List[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            # Generate query embedding
            if self.embedding_generator is None:
                from ..processing.embedding_generator import EmbeddingGenerator
                from ..core.config import Config
                self.embedding_generator = EmbeddingGenerator(Config().embedding)
            
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            if not query_embedding or len(query_embedding) == 0:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Convert to numpy array for FAISS
            query_embedding = np.array(query_embedding)
            
            # Search in appropriate backend
            if self.backend == 'faiss':
                await self._ensure_faiss_loaded()
                return await self._search_faiss(query_embedding, doc_ids, top_k, similarity_threshold)
            elif self.backend == 'chroma':
                return await self._search_chroma(query_embedding, doc_ids, top_k, similarity_threshold)
            elif self.backend == 'pinecone':
                return await self._search_pinecone(query_embedding, doc_ids, top_k, similarity_threshold)
            
            return []
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}") from e
    
    async def _search_faiss(
        self,
        query_embedding: np.ndarray,
        doc_ids: List[str],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < similarity_threshold:
                continue
            
            if idx in self.metadata_store:
                metadata = self.metadata_store[idx]
                
                # Filter by doc_ids if specified
                if doc_ids and metadata.get('doc_id') not in doc_ids:
                    continue
                
                result = {
                    'doc_id': metadata.get('doc_id'),
                    'chunk_id': metadata.get('chunk_id'),
                    'content': metadata.get('content', ''),
                    'score': float(score),
                    'chunk_type': metadata.get('chunk_type'),
                    'page_number': metadata.get('page_number'),
                    'confidence_score': metadata.get('confidence_score'),
                    'search_type': 'semantic'
                }
                
                results.append(result)
        
        return results[:top_k]
    
    async def _search_chroma(
        self,
        query_embedding: np.ndarray,
        doc_ids: List[str],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        # Prepare where clause for filtering
        where_clause = None
        if doc_ids:
            where_clause = {"doc_id": {"$in": doc_ids}}
        
        # Query collection
        query_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        results = []
        if query_results['ids'] and len(query_results['ids']) > 0:
            for i, (doc_id, distance, metadata, document) in enumerate(zip(
                query_results['ids'][0],
                query_results['distances'][0],
                query_results['metadatas'][0],
                query_results['documents'][0]
            )):
                # Convert distance to similarity score (ChromaDB uses distance)
                similarity = 1.0 - distance
                
                if similarity < similarity_threshold:
                    continue
                
                result = {
                    'doc_id': metadata.get('doc_id'),
                    'chunk_id': metadata.get('chunk_id'),
                    'content': document,
                    'score': similarity,
                    'chunk_type': metadata.get('chunk_type'),
                    'page_number': int(metadata.get('page_number', 0)) if metadata.get('page_number') else None,
                    'confidence_score': float(metadata.get('confidence_score', 0)) if metadata.get('confidence_score') else None,
                    'search_type': 'semantic'
                }
                
                results.append(result)
        
        return results
    
    async def _search_pinecone(
        self,
        query_embedding: np.ndarray,
        doc_ids: List[str],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search Pinecone index."""
        # Prepare filter
        filter_dict = {}
        if doc_ids:
            filter_dict['doc_id'] = {'$in': doc_ids}
        
        # Query Pinecone
        query_results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        results = []
        for match in query_results['matches']:
            if match['score'] < similarity_threshold:
                continue
            
            metadata = match.get('metadata', {})
            
            result = {
                'doc_id': metadata.get('doc_id'),
                'chunk_id': metadata.get('chunk_id'),
                'content': metadata.get('content', ''),
                'score': match['score'],
                'chunk_type': metadata.get('chunk_type'),
                'page_number': int(metadata.get('page_number', 0)) if metadata.get('page_number') else None,
                'confidence_score': float(metadata.get('confidence_score', 0)) if metadata.get('confidence_score') else None,
                'search_type': 'semantic'
            }
            
            results.append(result)
        
        return results
    
    async def _save_faiss_index(self):
        """Save FAISS index to disk."""
        if self.backend == 'faiss' and self.config.faiss.index_path:
            try:
                index_path = Path(self.config.faiss.index_path)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save index
                import faiss
                faiss.write_index(self.index, str(index_path))
                
                # Save metadata
                metadata_path = index_path.with_suffix('.metadata.pkl')
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata_store': self.metadata_store,
                        'doc_id_to_indices': self.doc_id_to_indices
                    }, f)
                
                logger.debug(f"FAISS index saved to {index_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {str(e)}")
    
    async def _load_faiss_index(self):
        """Load FAISS index from disk."""
        if self.backend == 'faiss' and self.config.faiss.index_path:
            try:
                index_path = Path(self.config.faiss.index_path)
                metadata_path = index_path.with_suffix('.metadata.pkl')
                
                if index_path.exists() and metadata_path.exists():
                    # Load index
                    import faiss
                    self.index = faiss.read_index(str(index_path))
                    
                    # Load metadata
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.metadata_store = data['metadata_store']
                        self.doc_id_to_indices = data['doc_id_to_indices']
                    
                    logger.info(f"FAISS index loaded from {index_path} ({self.index.ntotal} vectors)")
                
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'backend': self.backend,
            'total_vectors': 0,
            'total_documents': len(self.doc_id_to_indices)
        }
        
        if self.backend == 'faiss':
            stats['total_vectors'] = self.index.ntotal if self.index else 0
            stats['index_type'] = self.config.faiss.index_type
        elif self.backend == 'chroma':
            stats['total_vectors'] = self.collection.count()
            stats['collection_name'] = self.config.collection_name
        elif self.backend == 'pinecone':
            index_stats = self.index.describe_index_stats()
            stats['total_vectors'] = index_stats.get('total_vector_count', 0)
            stats['index_name'] = self.config.pinecone.index_name
        
        return stats
    
    async def clear(self):
        """Clear all vectors from the store."""
        try:
            if self.backend == 'faiss':
                # Reset FAISS index
                self.index.reset()
                self.metadata_store.clear()
                self.doc_id_to_indices.clear()
            elif self.backend == 'chroma':
                # Delete and recreate collection
                self.chroma_client.delete_collection(self.config.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            elif self.backend == 'pinecone':
                # Delete all vectors (this is expensive in Pinecone)
                self.index.delete(delete_all=True)
            
            logger.info(f"Vector store ({self.backend}) cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            raise VectorStoreError(f"Failed to clear store: {str(e)}") from e
    
    async def optimize(self):
        """Optimize the vector store for better performance."""
        try:
            if self.backend == 'faiss':
                # Save index
                await self._save_faiss_index()
                logger.info("FAISS index optimized and saved")
            elif self.backend == 'chroma':
                # ChromaDB handles optimization automatically
                logger.info("ChromaDB optimization requested (handled automatically)")
            elif self.backend == 'pinecone':
                # Pinecone handles optimization automatically
                logger.info("Pinecone optimization requested (handled automatically)")
                
        except Exception as e:
            logger.warning(f"Vector store optimization failed: {str(e)}")
    
    async def delete_document(self, doc_id: str):
        """Delete all vectors for a specific document."""
        try:
            if self.backend == 'faiss':
                # FAISS doesn't support deletion, would need to rebuild index
                logger.warning("FAISS doesn't support vector deletion")
            elif self.backend == 'chroma':
                # Delete by doc_id
                self.collection.delete(where={"doc_id": doc_id})
            elif self.backend == 'pinecone':
                # Delete by doc_id filter
                self.index.delete(filter={"doc_id": doc_id})
            
            # Clean up local mappings
            if doc_id in self.doc_id_to_indices:
                del self.doc_id_to_indices[doc_id]
            
            logger.debug(f"Deleted vectors for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            raise VectorStoreError(f"Failed to delete document: {str(e)}") from e
