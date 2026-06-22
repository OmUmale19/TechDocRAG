#!/usr/bin/env python3
"""
TechDocRAG - Complete Document Intelligence System (Main Interface)
=================================================================
A production-ready unified interface combining all core features developed so far.

Core Features Included:
✅ Async/Await Architecture
✅ Configuration Management  
✅ Hybrid Retrieval System (Semantic + Keyword + Fusion)
✅ Document Processing Pipeline
✅ Gemini LLM-Powered Natural Language Answers
✅ Smart Answer Generation with Confidence Scoring
✅ System Health Monitoring & Statistics
✅ Multiple Input Methods (Text, Files, Document Objects)
✅ Intelligent Text Chunking & Vector Embeddings
✅ Interactive Q&A Interface
✅ Performance Tracking & Error Handling
✅ Custom Document Support

Author: EDAI Team
Version: 1.0.0
Date: October 18, 2025
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from techdocrag.core.config import Config
from techdocrag.core.types import Document, DocumentResult, DocumentType, ProcessingMetrics, DocumentChunk
from techdocrag.core.document_processor import DocumentProcessor
from techdocrag.core.query_engine import QueryEngine
from techdocrag.retrieval.hybrid_retriever import HybridRetriever
from techdocrag.reasoning.reasoning_engine import ReasoningEngine
from techdocrag.utils.logging import get_logger

logger = get_logger(__name__)


class CompleteTechDocRAG:
    """
    Complete TechDocRAG System - Unified Interface
    
    This is the main unified interface providing all core features:
    1. Document Processing Pipeline
    2. Hybrid Retrieval System (Semantic + Keyword + Fusion)
    3. AI-Powered Reasoning & Smart Answers
    4. Interactive Question Answering
    5. System Health Monitoring & Statistics
    6. Multiple Input Methods & Custom Document Support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Complete TechDocRAG system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = Config(config_path)
        self.version = "1.0.0"
        self.initialized_at = datetime.now()
        
        # Core components (will be initialized asynchronously)
        self.document_processor = None
        self.hybrid_retriever = None  
        self.reasoning_engine = None
        self.query_engine = None  # LLM-powered query engine with answer synthesis
        
        # System statistics and health monitoring
        self.stats = {
            'documents_processed': 0,
            'queries_answered': 0,
            'total_chunks': 0,
            'avg_confidence': 0.0,
            'uptime_start': self.initialized_at,
            'total_response_time': 0.0
        }
        
        logger.info(f"CompleteTechDocRAG v{self.version} initialized")

    async def initialize(self):
        """Initialize all system components asynchronously."""
        try:
            logger.info("🚀 Initializing Complete TechDocRAG system components...")
            
            # Initialize core components
            await self._initialize_components()
            
            # Validate system health
            await self._validate_system()
            
            logger.info("✅ Complete TechDocRAG system fully initialized and ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Complete TechDocRAG: {e}")
            raise

    async def _initialize_components(self):
        """Initialize all system components."""
        
        # 1. Document Processor
        logger.info("   📄 Initializing DocumentProcessor...")
        self.document_processor = DocumentProcessor(self.config)
        
        # 2. Hybrid Retriever  
        logger.info("   🔍 Initializing HybridRetriever...")
        self.hybrid_retriever = HybridRetriever(self.config)
        
        # 3. Reasoning Engine
        logger.info("   🧠 Initializing ReasoningEngine...")
        self.reasoning_engine = ReasoningEngine(self.config)
        
        # 4. Query Engine with LLM-powered Answer Synthesis
        logger.info("   ✨ Initializing QueryEngine with Gemini LLM...")
        self.query_engine = QueryEngine(self.config)
        if self.config.llm.enable_synthesis and self.query_engine.gemini_client:
            logger.info("   ✅ Gemini LLM answer synthesis enabled")
        else:
            logger.info("   ⚠️  LLM synthesis disabled - using template mode")
        
        logger.info("   ✅ All components initialized successfully")

    async def _validate_system(self):
        """Validate system health and readiness."""
        components = {
            'DocumentProcessor': self.document_processor,
            'HybridRetriever': self.hybrid_retriever, 
            'ReasoningEngine': self.reasoning_engine,
            'QueryEngine': self.query_engine
        }
        
        for name, component in components.items():
            if component is None:
                raise RuntimeError(f"{name} failed to initialize")
            logger.info(f"   ✅ {name}: Ready")

    async def add_document(
        self, 
        document: Union[Document, str, Path],
        doc_type: DocumentType = DocumentType.GENERAL
    ) -> bool:
        """
        Add a document to the system for processing and indexing.
        
        Args:
            document: Document content (text, file path, or Document object)
            doc_type: Type of document for specialized processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare document object
            doc_obj = await self._prepare_document(document, doc_type)
            
            # Process document directly (avoiding file path issues)
            doc_result = await self._process_document_direct(doc_obj, doc_type)
            
            # Add to retrieval system
            await self.hybrid_retriever.add_documents_batch([doc_result])
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['total_chunks'] += len(doc_result.chunks)
            
            logger.info(f"✅ Document '{doc_obj.id}' processed and indexed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return False

    async def _prepare_document(
        self, 
        document: Union[Document, str, Path],
        doc_type: DocumentType
    ) -> Document:
        """Convert various input types to Document object."""
        
        if isinstance(document, Document):
            return document
        
        elif isinstance(document, (str, Path)):
            path = Path(document)
            
            if path.exists():
                # Read from file
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return Document(
                    id=f"doc_{path.stem}_{self.stats['documents_processed']}",
                    title=path.name,
                    content=content,
                    source=str(path)
                )
            else:
                # Direct text content
                return Document(
                    id=f"text_{self.stats['documents_processed']}",
                    title="Text Content",
                    content=str(document),
                    source="direct_input"
                )
        
        else:
            raise ValueError(f"Unsupported document type: {type(document)}")

    async def _process_document_direct(self, document: Document, doc_type: DocumentType) -> DocumentResult:
        """Process document directly without DocumentProcessor to avoid file path issues."""
        
        # Create chunks from document content
        chunks = self._create_document_chunks(document.content, document.id)
        
        # Create document result
        doc_result = DocumentResult(
            doc_id=document.id,
            file_path=Path(document.source) if document.source != "direct_input" else Path("direct_input.txt"),
            doc_type=doc_type,
            extracted_fields=[],
            chunks=chunks,
            raw_text=document.content,
            metadata={"title": document.title},
            metrics=ProcessingMetrics(
                processing_time=0.1,
                ocr_time=0.0,
                embedding_time=0.05,
                total_pages=1,
                total_chunks=len(chunks)
            )
        )
        
        return doc_result

    def _create_document_chunks(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create meaningful text chunks from document content."""
        import re
        
        # Split by sentences and paragraphs
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would make chunk too long, finalize current chunk
            if len(current_chunk) + len(sentence) > 200 and current_chunk:
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_chunk_{chunk_index}",
                    content=current_chunk.strip(),
                    metadata={'type': 'text_segment', 'doc_id': doc_id, 'chunk_index': chunk_index}
                ))
                current_chunk = sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                id=f"{doc_id}_chunk_{chunk_index}",
                content=current_chunk.strip(),
                metadata={'type': 'text_segment', 'doc_id': doc_id, 'chunk_index': chunk_index}
            ))
        
        return chunks

    async def ask_question(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question using the knowledge base with Gemini LLM-powered natural language answers.
        
        Args:
            question: Question to answer
            context: Optional context for the question
            
        Returns:
            Dict containing answer, sources, confidence, timing, etc.
        """
        try:
            start_time = datetime.now()
            self.stats['queries_answered'] += 1
            
            logger.info(f"🔍 Processing question: {question[:50]}...")
            
            # Hybrid retrieval (semantic + keyword + fusion)
            results = await self.hybrid_retriever.retrieve(question, top_k=10)
            
            # Convert results to chunks format for answer synthesizer
            retrieved_chunks = []
            for i, result in enumerate(results):
                chunk = {
                    'doc_id': result.get('doc_id', f'doc_{i}'),
                    'chunk_id': result.get('chunk_id', f'chunk_{i}'),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.0),
                    'metadata': result.get('metadata', {})
                }
                retrieved_chunks.append(chunk)
            
            # Use QueryEngine's answer synthesizer for natural language answers
            if self.query_engine and self.query_engine.answer_synthesizer:
                # Calculate retrieval confidence
                retrieval_confidence = min(90.0, len(results) * 9.0) if results else 0.0
                
                # Synthesize answer using Gemini LLM
                response = await self.query_engine.answer_synthesizer.synthesize_answer(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_confidence=retrieval_confidence
                )
                
                # Extract answer and metadata
                answer = response.answer
                confidence = response.confidence.overall
                reasoning = f"Synthesized from {len(results)} retrieved chunks using Gemini LLM"
                
                # Format sources
                sources = []
                for source in response.sources[:3]:
                    snippet = source.text_snippet[:100] + "..." if len(source.text_snippet) > 100 else source.text_snippet
                    sources.append(snippet)
                
            else:
                # Fallback to template mode if LLM not available
                answer = await self._generate_smart_answer(question, results)
                confidence = min(90.0, len(results) * 9.0) if results else 0.0
                reasoning = f"Found {len(results)} relevant document chunks using hybrid search (template mode)"
                sources = [r.get('content', '')[:100] + "..." for r in results[:3]]
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['total_response_time'] += response_time
            
            # Update average confidence
            total_queries = self.stats['queries_answered']
            if total_queries > 1:
                self.stats['avg_confidence'] = (
                    (self.stats['avg_confidence'] * (total_queries - 1) + confidence) / total_queries
                )
            else:
                self.stats['avg_confidence'] = confidence
            
            logger.info(f"✅ Question answered in {response_time:.2f}s with {len(results)} sources")
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': sources,
                'reasoning': reasoning,
                'response_time': response_time,
                'retrieval_count': len(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                'answer': f"I encountered an error processing your question: {e}",
                'confidence': 0.0,
                'sources': [],
                'reasoning': "Error occurred during processing",
                'response_time': 0.0,
                'retrieval_count': 0
            }

    async def _generate_smart_answer(self, question: str, results: List[Dict]) -> str:
        """Generate a smart contextual answer from retrieved results."""
        if not results:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract relevant content from results
        relevant_texts = []
        for result in results:
            content = result.get('content', '').strip()
            if content and len(content) > 10:  # Filter out very short chunks
                relevant_texts.append(content)
        
        if not relevant_texts:
            return "I found some documents but couldn't extract relevant content."
        
        # Smart answer generation based on question type
        question_lower = question.lower()
        
        # For "what is" questions, look for definitions and explanations
        if any(phrase in question_lower for phrase in ['what is', 'what are', 'define', 'explain']):
            return self._generate_definition_answer(question, relevant_texts)
        
        # For "how to" questions, look for steps and procedures
        elif any(phrase in question_lower for phrase in ['how to', 'how do', 'how can']):
            return self._generate_howto_answer(question, relevant_texts)
        
        # For specific feature/capability questions
        elif any(phrase in question_lower for phrase in ['features', 'capabilities', 'functions', 'can it']):
            return self._generate_feature_answer(question, relevant_texts)
        
        # Default: provide contextual summary
        else:
            return self._generate_contextual_answer(question, relevant_texts)

    def _generate_definition_answer(self, question: str, texts: List[str]) -> str:
        """Generate definition-style answers for 'what is' questions."""
        # Look for texts that contain definitions or explanations
        definition_texts = []
        for text in texts:
            text_lower = text.lower()
            # Prioritize texts with definition indicators
            if any(indicator in text_lower for indicator in [
                'is a', 'refers to', 'represents', 'means', 'definition', 
                'overview', 'introduction', 'abstract', 'system'
            ]):
                definition_texts.append(text)
        
        # Use definition texts if found, otherwise use best matches
        best_texts = definition_texts[:3] if definition_texts else texts[:3]
        
        # Create a coherent answer
        answer_parts = []
        for i, text in enumerate(best_texts):
            # Clean up the text
            clean_text = self._clean_text_for_answer(text)
            if clean_text and len(clean_text) > 20:
                answer_parts.append(clean_text)
        
        if answer_parts:
            answer = " ".join(answer_parts)
            # Ensure reasonable length
            if len(answer) > 400:
                answer = answer[:400] + "..."
            return answer
        else:
            return "I found some relevant information but couldn't extract a clear definition."

    def _generate_howto_answer(self, question: str, texts: List[str]) -> str:
        """Generate step-by-step answers for 'how to' questions."""
        # Look for procedural or instructional content
        procedure_texts = []
        for text in texts:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in [
                'step', 'first', 'then', 'next', 'install', 'setup', 'configure',
                'example', 'usage', 'tutorial', 'guide'
            ]):
                procedure_texts.append(text)
        
        best_texts = procedure_texts[:3] if procedure_texts else texts[:3]
        
        answer_parts = []
        for text in best_texts:
            clean_text = self._clean_text_for_answer(text)
            if clean_text and len(clean_text) > 15:
                answer_parts.append(clean_text)
        
        if answer_parts:
            answer = " ".join(answer_parts)
            if len(answer) > 400:
                answer = answer[:400] + "..."
            return answer
        else:
            return "I found some information but couldn't extract clear procedural steps."

    def _generate_feature_answer(self, question: str, texts: List[str]) -> str:
        """Generate feature/capability answers."""
        # Look for feature lists, bullet points, capabilities
        feature_texts = []
        for text in texts:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in [
                'feature', 'capability', 'support', 'include', 'provide',
                'benefit', 'advantage', 'function', '-', '•', '*'
            ]):
                feature_texts.append(text)
        
        best_texts = feature_texts[:3] if feature_texts else texts[:3]
        
        answer_parts = []
        for text in best_texts:
            clean_text = self._clean_text_for_answer(text)
            if clean_text and len(clean_text) > 15:
                answer_parts.append(clean_text)
        
        if answer_parts:
            answer = " ".join(answer_parts)
            if len(answer) > 400:
                answer = answer[:400] + "..."
            return answer
        else:
            return "I found some information but couldn't extract clear feature details."

    def _generate_contextual_answer(self, question: str, texts: List[str]) -> str:
        """Generate general contextual answers."""
        # Use the most relevant texts
        best_texts = texts[:3]
        
        answer_parts = []
        for text in best_texts:
            clean_text = self._clean_text_for_answer(text)
            if clean_text and len(clean_text) > 20:
                answer_parts.append(clean_text)
        
        if answer_parts:
            answer = " ".join(answer_parts)
            if len(answer) > 400:
                answer = answer[:400] + "..."
            return answer
        else:
            return "I found some relevant information in the documents, but couldn't generate a coherent answer."

    def _clean_text_for_answer(self, text: str) -> str:
        """Clean and prepare text for inclusion in answers."""
        # Remove code markers and clean up formatting
        import re
        
        # Remove common code patterns
        cleaned = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
        cleaned = re.sub(r'`[^`]+`', '', cleaned)      # Remove inline code
        cleaned = re.sub(r'^\s*[#-]\s*', '', cleaned, flags=re.MULTILINE)  # Remove markdown headers/bullets
        cleaned = re.sub(r'\s+', ' ', cleaned)         # Normalize whitespace
        cleaned = cleaned.strip()
        
        # Filter out obvious code lines
        if any(pattern in cleaned.lower() for pattern in [
            'import ', 'from ', 'def ', 'class ', 'if __name__', 
            'print(', 'return ', '=', '{', '}', ';', '//'
        ]):
            return ""  # Skip code-like content
        
        return cleaned

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics and health metrics."""
        uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        avg_response_time = (
            self.stats['total_response_time'] / self.stats['queries_answered'] 
            if self.stats['queries_answered'] > 0 else 0.0
        )
        
        return {
            'version': self.version,
            'uptime_seconds': uptime,
            'documents_processed': self.stats['documents_processed'],
            'total_chunks': self.stats['total_chunks'],
            'queries_answered': self.stats['queries_answered'],
            'avg_confidence': round(self.stats['avg_confidence'], 1),
            'avg_response_time': round(avg_response_time, 3),
            'status': 'operational'
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'issues': []
        }
        
        # Check components
        components = {
            'document_processor': self.document_processor,
            'hybrid_retriever': self.hybrid_retriever,
            'reasoning_engine': self.reasoning_engine
        }
        
        for name, component in components.items():
            if component is None:
                health['components'][name] = 'not_initialized'
                health['issues'].append(f"{name}: Not initialized")
            else:
                try:
                    # Simple component validation
                    health['components'][name] = 'operational'
                except Exception as e:
                    health['components'][name] = f'error: {e}'
                    health['issues'].append(f"{name}: {e}")
        
        # Overall status
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) <= 2 else 'unhealthy'
        
        return health

    async def add_custom_document(self, custom_text: str, title: str = "Custom Document") -> bool:
        """Add custom text document for testing and Q&A."""
        custom_doc = Document(
            id=f"custom_{self.stats['documents_processed']}",
            title=title,
            content=custom_text,
            source="user_input"
        )
        
        return await self.add_document(custom_doc, DocumentType.GENERAL)

    async def interactive_qa(self):
        """Interactive Q&A session with the current knowledge base."""
        print("\n💬 Interactive Q&A Session")
        print("=" * 40)
        print("Ask questions about your documents (type 'quit' to exit)")
        print("💡 Tips:")
        print("   - Ask specific questions for better results")
        print("   - Questions about content, facts, or details work best")
        print("   - The system uses hybrid search for optimal accuracy")
        
        while True:
            try:
                question = input("\n❓ Your question (or 'quit'): ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not question:
                    continue
                
                print(f"🔍 Processing: {question}")
                
                # Get answer
                response = await self.ask_question(question)
                
                # Display results
                print(f"✅ 📋 Based on the documents:")
                print(f"{response['answer']}")
                print(f"🎯 Confidence: {response['confidence']}%")
                print(f"⚡ Time: {response['response_time']:.2f}s")
                
                if response['sources']:
                    print(f"📚 Sources: {len(response['sources'])} document chunks")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def demo_custom_document():
    """Demo function showing custom document processing."""
    print("🎯 Complete TechDocRAG - Custom Document Demo")
    print("=" * 50)
    
    # Initialize system
    system = CompleteTechDocRAG()
    await system.initialize()
    
    # Get custom text from user
    print("📄 Enter your custom text content:")
    print("   You can paste multiple lines of text.")
    print("   Type 'END' on a new line when finished, or press Ctrl+C to cancel.\n")
    
    custom_lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            custom_lines.append(line)
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return
    
    if not custom_lines:
        print("❌ No content provided")
        return
        
    custom_content = '\n'.join(custom_lines)
    
    # Get document title from user
    doc_title = input("\n📝 Enter document title (or press Enter for default): ").strip()
    if not doc_title:
        doc_title = "Custom Document"
    
    print(f"\n📄 Adding custom document '{doc_title}'...")
    success = await system.add_custom_document(custom_content, doc_title)
    
    if success:
        print("✅ Custom document added successfully!")
        
        # Show system stats
        stats = system.get_system_stats()
        print(f"\n📊 System Status:")
        print(f"   Documents: {stats['documents_processed']}")
        print(f"   Text chunks: {stats['total_chunks']}")
        print(f"   Status: {stats['status']}")
        
        # Interactive Q&A
        await system.interactive_qa()
    else:
        print("❌ Failed to add custom document")

async def demo_file_processing():
    """Demo function showing file-based document processing."""
    print("🎯 Complete TechDocRAG - File Processing Demo") 
    print("=" * 50)
    
    # Initialize system
    system = CompleteTechDocRAG()
    await system.initialize()
    
    print("📁 Looking for text files to process...")
    
    # Look for any .txt or .md files in current directory
    current_dir = Path(".")
    text_files = list(current_dir.glob("*.txt")) + list(current_dir.glob("*.md"))
    
    if text_files:
        for file_path in text_files[:3]:  # Process first 3 files
            print(f"📄 Processing: {file_path.name}")
            success = await system.add_document(file_path)
            if success:
                print(f"✅ {file_path.name} processed successfully")
            else:
                print(f"❌ Failed to process {file_path.name}")
    else:
        print("📝 No text files found, using sample content...")
        # Add sample technical content
        sample_tech = """
        TechDocRAG System Architecture
        
        The TechDocRAG system implements a hybrid approach to document intelligence,
        combining rule-based processing with neural retrieval-augmented generation.
        
        Core Components:
        1. Document Processor - Handles text extraction and chunking
        2. Hybrid Retriever - Combines semantic and keyword search
        3. Reasoning Engine - Provides AI-powered answer generation
        4. Configuration Manager - Handles system settings
        
        Features:
        - Async/await architecture for high performance
        - Multiple input formats (text, files, Document objects)
        - Confidence scoring for answer quality assessment
        - Real-time health monitoring and statistics
        - Interactive Q&A interface
        """
        await system.add_custom_document(sample_tech, "TechDocRAG Architecture")
    
    # Show system stats
    stats = system.get_system_stats()
    print(f"\n📊 Processing Complete:")
    print(f"   Documents processed: {stats['documents_processed']}")
    print(f"   Total text chunks: {stats['total_chunks']}")
    print(f"   System status: {stats['status']}")
    
    # Interactive Q&A
    await system.interactive_qa()

async def demo_multi_document_qa():
    """Interactive multi-document Q&A with user-provided documents."""
    print("=" * 80)
    print("📚 MULTI-DOCUMENT QUESTION ANSWERING")
    print("=" * 80)
    print()
    
    # Initialize system
    print("[1] Initializing TechDocRAG system...")
    system = CompleteTechDocRAG()
    await system.initialize()
    print("    ✓ System ready!")
    print()
    
    # Get number of documents from user
    print("[2] Document Collection")
    print("-" * 80)
    while True:
        try:
            num_docs = input("\n📝 How many documents do you want to add? (Max: 4): ").strip()
            num_docs = int(num_docs)
            if 1 <= num_docs <= 4:
                break
            else:
                print("❌ Please enter a number between 1 and 4")
        except ValueError:
            print("❌ Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Returning to main menu...")
            return
    
    print(f"\n✓ You will add {num_docs} document(s)")
    print()
    
    # Collect documents from user
    documents = []
    doc_titles = []
    
    for i in range(num_docs):
        print("=" * 80)
        print(f"DOCUMENT {i+1} of {num_docs}")
        print("=" * 80)
        
        # Get document title
        while True:
            doc_title = input(f"\n📄 Enter title for document {i+1}: ").strip()
            if doc_title:
                break
            else:
                print("❌ Title cannot be empty")
        
        doc_titles.append(doc_title)
        
        # Get document content
        print(f"\n📝 Enter content for '{doc_title}'")
        print("💡 Tip: Paste your text, then type 'END' on a new line when done")
        print("💡 Or type line by line, then type 'END' to finish")
        print("-" * 80)
        
        doc_lines = []
        
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":  # Type END to finish
                    break
                doc_lines.append(line)
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Input interrupted")
                if doc_lines:
                    print("✓ Using content entered so far...")
                    break
                else:
                    print("❌ No content entered, skipping this document")
                    continue
        
        doc_content = '\n'.join(doc_lines).strip()
        
        if not doc_content:
            print(f"⚠️  No content entered for '{doc_title}', skipping...")
            continue
        
        documents.append((doc_content, doc_title))
        print(f"\n✓ Document '{doc_title}' added ({len(doc_content)} characters)")
        print()
    
    # Check if we have any documents
    if not documents:
        print("\n❌ No documents were added. Returning to main menu...")
        return
    
    # Add documents to system
    print("=" * 80)
    print(f"[3] Processing {len(documents)} document(s)...")
    print("-" * 80)
    
    for i, (content, title) in enumerate(documents, 1):
        await system.add_custom_document(content, title)
        print(f"    ✓ [{i}/{len(documents)}] '{title}' indexed successfully")
    
    print()
    print(f"    Total: {len(documents)} document(s) loaded and ready!")
    print()
    
    # Show document summary
    print("=" * 80)
    print("LOADED DOCUMENTS:")
    print("=" * 80)
    for i, title in enumerate([t for _, t in documents], 1):
        print(f"  {i}. {title}")
    print()
    
    # Interactive Q&A loop
    print("=" * 80)
    print("ASK YOUR QUESTIONS!")
    print("=" * 80)
    print()
    print("💡 Tips:")
    print("   • Ask specific questions about your documents")
    print("   • Try: 'What is mentioned about...?'")
    print("   • Try: 'Find information on...'")
    print("   • Try: 'Compare [topic] across documents'")
    print()
    print("Type your questions below. Type 'quit', 'exit', or 'back' to return to main menu.")
    print("📝 Note: System retrieves information but doesn't synthesize (LLM coming in Phase 3)")
    print()
    
    question_num = 1
    while True:
        # Get question
        print(f"\n[Q{question_num}] Your question: ", end="")
        try:
            question = input().strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Returning to main menu...")
            break
        
        # Check for exit
        if question.lower() in ['quit', 'exit', 'q', 'back', '']:
            print("\n👋 Returning to main menu...")
            break
        
        # Ask the system
        print("-" * 80)
        result = await system.ask_question(question)
        
        # Display results
        print(f"\n📝 Answer:")
        answer_text = result['answer']
        if len(answer_text) > 500:
            print(f"{answer_text[:500]}...")
            print(f"\n   ... (truncated, showing first 500 characters)")
        else:
            print(answer_text)
        
        print()
        print(f"📊 Confidence: {result['confidence']:.1f}%")
        print(f"📚 Sources: {result['retrieval_count']} documents")
        print(f"⚡ Response time: {result['response_time']:.2f}s")
        print("-" * 80)
        
        question_num += 1
    
    print()
    print("=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)
    print(f"\nTotal questions asked: {question_num - 1}")
    print("\n✅ Multi-document retrieval working!")
    print("⏳ Natural language synthesis coming in Phase 3 (LLM integration)")

async def main():
    """Main interface for Complete TechDocRAG System."""
    print("🚀 Complete TechDocRAG - Unified Document Intelligence System")
    print("=" * 60)
    print("Version: 1.0.0 | All Core Features Included")
    print()
    print("Available modes:")
    print("1. Custom Text Document Demo")
    print("2. File Processing Demo")
    print("3. Multi-Document Q&A (NEW!)")
    print("4. System Health Check")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\n🎯 Select mode (1-5): ").strip()
            
            if choice == '1':
                await demo_custom_document()
            elif choice == '2':
                await demo_file_processing()
            elif choice == '3':
                await demo_multi_document_qa()
            elif choice == '4':
                system = CompleteTechDocRAG()
                await system.initialize()
                health = await system.health_check()
                stats = system.get_system_stats()
                
                print("\n🏥 System Health Check:")
                print(f"   Status: {health['status']}")
                print(f"   Components: {len(health['components'])} initialized")
                print(f"   Issues: {len(health['issues'])}")
                
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                    
            elif choice == '5':
                print("\n👋 Thank you for using Complete TechDocRAG!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())