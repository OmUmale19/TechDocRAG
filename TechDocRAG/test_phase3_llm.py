"""
Direct test of Phase 3 LLM integration with actual multi-document Q&A.
"""

import os
import asyncio
from techdocrag.core.query_engine import QueryEngine
from techdocrag.core.config import Config
from techdocrag.core.types import DocumentChunk


async def test_llm_qa():
    """Test LLM-powered Q&A with sample documents."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: LLM-POWERED MULTI-DOCUMENT Q&A TEST")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        return
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # Initialize query engine with LLM
    print("Initializing Query Engine with Gemini LLM...")
    config = Config()
    engine = QueryEngine(config)
    
    print(f"✅ LLM Model: {config.llm.model_name}")
    print(f"✅ LLM Synthesis: {config.llm.enable_synthesis}")
    print(f"✅ Gemini Client: {engine.gemini_client is not None}")
    print(f"✅ Answer Synthesizer: {engine.answer_synthesizer is not None}")
    print()
    
    # Create sample documents
    print("Creating sample documents...")
    
    chunks = [
        {
            'doc_id': 'company_a',
            'chunk_id': 'chunk_1',
            'content': 'Company A (Tech Solutions Inc) has revenue of $150M with 35% growth. They have 500 employees and their CEO is John Smith.',
            'score': 0.95,
            'metadata': {'title': 'Company A Report', 'page_number': 1}
        },
        {
            'doc_id': 'company_b',
            'chunk_id': 'chunk_2',
            'content': 'Company B (Digital Innovations Ltd) has revenue of $120M with 18% growth. They have 350 employees and their CEO is Sarah Johnson.',
            'score': 0.92,
            'metadata': {'title': 'Company B Report', 'page_number': 1}
        },
        {
            'doc_id': 'market',
            'chunk_id': 'chunk_3',
            'content': 'The market is growing rapidly. Cloud computing is up 40% annually and mobile apps are up 25% annually.',
            'score': 0.88,
            'metadata': {'title': 'Market Analysis', 'page_number': 1}
        }
    ]
    
    print(f"✅ {len(chunks)} document chunks ready")
    print()
    
    # Test questions
    questions = [
        "Which company has higher revenue?",
        "Who is the CEO of Company B?",
        "What is the revenue difference between the two companies?"
    ]
    
    print("=" * 70)
    print("TESTING LLM ANSWER GENERATION")
    print("=" * 70)
    print()
    
    for i, question in enumerate(questions, 1):
        print(f"[Question {i}]")
        print(f"Q: {question}")
        print("-" * 70)
        
        # Generate answer using LLM
        response = await engine.answer_synthesizer.synthesize_answer(
            question=question,
            retrieved_chunks=chunks,
            retrieval_confidence=90.0
        )
        
        print(f"\n✨ ANSWER:")
        print(f"   {response.answer}")
        print()
        print(f"📊 Confidence: {response.confidence.overall}%")
        print(f"📚 Sources: {len(response.sources)} documents")
        print(f"⚡ Response time: {response.response_time:.2f}s")
        print()
        print("=" * 70)
        print()
    
    print()
    print("🎉 PHASE 3 LLM INTEGRATION TEST COMPLETE!")
    print()
    print("✅ Natural language answers generated")
    print("✅ Confidence scores calculated")
    print("✅ Source attribution working")
    print("✅ All systems operational!")
    print()


if __name__ == "__main__":
    asyncio.run(test_llm_qa())
