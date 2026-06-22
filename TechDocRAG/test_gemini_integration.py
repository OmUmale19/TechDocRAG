"""
Test script for Gemini LLM Integration (Phase 3)
Tests natural language answer generation with citations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from techdocrag.core.config import Config
from techdocrag.core.types import Document, DocumentChunk
from techdocrag.llm.gemini_client import GeminiClient
from techdocrag.reasoning.answer_synthesizer import AnswerSynthesizer
from techdocrag.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_gemini_connection():
    """Test 1: Verify Gemini API connection."""
    print("\n" + "="*80)
    print("TEST 1: Gemini API Connection")
    print("="*80)
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set!")
            print("\n💡 To get your API key:")
            print("   1. Visit https://makersuite.google.com/app/apikey")
            print("   2. Create an API key")
            print("   3. Set it: $env:GEMINI_API_KEY='your-api-key-here'")
            return False
        
        client = GeminiClient(api_key=api_key)
        connected = await client.test_connection()
        
        if connected:
            print("✅ Gemini API connection successful!")
            print(f"   Model: {client.model_name}")
            return True
        else:
            print("❌ Gemini API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False


async def test_answer_synthesis():
    """Test 2: Test answer synthesis with sample data."""
    print("\n" + "="*80)
    print("TEST 2: Answer Synthesis")
    print("="*80)
    
    try:
        # Sample document chunks (from our multi-doc demo)
        sample_chunks = [
            {
                'doc_id': 'company_a',
                'chunk_id': 'chunk_1',
                'content': """COMPANY A - ANNUAL REPORT 2023
Company Name: Tech Solutions Inc
Main Product: Cloud Software
Revenue: $150M
Growth Rate: +35%
Employees: 500
Market: North America
CEO: John Smith
Founded: 2015""",
                'metadata': {'title': 'Company A Annual Report'},
                'score': 0.95
            },
            {
                'doc_id': 'company_b',
                'chunk_id': 'chunk_2',
                'content': """COMPANY B - ANNUAL REPORT 2023
Company Name: Digital Innovations Ltd
Main Product: Mobile Apps
Revenue: $120M
Growth Rate: +18%
Employees: 350
Market: Europe
CEO: Sarah Johnson
Founded: 2018""",
                'metadata': {'title': 'Company B Annual Report'},
                'score': 0.88
            }
        ]
        
        # Test questions
        questions = [
            "Which company has higher revenue?",
            "What is the revenue of Company A?",
            "Who is the CEO of Company B?",
            "Which company makes mobile applications?"
        ]
        
        # Initialize synthesizer
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  No API key - testing template mode only")
            synthesizer = AnswerSynthesizer(enable_synthesis=False)
        else:
            client = GeminiClient(api_key=api_key)
            synthesizer = AnswerSynthesizer(gemini_client=client, enable_synthesis=True)
        
        print(f"\n📊 Synthesizer mode: {'LLM' if synthesizer.enable_synthesis else 'Template'}\n")
        
        # Test each question
        for i, question in enumerate(questions, 1):
            print(f"\n[Q{i}] {question}")
            print("-" * 80)
            
            response = await synthesizer.synthesize_answer(
                question=question,
                retrieved_chunks=sample_chunks,
                retrieval_confidence=85.0
            )
            
            print(f"\n📝 ANSWER:")
            print(f"   {response.answer}\n")
            
            print(f"📊 CONFIDENCE: {response.confidence.overall}%")
            print(f"   - Retrieval: {response.confidence.retrieval}%")
            print(f"   - Reasoning: {response.confidence.reasoning}%")
            print(f"   - Source Quality: {response.confidence.source_quality}%")
            
            print(f"\n📚 SOURCES ({len(response.sources)} total):")
            for src in response.sources[:3]:  # Show top 3
                cited = "✓" if src.cited_in_answer else " "
                print(f"   [{cited}] {src.title} (relevance: {src.relevance_score:.2f})")
            
            if response.metadata.get('llm_reasoning'):
                print(f"\n💭 REASONING:")
                print(f"   {response.metadata['llm_reasoning']}")
            
            print()
        
        print("="*80)
        print("✅ Answer synthesis test completed!")
        
        # Show stats
        stats = synthesizer.get_stats()
        print(f"\n📈 Statistics:")
        print(f"   Total syntheses: {stats['total_syntheses']}")
        print(f"   LLM syntheses: {stats['llm_syntheses']}")
        print(f"   Template answers: {stats['template_answers']}")
        print(f"   Avg synthesis time: {stats['avg_synthesis_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Answer synthesis test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_integration():
    """Test 3: Test full integration with query engine."""
    print("\n" + "="*80)
    print("TEST 3: Full Query Engine Integration")
    print("="*80)
    
    try:
        from techdocrag.core.query_engine import QueryEngine
        
        # Set API key if available
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  No API key - query engine will use template mode")
        
        # Initialize query engine
        config = Config()
        config.llm.enable_synthesis = bool(api_key)
        config.llm.api_key = api_key
        
        engine = QueryEngine(config)
        await engine.initialize()
        
        # Add sample documents
        doc_a = Document(
            id="company_a",
            title="Company A Annual Report",
            content="""COMPANY A - ANNUAL REPORT 2023
Company Name: Tech Solutions Inc
Main Product: Cloud Software
Revenue: $150M
Growth Rate: +35%
Employees: 500
Market: North America
CEO: John Smith
Founded: 2015""",
            source="company_a_report.pdf"
        )
        
        doc_b = Document(
            id="company_b",
            title="Company B Annual Report",
            content="""COMPANY B - ANNUAL REPORT 2023
Company Name: Digital Innovations Ltd
Main Product: Mobile Apps
Revenue: $120M
Growth Rate: +18%
Employees: 350
Market: Europe
CEO: Sarah Johnson
Founded: 2018""",
            source="company_b_report.pdf"
        )
        
        await engine.add_custom_document(doc_a)
        await engine.add_custom_document(doc_b)
        
        print("✅ Documents added to query engine")
        
        # Test query
        question = "Which company has higher revenue and by how much?"
        print(f"\n❓ Question: {question}")
        print("-" * 80)
        
        response = await engine.ask_question(question)
        
        print(f"\n📝 ANSWER:")
        print(f"   {response['answer']}\n")
        
        print(f"📊 Confidence: {response['confidence']}%")
        print(f"📚 Sources: {response['sources_count']}")
        print(f"⚡ Response time: {response['response_time']:.3f}s")
        
        print("\n✅ Full integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   PHASE 3: GEMINI LLM INTEGRATION TEST                     ║
║                    Natural Language Answer Generation                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = []
    
    # Test 1: API Connection
    results.append(("Gemini API Connection", await test_gemini_connection()))
    
    # Test 2: Answer Synthesis
    results.append(("Answer Synthesis", await test_answer_synthesis()))
    
    # Test 3: Full Integration
    # results.append(("Full Integration", await test_full_integration()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 3 LLM integration is working! 🎉")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
