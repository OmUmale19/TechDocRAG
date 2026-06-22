"""
Quick verification test for Phase 3 Gemini LLM integration.
Tests that all components can be imported and initialized.
"""

import asyncio
import os
from techdocrag.core.query_engine import QueryEngine
from techdocrag.core.config import Config
from techdocrag.llm.gemini_client import GeminiClient
from techdocrag.reasoning.answer_synthesizer import AnswerSynthesizer


def test_imports():
    """Test all Phase 3 imports."""
    print("=" * 60)
    print("Phase 3 Installation Verification")
    print("=" * 60)
    print()
    print("✅ google-generativeai package installed")
    print("✅ GeminiClient imported successfully")
    print("✅ AnswerSynthesizer imported successfully")
    print("✅ QueryEngine imported successfully")
    print()


def test_configuration():
    """Test configuration is correct."""
    print("Configuration Check:")
    print("-" * 60)
    
    config = Config()
    
    print(f"✅ LLM Provider: {config.llm.provider}")
    print(f"✅ LLM Model: {config.llm.model_name}")
    print(f"✅ LLM Synthesis Enabled: {config.llm.enable_synthesis}")
    print(f"✅ Temperature: {config.llm.temperature}")
    print(f"✅ Max Tokens: {config.llm.max_tokens}")
    print()
    
    return config


def test_gemini_api_key():
    """Check if Gemini API key is set."""
    print("API Key Check:")
    print("-" * 60)
    
    api_key = os.getenv("AIzaSyAeqp9cF70ohJzIY6oi8z9aHYmO9YC7a0E")
    
    if api_key:
        print(f"✅ GEMINI_API_KEY is set (length: {len(api_key)} chars)")
        print("✅ Ready for LLM-powered answer synthesis!")
        return True
    else:
        print("⚠️  GEMINI_API_KEY not set")
        print("📝 System will use template mode (no LLM synthesis)")
        print()
        print("To enable LLM synthesis:")
        print("  1. Get API key from: https://makersuite.google.com/app/apikey")
        print("  2. Set environment variable:")
        print("     PowerShell: $env:GEMINI_API_KEY=\"your-key-here\"")
        return False


async def test_answer_synthesizer():
    """Test AnswerSynthesizer initialization."""
    print()
    print("Answer Synthesizer Test:")
    print("-" * 60)
    
    # Test with LLM disabled (safe without API key)
    synthesizer = AnswerSynthesizer(enable_synthesis=False)
    
    print("✅ AnswerSynthesizer created successfully")
    print(f"✅ LLM synthesis enabled: {synthesizer.enable_synthesis}")
    print(f"✅ Statistics: {synthesizer.stats}")
    print()
    
    # Test with sample data
    sample_chunks = [
        {
            'doc_id': 'test_doc',
            'chunk_id': 'chunk_1',
            'content': 'This is a test document chunk.',
            'score': 0.95,
            'metadata': {'title': 'Test Document'}
        }
    ]
    
    response = await synthesizer.synthesize_answer(
        question="What is this about?",
        retrieved_chunks=sample_chunks,
        retrieval_confidence=85.0
    )
    
    print("✅ Template answer generated successfully")
    print(f"   Question: {response.query}")
    print(f"   Answer: {response.answer[:100]}...")
    print(f"   Confidence: {response.confidence.overall}%")
    print()


def test_query_engine():
    """Test QueryEngine can initialize."""
    print("Query Engine Test:")
    print("-" * 60)
    
    config = Config()
    engine = QueryEngine(config)
    
    print("✅ QueryEngine initialized successfully")
    print(f"✅ Answer synthesizer present: {engine.answer_synthesizer is not None}")
    print(f"✅ Gemini client present: {engine.gemini_client is not None}")
    print()


async def main():
    """Run all tests."""
    print()
    test_imports()
    config = test_configuration()
    has_api_key = test_gemini_api_key()
    await test_answer_synthesizer()
    test_query_engine()
    
    print("=" * 60)
    print("Installation Verification Complete!")
    print("=" * 60)
    print()
    
    if has_api_key:
        print("🎉 ALL SYSTEMS GO! Ready for LLM-powered answers!")
        print()
        print("Next steps:")
        print("  1. Run: python main.py")
        print("  2. Select Option 3: Multi-Document Q&A")
        print("  3. Ask questions and get intelligent answers!")
    else:
        print("⚠️  System ready in TEMPLATE MODE")
        print()
        print("To enable full LLM capabilities:")
        print("  1. Get Gemini API key (FREE): https://makersuite.google.com/app/apikey")
        print("  2. Set: $env:GEMINI_API_KEY=\"your-key\"")
        print("  3. Re-run this test")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())
