"""
Quick test to verify Gemini API connection.
"""

import os
import asyncio
from techdocrag.llm.gemini_client import GeminiClient


async def test_gemini_connection():
    """Test Gemini API connection with actual API call."""
    
    print("=" * 60)
    print("Gemini API Connection Test")
    print("=" * 60)
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print()
        print("Please set it:")
        print('  $env:GEMINI_API_KEY="your-api-key-here"')
        return False
    
    print(f"✅ API Key found (length: {len(api_key)} chars)")
    print()
    
    try:
        # Initialize Gemini client
        print("Initializing Gemini client...")
        client = GeminiClient(
            api_key=api_key,
            model_name="gemini-2.0-flash",
            temperature=0.1
        )
        print("✅ GeminiClient initialized")
        print()
        
        # Test connection
        print("Testing API connection...")
        success = await client.test_connection()
        
        if success:
            print("✅ API connection successful!")
            print()
            
            # Test answer generation
            print("Testing answer generation...")
            print("-" * 60)
            
            test_question = "What is 150 minus 120?"
            test_context = [
                "Company A has revenue of $150M.",
                "Company B has revenue of $120M."
            ]
            test_sources = [
                {'doc_id': 'doc1', 'title': 'Company A Report'},
                {'doc_id': 'doc2', 'title': 'Company B Report'}
            ]
            
            result = await client.generate_answer(
                question=test_question,
                context=test_context,
                sources=test_sources
            )
            
            print(f"Question: {test_question}")
            print()
            print(f"✨ ANSWER: {result['answer']}")
            print()
            print(f"📊 Confidence: {result['confidence']}%")
            print(f"💭 Reasoning: {result['reasoning']}")
            print(f"📚 Citations: {result.get('citations', [])}")
            print()
            print("=" * 60)
            print("🎉 GEMINI LLM INTEGRATION FULLY WORKING!")
            print("=" * 60)
            
            return True
        else:
            print("❌ API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print()
        print("Possible issues:")
        print("  - API key is invalid")
        print("  - Network connection issues")
        print("  - API quota exceeded")
        return False


if __name__ == "__main__":
    asyncio.run(test_gemini_connection())
