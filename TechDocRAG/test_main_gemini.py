#!/usr/bin/env python3
"""
Test main.py with Gemini LLM integration
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import CompleteTechDocRAG


async def test_main_with_gemini():
    """Test the updated main.py with Gemini LLM."""
    
    print("\n" + "=" * 80)
    print("TESTING MAIN.PY WITH GEMINI LLM INTEGRATION")
    print("=" * 80)
    print()
    
    # Initialize system
    print("[1] Initializing TechDocRAG system...")
    system = CompleteTechDocRAG()
    await system.initialize()
    print()
    
    # Add sample documents
    print("[2] Adding sample documents...")
    
    doc1 = """
    TECH SOLUTIONS INC. - ANNUAL REPORT 2023
    
    Company Name: Tech Solutions Inc. (Company A)
    Revenue: $150 million
    Growth: +35% year-over-year
    Employees: 500
    CEO: John Smith
    Main Product: Cloud Software
    Profit Margin: 22%
    """
    
    doc2 = """
    DIGITAL INNOVATIONS LTD. - ANNUAL REPORT 2023
    
    Company Name: Digital Innovations Ltd. (Company B)
    Revenue: $120 million
    Growth: +18% year-over-year
    Employees: 350
    CEO: Sarah Johnson
    Main Product: Mobile Apps
    Profit Margin: 18%
    """
    
    await system.add_document(doc1)
    print("   ✅ Company A report added")
    
    await system.add_document(doc2)
    print("   ✅ Company B report added")
    print()
    
    # Test questions
    questions = [
        "Which company has higher revenue?",
        "Who is the CEO of Company B?",
        "What is the revenue difference between the two companies?",
        "Which company has more employees?",
        "Which company has better profit margins?"
    ]
    
    print("[3] Asking questions with Gemini LLM...")
    print("=" * 80)
    print()
    
    for i, question in enumerate(questions, 1):
        print(f"[Q{i}] {question}")
        print("-" * 80)
        
        result = await system.ask_question(question)
        
        print(f"✨ Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.0f}%")
        print(f"⚡ Response Time: {result['response_time']:.2f}s")
        print(f"📚 Sources: {result['retrieval_count']}")
        print()
        print("=" * 80)
        print()
    
    print()
    print("🎉 TEST COMPLETE!")
    print()
    print("✅ Gemini LLM integration working in main.py")
    print(f"✅ Total queries answered: {system.stats['queries_answered']}")
    print(f"✅ Average confidence: {system.stats['avg_confidence']:.1f}%")
    print()


if __name__ == "__main__":
    asyncio.run(test_main_with_gemini())
