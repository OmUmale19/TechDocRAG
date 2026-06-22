#!/usr/bin/env python3
"""
Interactive Gemini LLM Q&A Session
==================================
Ask your own questions to test the Gemini-powered multi-document Q&A system.
"""

import os
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from techdocrag.core.query_engine import QueryEngine
from techdocrag.core.config import Config


async def main():
    """Run interactive Q&A session."""
    
    print("\n" + "=" * 80)
    print("INTERACTIVE GEMINI LLM Q&A SESSION")
    print("=" * 80)
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY environment variable not set!")
        print("\nPlease set it using:")
        print('   $env:GEMINI_API_KEY="your-api-key-here"')
        return
    
    print("✅ Gemini API Key detected")
    print()
    
    # Initialize query engine
    print("🚀 Initializing Gemini-powered Query Engine...")
    config = Config()
    query_engine = QueryEngine(config)
    print(f"✅ Model: {config.llm.model_name}")
    print()
    
    # Prepare sample documents
    print("📚 Loading sample company reports...")
    
    document_chunks = [
        {
            'doc_id': 'company_a',
            'chunk_id': 'company_a_chunk_0',
            'content': '''
            TECH SOLUTIONS INC. - ANNUAL REPORT 2023
            
            Company Overview:
            - Company Name: Tech Solutions Inc. (Company A)
            - Industry: Cloud Software & SaaS
            - Founded: 2015
            - Headquarters: San Francisco, CA
            
            Financial Performance:
            - Annual Revenue: $150 million
            - Year-over-Year Growth: +35%
            - Profit Margin: 22%
            - R&D Investment: $25 million
            
            Operations:
            - Total Employees: 500
            - Engineering Team: 200
            - Sales & Marketing: 150
            - Office Locations: 5 (USA, Canada, UK)
            
            Leadership:
            - CEO: John Smith
            - CTO: Emily Chen
            - CFO: Michael Brown
            
            Key Products:
            - CloudPlatform Pro (Main Product)
            - AI Analytics Suite
            - Developer Tools
            
            2023 Achievements:
            - Launched new AI-powered features
            - Expanded to 3 new international markets
            - Won "Best Cloud Platform 2023" award
            - Reached 10,000+ enterprise customers
            
            Market Position:
            - Market Share: 15% in cloud software
            - Customer Satisfaction: 4.8/5.0
            - Net Promoter Score: 72
            ''',
            'score': 1.0,
            'metadata': {'title': 'Tech Solutions Inc. Annual Report', 'year': 2023}
        },
        {
            'doc_id': 'company_b',
            'chunk_id': 'company_b_chunk_0',
            'content': '''
            DIGITAL INNOVATIONS LTD. - ANNUAL REPORT 2023
            
            Company Overview:
            - Company Name: Digital Innovations Ltd. (Company B)
            - Industry: Mobile Applications
            - Founded: 2018
            - Headquarters: London, UK
            
            Financial Performance:
            - Annual Revenue: $120 million
            - Year-over-Year Growth: +18%
            - Profit Margin: 18%
            - R&D Investment: $15 million
            
            Operations:
            - Total Employees: 350
            - Engineering Team: 140
            - Sales & Marketing: 100
            - Office Locations: 3 (UK, Germany, France)
            
            Leadership:
            - CEO: Sarah Johnson
            - CTO: David Lee
            - CFO: Rachel Green
            
            Key Products:
            - MobileApp Builder
            - Social Connect Platform
            - E-commerce Solutions
            
            2023 Achievements:
            - Reached 10 million active users
            - Acquired competitor "AppMaster" for $20M
            - Opened new office in Berlin
            - Launched AI chatbot integration
            
            Market Position:
            - Market Share: 12% in mobile apps
            - Customer Satisfaction: 4.6/5.0
            - App Store Rating: 4.7/5.0
            ''',
            'score': 1.0,
            'metadata': {'title': 'Digital Innovations Ltd. Annual Report', 'year': 2023}
        },
        {
            'doc_id': 'market_analysis',
            'chunk_id': 'market_analysis_chunk_0',
            'content': '''
            TECH INDUSTRY MARKET ANALYSIS 2023
            
            Market Trends:
            - Cloud computing market growing at 40% annually
            - Mobile app market growing at 25% annually
            - AI integration becoming standard across all platforms
            - Remote work driving increased software demand
            - Cybersecurity spending up 30%
            
            Top Performing Sectors:
            1. Cloud & SaaS Solutions - Highest growth sector
            2. AI & Machine Learning Tools - Emerging market
            3. Mobile Applications - Mature but stable
            4. Cybersecurity - Critical infrastructure
            
            Key Success Factors:
            - Companies with AI features showing 2x higher growth
            - Cloud-based solutions outperforming traditional software
            - International expansion correlating with revenue growth
            - High R&D investment (>15% of revenue) driving innovation
            
            Predictions for 2024:
            - Continued strong growth in cloud sector (35-45%)
            - Mobile apps facing increased competition
            - AI becoming mandatory, not optional
            - Consolidation through M&A activity
            ''',
            'score': 1.0,
            'metadata': {'title': 'Tech Industry Market Analysis', 'year': 2023}
        }
    ]
    
    print("✅ 3 documents loaded (Company A, Company B, Market Analysis)")
    print()
    
    print("=" * 80)
    print("READY TO ANSWER YOUR QUESTIONS!")
    print("=" * 80)
    print()
    print("Example questions you can ask:")
    print("  • Which company has higher revenue?")
    print("  • Who is the CEO of Company A?")
    print("  • What is the revenue difference between the companies?")
    print("  • Which company has better growth rate?")
    print("  • What market trends are mentioned?")
    print()
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 80)
    print()
    
    # Interactive Q&A loop
    question_count = 0
    while True:
        # Get user question
        question = input("❓ Your Question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for testing! Goodbye!")
            break
        
        question_count += 1
        print()
        print("-" * 80)
        
        try:
            # Generate answer using Gemini
            response = await query_engine.answer_synthesizer.synthesize_answer(
                question=question,
                retrieved_chunks=document_chunks,
                retrieval_confidence=90.0
            )
            
            # Display answer
            print()
            print("✨ ANSWER:")
            print(f"   {response.answer}")
            print()
            print(f"📊 Confidence: {response.confidence.overall}%")
            print(f"⚡ Response Time: {response.response_time:.2f}s")
            
            if response.sources:
                print(f"📚 Sources: {len(response.sources)} documents")
            
            print()
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("-" * 80)
            print()
    
    print()
    print("=" * 80)
    print(f"SESSION SUMMARY: {question_count} questions answered")
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
