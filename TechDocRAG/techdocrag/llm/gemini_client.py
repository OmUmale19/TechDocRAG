"""
Gemini LLM Client for TechDocRAG
Handles integration with Google's Gemini API for answer generation.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.exceptions import LLMError

logger = get_logger(__name__)


class GeminiClient:
    """
    Client for Google Gemini LLM API.
    
    Handles:
    - API authentication
    - Prompt construction
    - Response generation
    - Error handling and retries
    - Rate limiting
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 30
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (will use GEMINI_API_KEY env var if not provided)
            model_name: Model to use (gemini-pro, gemini-pro-vision, etc.)
            temperature: Response randomness (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter.",
                error_code="MISSING_API_KEY"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            candidate_count=1
        )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"GeminiClient initialized with model: {model_name}")
    
    async def generate_answer(
        self,
        question: str,
        context: List[str],
        sources: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate natural language answer from question and retrieved context.
        
        Args:
            question: User's question
            context: List of retrieved text chunks
            sources: List of source metadata (doc_id, title, etc.)
            system_prompt: Optional system prompt override
        
        Returns:
            Dict containing:
                - answer: Generated natural language answer
                - confidence: Confidence score (0-100)
                - reasoning: Explanation of how answer was derived
                - citations: List of source citations used
        """
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        try:
            # Build prompt
            prompt = self._build_prompt(question, context, sources, system_prompt)
            
            # Generate response
            response = await self._generate_with_retry(prompt)
            
            # Parse response
            result = self._parse_response(response, sources)
            
            # Update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_requests'] += 1
            self._update_avg_response_time(response_time)
            
            logger.info(
                f"Generated answer for question in {response_time:.2f}s",
                extra={'question': question[:100], 'confidence': result.get('confidence', 0)}
            )
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Answer generation failed: {str(e)}", extra={'question': question[:100]})
            raise LLMError(
                f"Failed to generate answer: {str(e)}",
                error_code="GENERATION_FAILED",
                details={'question': question, 'error': str(e)}
            ) from e
    
    def _build_prompt(
        self,
        question: str,
        context: List[str],
        sources: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Build the complete prompt for Gemini."""
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = """You are an intelligent document assistant helping users find information from their documents.

Your task is to:
1. Analyze the provided document chunks carefully
2. Generate a clear, accurate answer to the user's question
3. Include specific citations from the sources
4. Provide confidence level and reasoning

Format your response EXACTLY as follows:

ANSWER: [Your natural language answer here with specific facts and figures]

SOURCES: [List which documents you used, e.g., "Company A Annual Report (Page 2), Market Analysis"]

CONFIDENCE: [A number between 0-100 representing your confidence]

REASONING: [Brief explanation of how you derived this answer]

Be specific, use exact numbers when available, and always cite your sources."""

        # Build context section
        context_section = "\n\n".join([
            f"[DOCUMENT {i+1}]\n{chunk}"
            for i, chunk in enumerate(context[:5])  # Limit to top 5 chunks
        ])
        
        # Build source information
        source_info = "\n".join([
            f"- Document {i+1}: {src.get('title', 'Unknown')} (ID: {src.get('doc_id', 'N/A')})"
            for i, src in enumerate(sources[:5])
        ])
        
        # Complete prompt
        prompt = f"""{system_prompt}

================================================================================
AVAILABLE DOCUMENTS:
{source_info}

================================================================================
DOCUMENT CONTENT:
{context_section}

================================================================================
USER QUESTION:
{question}

================================================================================
YOUR RESPONSE:"""
        
        return prompt
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with retry logic."""
        
        for attempt in range(max_retries):
            try:
                # Use asyncio to run the synchronous Gemini API call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )
                )
                
                # Extract text from response
                return response.text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Gemini API call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _parse_response(self, response_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse structured response from Gemini."""
        
        result = {
            'answer': '',
            'confidence': 0,
            'reasoning': '',
            'citations': []
        }
        
        try:
            # Split response into sections
            lines = response_text.strip().split('\n')
            current_section = None
            
            for line in lines:
                line_upper = line.strip().upper()
                
                if line_upper.startswith('ANSWER:'):
                    current_section = 'answer'
                    result['answer'] = line[7:].strip()  # Remove "ANSWER:"
                elif line_upper.startswith('SOURCES:'):
                    current_section = 'sources'
                    citations_text = line[8:].strip()  # Remove "SOURCES:"
                    if citations_text:
                        result['citations'] = [c.strip() for c in citations_text.split(',')]
                elif line_upper.startswith('CONFIDENCE:'):
                    current_section = 'confidence'
                    conf_text = line[11:].strip()  # Remove "CONFIDENCE:"
                    # Extract number from text
                    import re
                    numbers = re.findall(r'\d+', conf_text)
                    if numbers:
                        result['confidence'] = min(100, max(0, int(numbers[0])))
                elif line_upper.startswith('REASONING:'):
                    current_section = 'reasoning'
                    result['reasoning'] = line[10:].strip()  # Remove "REASONING:"
                elif line.strip() and current_section:
                    # Continue current section
                    if current_section == 'answer':
                        result['answer'] += ' ' + line.strip()
                    elif current_section == 'reasoning':
                        result['reasoning'] += ' ' + line.strip()
            
            # Fallback: if parsing failed, use entire response as answer
            if not result['answer']:
                result['answer'] = response_text.strip()
                result['confidence'] = 50  # Medium confidence if parsing failed
                result['reasoning'] = "Answer extracted from LLM response"
            
            # Add source metadata to citations
            result['source_metadata'] = sources
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response structure: {str(e)}")
            result['answer'] = response_text.strip()
            result['confidence'] = 50
            result['reasoning'] = "Raw LLM response (parsing failed)"
        
        return result
    
    def _update_avg_response_time(self, new_time: float):
        """Update running average of response times."""
        total = self.stats['successful_requests']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (current_avg * (total - 1) + new_time) / total
    
    async def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            response = await self._generate_with_retry("Say 'Hello'")
            logger.info("Gemini API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            'model_name': self.model_name,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests'] * 100
                if self.stats['total_requests'] > 0 else 0
            )
        }
