"""
Calculation engine for numerical computations in TechDocRAG.
Handles mathematical expressions, validations, and financial calculations.
"""

import asyncio
import re
import math
from typing import List, Dict, Any, Optional, Union
from decimal import Decimal, ROUND_HALF_UP

from ..core.types import CalculationResult
from ..core.config import CalculationConfig
from ..utils.logging import get_logger, performance_log
from ..utils.exceptions import CalculationError

logger = get_logger(__name__)


class CalculationEngine:
    """
    Advanced calculation engine for document analysis.
    
    Features:
    - Safe mathematical expression evaluation
    - Financial calculations (tax, totals, percentages)
    - Unit conversions
    - Statistical operations
    - Validation and error handling
    """
    
    def __init__(self, config: CalculationConfig):
        self.config = config
        
        # Safe mathematical functions
        self.safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'sqrt': math.sqrt,
            'pow': pow,
            'log': math.log,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # Calculation statistics
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'calculation_types': {}
        }
        
        logger.info("CalculationEngine initialized")
    
    @performance_log("calculation")
    async def calculate(
        self,
        expression: str,
        context: Dict[str, Any] = None,
        calculation_type: str = 'general'
    ) -> CalculationResult:
        """
        Perform calculation on mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            context: Additional context variables
            calculation_type: Type of calculation (general, financial, statistical)
            
        Returns:
            CalculationResult with result and metadata
        """
        self.stats['total_calculations'] += 1
        self.stats['calculation_types'][calculation_type] = (
            self.stats['calculation_types'].get(calculation_type, 0) + 1
        )
        
        try:
            logger.debug(f"Calculating: {expression} (type: {calculation_type})")
            
            # Preprocess expression
            cleaned_expression = self._preprocess_expression(expression)
            
            # Validate expression
            if not self._is_safe_expression(cleaned_expression):
                raise CalculationError(f"Unsafe expression: {expression}")
            
            # Perform calculation based on type
            if calculation_type == 'financial':
                result = await self._calculate_financial(cleaned_expression, context)
            elif calculation_type == 'statistical':
                result = await self._calculate_statistical(cleaned_expression, context)
            else:
                result = await self._calculate_general(cleaned_expression, context)
            
            # Create result object
            calc_result = CalculationResult(
                expression=expression,
                result=result['value'],
                calculation_type=calculation_type,
                confidence=result.get('confidence', 0.9),
                metadata=result.get('metadata', {}),
                error=None
            )
            
            self.stats['successful_calculations'] += 1
            logger.debug(f"Calculation successful: {expression} = {result['value']}")
            
            return calc_result
            
        except Exception as e:
            self.stats['failed_calculations'] += 1
            logger.error(f"Calculation failed for {expression}: {str(e)}")
            
            return CalculationResult(
                expression=expression,
                result=None,
                calculation_type=calculation_type,
                confidence=0.0,
                metadata={'error_type': type(e).__name__},
                error=str(e)
            )
    
    def _preprocess_expression(self, expression: str) -> str:
        """Preprocess mathematical expression for safe evaluation."""
        # Remove currency symbols and formatting
        cleaned = re.sub(r'[$,€£¥]', '', expression)
        
        # Replace percentage with /100
        cleaned = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', cleaned)
        
        # Handle common mathematical operations
        cleaned = re.sub(r'\bof\b', '*', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bplus\b', '+', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bminus\b', '-', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\btimes\b', '*', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bdivided by\b', '/', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', '', cleaned)
        
        return cleaned
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe to evaluate."""
        # List of dangerous patterns
        dangerous_patterns = [
            r'__',  # Dunder methods
            r'import',  # Import statements
            r'exec',  # Code execution
            r'eval',  # Dynamic evaluation
            r'open',  # File operations
            r'file',  # File operations
            r'input',  # User input
            r'raw_input',  # User input
            r'compile',  # Code compilation
            r'globals',  # Global namespace
            r'locals',  # Local namespace
            r'vars',  # Variable access
            r'dir',  # Directory listing
            r'help',  # Help system
            r'quit',  # Exit commands
            r'exit',  # Exit commands
        ]
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False
        
        # Only allow specific characters
        allowed_chars = r'[0-9+\-*/().,\s]'
        if not re.match(f'^{allowed_chars}+$', expression):
            return False
        
        return True
    
    async def _calculate_general(
        self,
        expression: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform general mathematical calculation."""
        try:
            # Use eval with restricted namespace for safety
            namespace = {
                '__builtins__': {},
                **self.safe_functions
            }
            
            # Add context variables if provided
            if context:
                for key, value in context.items():
                    if isinstance(value, (int, float)):
                        namespace[key] = value
            
            result = eval(expression, namespace)
            
            return {
                'value': float(result),
                'confidence': 0.95,
                'metadata': {
                    'method': 'eval',
                    'namespace_size': len(namespace)
                }
            }
            
        except Exception as e:
            raise CalculationError(f"General calculation failed: {str(e)}")
    
    async def _calculate_financial(
        self,
        expression: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform financial calculation with higher precision."""
        try:
            # Use Decimal for financial precision
            decimal_expression = expression
            
            # Convert numbers to Decimal
            numbers = re.findall(r'\d+(?:\.\d+)?', expression)
            for num in numbers:
                decimal_num = str(Decimal(num))
                decimal_expression = decimal_expression.replace(num, decimal_num, 1)
            
            # Create namespace with Decimal operations
            namespace = {
                '__builtins__': {},
                'Decimal': Decimal,
                'ROUND_HALF_UP': ROUND_HALF_UP,
                'abs': abs,
                'min': min,
                'max': max,
                'round': round
            }
            
            # Add context variables as Decimal
            if context:
                for key, value in context.items():
                    if isinstance(value, (int, float)):
                        namespace[key] = Decimal(str(value))
            
            result = eval(decimal_expression, namespace)
            
            # Convert back to float for consistency
            if isinstance(result, Decimal):
                result = float(result.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            
            return {
                'value': result,
                'confidence': 0.98,
                'metadata': {
                    'method': 'decimal_precision',
                    'precision': '0.01'
                }
            }
            
        except Exception as e:
            # Fallback to general calculation
            return await self._calculate_general(expression, context)
    
    async def _calculate_statistical(
        self,
        expression: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform statistical calculation."""
        try:
            # Extract numbers from expression for statistical operations
            numbers = re.findall(r'-?\d+(?:\.\d+)?', expression)
            values = [float(num) for num in numbers]
            
            if not values:
                raise CalculationError("No numbers found for statistical calculation")
            
            # Determine statistical operation
            if 'average' in expression.lower() or 'mean' in expression.lower():
                result = sum(values) / len(values)
                operation = 'mean'
            elif 'median' in expression.lower():
                sorted_values = sorted(values)
                n = len(sorted_values)
                if n % 2 == 0:
                    result = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                else:
                    result = sorted_values[n//2]
                operation = 'median'
            elif 'sum' in expression.lower() or 'total' in expression.lower():
                result = sum(values)
                operation = 'sum'
            elif 'max' in expression.lower() or 'maximum' in expression.lower():
                result = max(values)
                operation = 'maximum'
            elif 'min' in expression.lower() or 'minimum' in expression.lower():
                result = min(values)
                operation = 'minimum'
            else:
                # Default to sum for statistical context
                result = sum(values)
                operation = 'sum'
            
            return {
                'value': result,
                'confidence': 0.9,
                'metadata': {
                    'method': 'statistical',
                    'operation': operation,
                    'sample_size': len(values),
                    'values': values
                }
            }
            
        except Exception as e:
            raise CalculationError(f"Statistical calculation failed: {str(e)}")
    
    async def calculate_invoice_total(
        self,
        line_items: List[Dict[str, float]],
        tax_rate: float = 0.0,
        discount: float = 0.0
    ) -> CalculationResult:
        """Calculate invoice total from line items."""
        try:
            subtotal = 0.0
            
            # Calculate subtotal
            for item in line_items:
                quantity = item.get('quantity', 1)
                unit_price = item.get('unit_price', 0.0)
                item_total = quantity * unit_price
                subtotal += item_total
            
            # Apply discount
            discount_amount = subtotal * discount
            discounted_subtotal = subtotal - discount_amount
            
            # Calculate tax
            tax_amount = discounted_subtotal * tax_rate
            
            # Calculate total
            total = discounted_subtotal + tax_amount
            
            return CalculationResult(
                expression=f"Invoice calculation: {len(line_items)} items",
                result=total,
                calculation_type='financial',
                confidence=0.95,
                metadata={
                    'subtotal': subtotal,
                    'discount_rate': discount,
                    'discount_amount': discount_amount,
                    'tax_rate': tax_rate,
                    'tax_amount': tax_amount,
                    'line_items_count': len(line_items)
                },
                error=None
            )
            
        except Exception as e:
            return CalculationResult(
                expression="Invoice calculation",
                result=None,
                calculation_type='financial',
                confidence=0.0,
                metadata={},
                error=str(e)
            )
    
    async def calculate_percentage(
        self,
        part: float,
        whole: float
    ) -> CalculationResult:
        """Calculate percentage."""
        try:
            if whole == 0:
                raise CalculationError("Cannot calculate percentage with zero denominator")
            
            percentage = (part / whole) * 100
            
            return CalculationResult(
                expression=f"({part} / {whole}) * 100",
                result=percentage,
                calculation_type='general',
                confidence=0.98,
                metadata={
                    'part': part,
                    'whole': whole,
                    'operation': 'percentage'
                },
                error=None
            )
            
        except Exception as e:
            return CalculationResult(
                expression=f"Percentage: {part} of {whole}",
                result=None,
                calculation_type='general',
                confidence=0.0,
                metadata={},
                error=str(e)
            )
    
    async def validate_calculation(
        self,
        expected: float,
        calculated: float,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """Validate calculated result against expected value."""
        difference = abs(expected - calculated)
        is_valid = difference <= tolerance
        
        return {
            'is_valid': is_valid,
            'expected': expected,
            'calculated': calculated,
            'difference': difference,
            'tolerance': tolerance,
            'percentage_error': (difference / max(abs(expected), 0.01)) * 100
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get calculation engine statistics."""
        success_rate = 0.0
        if self.stats['total_calculations'] > 0:
            success_rate = (
                self.stats['successful_calculations'] / 
                self.stats['total_calculations'] * 100
            )
        
        return {
            'total_calculations': self.stats['total_calculations'],
            'successful_calculations': self.stats['successful_calculations'],
            'failed_calculations': self.stats['failed_calculations'],
            'success_rate': success_rate,
            'calculation_types': self.stats['calculation_types'].copy()
        }
