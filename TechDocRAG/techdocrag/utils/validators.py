"""
Input validation utilities for TechDocRAG system.
Ensures data quality and security through comprehensive validation.
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from .exceptions import ValidationError


# Supported file types
SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.txt': 'text/plain',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff'
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZES = {
    'pdf': 100 * 1024 * 1024,     # 100 MB
    'docx': 50 * 1024 * 1024,     # 50 MB
    'doc': 50 * 1024 * 1024,      # 50 MB
    'txt': 10 * 1024 * 1024,      # 10 MB
    'image': 20 * 1024 * 1024,    # 20 MB per image
}

# Dangerous patterns to reject
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # JavaScript
    r'javascript:',                # JavaScript protocol
    r'vbscript:',                 # VBScript
    r'on\w+\s*=',                # Event handlers
    r'data:.*base64',             # Base64 data URLs
    r'<iframe[^>]*>',             # iframes
    r'<object[^>]*>',             # Object tags
    r'<embed[^>]*>',              # Embed tags
]


def validate_file_type(file_path: Union[str, Path], allowed_types: List[str] = None) -> bool:
    """
    Validate file type based on extension and MIME type.
    
    Args:
        file_path: Path to the file
        allowed_types: List of allowed extensions (default: all supported)
        
    Returns:
        True if file type is valid
        
    Raises:
        ValidationError: If file type is not supported
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if allowed_types is None:
        allowed_types = list(SUPPORTED_EXTENSIONS.keys())
    
    # Check extension
    if extension not in allowed_types:
        raise ValidationError(
            f"Unsupported file type: {extension}. Allowed types: {allowed_types}",
            error_code="INVALID_FILE_TYPE",
            details={'extension': extension, 'allowed': allowed_types}
        )
    
    # Check MIME type if file exists
    if path.exists():
        mime_type, _ = mimetypes.guess_type(str(path))
        expected_mime = SUPPORTED_EXTENSIONS.get(extension)
        
        if mime_type != expected_mime:
            # Some flexibility for common variations
            if not _is_mime_compatible(mime_type, expected_mime):
                raise ValidationError(
                    f"MIME type mismatch: expected {expected_mime}, got {mime_type}",
                    error_code="MIME_TYPE_MISMATCH",
                    details={'expected': expected_mime, 'actual': mime_type}
                )
    
    return True


def validate_file_size(file_path: Union[str, Path], max_size: Optional[int] = None) -> bool:
    """
    Validate file size against limits.
    
    Args:
        file_path: Path to the file
        max_size: Maximum size in bytes (default: based on file type)
        
    Returns:
        True if file size is valid
        
    Raises:
        ValidationError: If file is too large
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValidationError(
            f"File does not exist: {path}",
            error_code="FILE_NOT_FOUND"
        )
    
    file_size = path.stat().st_size
    extension = path.suffix.lower()
    
    # Determine max size
    if max_size is None:
        if extension == '.pdf':
            max_size = MAX_FILE_SIZES['pdf']
        elif extension in ['.docx', '.doc']:
            max_size = MAX_FILE_SIZES['docx']
        elif extension == '.txt':
            max_size = MAX_FILE_SIZES['txt']
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
            max_size = MAX_FILE_SIZES['image']
        else:
            max_size = MAX_FILE_SIZES['pdf']  # Default fallback
    
    if file_size > max_size:
        raise ValidationError(
            f"File too large: {file_size} bytes exceeds limit of {max_size} bytes",
            error_code="FILE_TOO_LARGE",
            details={
                'file_size': file_size,
                'max_size': max_size,
                'size_mb': round(file_size / 1024 / 1024, 2)
            }
        )
    
    return True


def sanitize_input(text: str, max_length: int = 10000, allow_html: bool = False) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags
        
    Returns:
        Sanitized text
        
    Raises:
        ValidationError: If input is dangerous or too long
    """
    if not isinstance(text, str):
        raise ValidationError(
            "Input must be a string",
            error_code="INVALID_INPUT_TYPE"
        )
    
    if len(text) > max_length:
        raise ValidationError(
            f"Input too long: {len(text)} characters exceeds limit of {max_length}",
            error_code="INPUT_TOO_LONG",
            details={'length': len(text), 'max_length': max_length}
        )
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValidationError(
                "Input contains potentially dangerous content",
                error_code="DANGEROUS_CONTENT",
                details={'pattern': pattern}
            )
    
    # Basic sanitization
    sanitized = text.strip()
    
    if not allow_html:
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Escape special characters
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
    
    return sanitized


def validate_query(query: str) -> str:
    """
    Validate and sanitize search queries.
    
    Args:
        query: Search query string
        
    Returns:
        Sanitized query
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError(
            "Query cannot be empty",
            error_code="EMPTY_QUERY"
        )
    
    sanitized_query = sanitize_input(query, max_length=1000, allow_html=False)
    
    # Additional query-specific validation
    if len(sanitized_query.split()) > 50:  # Reasonable word limit
        raise ValidationError(
            "Query too complex: maximum 50 words allowed",
            error_code="QUERY_TOO_COMPLEX"
        )
    
    return sanitized_query


def validate_document_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate document metadata fields.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Validated metadata
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValidationError(
            "Metadata must be a dictionary",
            error_code="INVALID_METADATA_TYPE"
        )
    
    validated = {}
    
    # Validate each field
    for key, value in metadata.items():
        # Sanitize key
        clean_key = sanitize_input(str(key), max_length=100, allow_html=False)
        
        # Sanitize value based on type
        if isinstance(value, str):
            clean_value = sanitize_input(value, max_length=1000, allow_html=False)
        elif isinstance(value, (int, float, bool)):
            clean_value = value
        elif isinstance(value, list):
            clean_value = [sanitize_input(str(item), max_length=500, allow_html=False) 
                          for item in value[:10]]  # Limit list size
        else:
            clean_value = sanitize_input(str(value), max_length=1000, allow_html=False)
        
        validated[clean_key] = clean_value
    
    return validated


def validate_confidence_score(score: float) -> float:
    """
    Validate confidence score is within valid range.
    
    Args:
        score: Confidence score
        
    Returns:
        Validated score
        
    Raises:
        ValidationError: If score is invalid
    """
    if not isinstance(score, (int, float)):
        raise ValidationError(
            "Confidence score must be a number",
            error_code="INVALID_CONFIDENCE_TYPE"
        )
    
    if not 0.0 <= score <= 1.0:
        raise ValidationError(
            f"Confidence score must be between 0.0 and 1.0, got {score}",
            error_code="INVALID_CONFIDENCE_RANGE",
            details={'score': score}
        )
    
    return float(score)


def validate_page_number(page: int, total_pages: int = None) -> int:
    """
    Validate page number.
    
    Args:
        page: Page number
        total_pages: Total pages (optional)
        
    Returns:
        Validated page number
        
    Raises:
        ValidationError: If page number is invalid
    """
    if not isinstance(page, int):
        raise ValidationError(
            "Page number must be an integer",
            error_code="INVALID_PAGE_TYPE"
        )
    
    if page < 1:
        raise ValidationError(
            f"Page number must be positive, got {page}",
            error_code="INVALID_PAGE_NUMBER"
        )
    
    if total_pages and page > total_pages:
        raise ValidationError(
            f"Page number {page} exceeds total pages {total_pages}",
            error_code="PAGE_OUT_OF_RANGE",
            details={'page': page, 'total_pages': total_pages}
        )
    
    return page


def _is_mime_compatible(actual: str, expected: str) -> bool:
    """Check if MIME types are compatible (allowing some variations)."""
    if actual == expected:
        return True
    
    # Common variations
    compatible_types = {
        'application/pdf': ['application/x-pdf'],
        'image/jpeg': ['image/jpg'],
        'text/plain': ['text/x-plain', 'application/x-empty'],
    }
    
    return actual in compatible_types.get(expected, [])


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> int:
    """
    Validate batch processing size.
    
    Args:
        batch_size: Requested batch size
        max_batch_size: Maximum allowed batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise ValidationError(
            "Batch size must be an integer",
            error_code="INVALID_BATCH_SIZE_TYPE"
        )
    
    if batch_size < 1:
        raise ValidationError(
            f"Batch size must be positive, got {batch_size}",
            error_code="INVALID_BATCH_SIZE"
        )
    
    if batch_size > max_batch_size:
        raise ValidationError(
            f"Batch size {batch_size} exceeds maximum {max_batch_size}",
            error_code="BATCH_SIZE_TOO_LARGE",
            details={'batch_size': batch_size, 'max_batch_size': max_batch_size}
        )
    
    return batch_size
