"""
Helper utilities for TechDocRAG system.
Common utility functions used across the system.
"""

import os
import uuid
import hashlib
import re
from pathlib import Path
from typing import Union, List, Optional
from datetime import datetime


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate unique identifier with optional prefix."""
    unique_id = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Create a safe filename by removing/replacing problematic characters."""
    # Remove or replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Trim length if necessary
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:max_length - len(ext)] + ext
        
    return safe_name.strip('_')


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def hash_content(content: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Generate hash of content for caching and deduplication."""
    if isinstance(content, str):
        content = content.encode('utf-8')
        
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(content)
    return hash_obj.hexdigest()


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using common patterns."""
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',  # DD Month YYYY
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches if matches and isinstance(matches[0], str) else [])
    
    return dates


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    return text


def is_valid_email(email: str) -> bool:
    """Check if string is a valid email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """Check if string is a valid phone number."""
    # Remove non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a valid length (7-15 digits)
    return 7 <= len(digits) <= 15


def mask_sensitive_data(text: str, entity_type: str = 'GENERIC') -> str:
    """Mask sensitive data based on type."""
    patterns = {
        'EMAIL': (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
        'PHONE': (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        'SSN': (r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN]'),
        'CREDIT_CARD': (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]'),
        'GENERIC': (r'\b\w+\b', '[REDACTED]')
    }
    
    if entity_type in patterns:
        pattern, replacement = patterns[entity_type]
        return re.sub(pattern, replacement, text)
    
    return text


def get_file_metadata(file_path: Union[str, Path]) -> dict:
    """Get comprehensive file metadata."""
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        'name': path.name,
        'size': stat.st_size,
        'size_formatted': format_bytes(stat.st_size),
        'extension': path.suffix.lower(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'absolute_path': str(path.absolute())
    }


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to specified range."""
    return max(min_val, min(max_val, score))


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average of values."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
        
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
        
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            
        return wrapper
    return decorator
