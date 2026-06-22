"""
Centralized logging system for TechDocRAG.
Provides structured logging with performance tracking and audit trails.
"""

import sys
import json
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from loguru import logger
from contextvars import ContextVar

from ..core.config import Config


# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class PerformanceLogger:
    """Performance tracking decorator and context manager."""
    
    def __init__(self, operation: str, track_memory: bool = False):
        self.operation = operation
        self.track_memory = track_memory
        self.start_time = None
        self.memory_before = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.track_memory:
            import psutil
            self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        
        log_data = {
            'operation': self.operation,
            'duration_ms': round(duration * 1000, 2),
            'request_id': request_id_var.get(),
            'user_id': user_id_var.get(),
        }
        
        if self.track_memory and self.memory_before:
            import psutil
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            log_data['memory_delta_mb'] = round(memory_after - self.memory_before, 2)
            
        if exc_type:
            logger.error(f"Operation failed: {self.operation}", extra=log_data)
        else:
            logger.info(f"Operation completed: {self.operation}", extra=log_data)


def performance_log(operation: str, track_memory: bool = False):
    """Decorator for automatic performance logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceLogger(f"{func.__module__}.{func.__name__}::{operation}", track_memory):
                return await func(*args, **kwargs)
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with PerformanceLogger(f"{func.__module__}.{func.__name__}::{operation}", track_memory):
                return func(*args, **kwargs)
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class AuditLogger:
    """Specialized logger for audit trails and compliance."""
    
    def __init__(self):
        self.config = Config()
        self.audit_file = Path(self.config.security.audit_log_path)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_document_access(self, doc_id: str, action: str, user_id: str = None, 
                           details: Dict[str, Any] = None):
        """Log document access for audit trail."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'document_access',
            'doc_id': doc_id,
            'action': action,  # upload, process, query, download
            'user_id': user_id or user_id_var.get(),
            'request_id': request_id_var.get(),
            'details': details or {}
        }
        
        logger.bind(audit=True).info("Document access", extra=audit_entry)
        
    def log_pii_detection(self, doc_id: str, pii_types: list, masked_count: int):
        """Log PII detection and masking."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'pii_detection',
            'doc_id': doc_id,
            'pii_types': pii_types,
            'masked_count': masked_count,
            'user_id': user_id_var.get(),
            'request_id': request_id_var.get(),
        }
        
        logger.bind(audit=True).warning("PII detected and masked", extra=audit_entry)
        
    def log_calculation(self, operation: str, inputs: Dict[str, Any], 
                       result: Any, confidence: float):
        """Log calculations for audit trail."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'calculation',
            'operation': operation,
            'inputs': inputs,
            'result': result,
            'confidence': confidence,
            'user_id': user_id_var.get(),
            'request_id': request_id_var.get(),
        }
        
        logger.bind(audit=True).info("Calculation performed", extra=audit_entry)


def setup_logging():
    """Configure logging system with structured output."""
    config = Config()
    
    # Remove default logger
    logger.remove()
    
    # Console logging with rich formatting
    logger.add(
        sys.stdout,
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        enqueue=True
    )
    
    # File logging with JSON format for structured logs
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "techdocrag.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="1 month",
        compression="gz",
        enqueue=True
    )
    
    # Separate audit log for compliance
    if config.security.enable_audit_logging:
        logger.add(
            config.security.audit_log_path,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {message}",
            filter=lambda record: record.get("extra", {}).get("audit", False),
            rotation="100 MB",
            retention="1 year",
            compression="gz",
            enqueue=True
        )
    
    # Error logging to separate file
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="3 months",
        compression="gz",
        enqueue=True
    )
    
    return logger


# Global instances
audit_logger = AuditLogger()


def get_logger(name: str = None):
    """Get logger instance for module."""
    if name:
        return logger.bind(module=name)
    return logger


# Context managers for request tracking
def set_request_context(request_id: str, user_id: str = None):
    """Set request context for logging."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def log_exception(exception: Exception, context: Dict[str, Any] = None):
    """Log exception with context."""
    log_data = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'request_id': request_id_var.get(),
        'user_id': user_id_var.get(),
        'context': context or {}
    }
    
    logger.exception("Exception occurred", extra=log_data)


# Import asyncio at module level to avoid import issues
import asyncio
