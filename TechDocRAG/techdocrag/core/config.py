"""
Configuration management for TechDocRAG system.
Centralized configuration with environment-specific overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class OCRConfig:
    """OCR engine configuration."""
    primary_engine: str = "tesseract"  # tesseract, azure, aws
    fallback_engines: list = field(default_factory=lambda: ["tesseract"])
    tesseract_path: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    aws_region: str = "us-east-1"
    confidence_threshold: float = 0.6


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    model_provider: str = "sentence-transformers"  # sentence-transformers, openai, huggingface
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda
    cache_embeddings: bool = True


@dataclass
class FAISSConfig:
    """FAISS specific configuration."""
    index_type: str = "flat"  # flat, ivf, hnsw
    nlist: int = 100
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    use_gpu: bool = False
    index_path: Optional[str] = None


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    provider: str = "faiss"  # faiss, chroma, pinecone, weaviate
    index_type: str = "flat"  # flat, ivf, hnsw
    similarity_metric: str = "cosine"
    storage_path: str = "./data/vector_store"
    embedding_dimension: int = 384  # Default for sentence transformers
    
    # FAISS specific
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    # Chroma specific
    chroma_host: str = "localhost"
    chroma_port: int = 8000


@dataclass
class LLMConfig:
    """Language model configuration."""
    provider: str = "gemini"  # openai, anthropic, gemini, huggingface, local
    model_name: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3
    enable_synthesis: bool = True  # Enable LLM answer synthesis


@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""
    top_k_semantic: int = 10
    top_k_keyword: int = 5
    fusion_alpha: float = 0.6  # weight for semantic vs keyword
    semantic_weight: float = 0.6  # alias for fusion_alpha for hybrid retrieval
    keyword_weight: float = 0.4  # weight for keyword results (1 - semantic_weight)
    rerank_model: Optional[str] = None
    enable_query_expansion: bool = True
    enable_hypothetical_questions: bool = True
    
    # BM25 specific parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # Keyword search specific parameters
    remove_stop_words: bool = True
    title_boost_factor: int = 2
    
    # Fusion algorithm configuration
    fusion_algorithm: str = "rrf"  # rrf, weighted_sum, rank_based, adaptive
    
    # Storage paths
    index_save_path: str = "./data/indexes"
    
    # RRF parameters
    rrf_k: int = 60  # RRF constant for fusion scoring


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    enable_table_extraction: bool = True
    enable_image_extraction: bool = True
    enable_layout_analysis: bool = True


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    enable_pii_detection: bool = True
    pii_confidence_threshold: float = 0.8
    mask_pii_in_responses: bool = True
    enable_audit_logging: bool = True
    audit_log_path: str = "./logs/audit.log"
    encryption_key_path: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    max_concurrent_requests: int = 100
    request_timeout: int = 300
    cache_size: int = 1000
    enable_batch_processing: bool = True
    batch_size: int = 10
    enable_async_processing: bool = True


@dataclass
class CalculationConfig:
    """Configuration for calculation engine."""
    precision: int = 6
    max_decimal_places: int = 2
    enable_currency_conversion: bool = False
    default_currency: str = "USD"
    enable_statistical_calculations: bool = True
    enable_financial_calculations: bool = True
    calculation_timeout: float = 10.0
    max_calculation_complexity: int = 1000


@dataclass
class ConfidenceConfig:
    """Configuration for confidence calculation."""
    base_confidence: float = 0.5
    evidence_weight: float = 0.3
    source_reliability_weight: float = 0.2
    calculation_precision_weight: float = 0.1
    consistency_weight: float = 0.4
    min_confidence_threshold: float = 0.1
    max_confidence_threshold: float = 0.95
    
    # Additional weights for confidence calculator
    source_weight: float = 0.2
    retrieval_weight: float = 0.25
    reasoning_weight: float = 0.2
    calculation_weight: float = 0.15
    consensus_weight: float = 0.2


class Config:
    """Centralized configuration manager for TechDocRAG."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file and environment variables."""
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()
        self._override_from_env()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.path.dirname(__file__), "../../configs/default.yaml")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}
            
        # Initialize configuration objects
        self.ocr = OCRConfig(**config_data.get('ocr', {}))
        self.embedding = EmbeddingConfig(**config_data.get('embedding', {}))
        self.vector_db = VectorDBConfig(**config_data.get('vector_db', {}))
        self.llm = LLMConfig(**config_data.get('llm', {}))
        self.retrieval = RetrievalConfig(**config_data.get('retrieval', {}))
        self.processing = ProcessingConfig(**config_data.get('processing', {}))
        self.security = SecurityConfig(**config_data.get('security', {}))
        self.performance = PerformanceConfig(**config_data.get('performance', {}))
        self.calculation = CalculationConfig(**config_data.get('calculation', {}))
        self.confidence = ConfidenceConfig(**config_data.get('confidence', {}))
        
        # Additional settings
        self.debug = config_data.get('debug', False)
        self.log_level = config_data.get('log_level', 'INFO')
        self.data_dir = config_data.get('data_dir', './data')
        self.cache_dir = config_data.get('cache_dir', './cache')
        
    def _override_from_env(self):
        """Override configuration from environment variables."""
        # OCR settings
        if os.getenv('AZURE_OCR_ENDPOINT'):
            self.ocr.azure_endpoint = os.getenv('AZURE_OCR_ENDPOINT')
        if os.getenv('AZURE_OCR_KEY'):
            self.ocr.azure_key = os.getenv('AZURE_OCR_KEY')
            
        # LLM settings
        if os.getenv('OPENAI_API_KEY'):
            self.llm.api_key = os.getenv('OPENAI_API_KEY')
        if os.getenv('LLM_MODEL'):
            self.llm.model_name = os.getenv('LLM_MODEL')
            
        # Vector DB settings
        if os.getenv('PINECONE_API_KEY'):
            self.vector_db.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if os.getenv('PINECONE_ENVIRONMENT'):
            self.vector_db.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
            
        # General settings
        if os.getenv('DEBUG'):
            self.debug = os.getenv('DEBUG').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            self.log_level = os.getenv('LOG_LEVEL')
            
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        
        config_data = {
            'ocr': self.ocr.__dict__,
            'embedding': self.embedding.__dict__,
            'vector_db': self.vector_db.__dict__,
            'llm': self.llm.__dict__,
            'retrieval': self.retrieval.__dict__,
            'processing': self.processing.__dict__,
            'security': self.security.__dict__,
            'performance': self.performance.__dict__,
            'debug': self.debug,
            'log_level': self.log_level,
            'data_dir': self.data_dir,
            'cache_dir': self.cache_dir
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
    def get_model_cache_dir(self) -> str:
        """Get model cache directory."""
        cache_dir = os.path.join(self.cache_dir, 'models')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    def get_document_cache_dir(self) -> str:
        """Get document cache directory."""
        cache_dir = os.path.join(self.cache_dir, 'documents')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Check required API keys based on providers
        if self.llm.provider == "openai" and not self.llm.api_key:
            errors.append("OpenAI API key required when using OpenAI provider")
            
        if self.ocr.primary_engine == "azure" and not (self.ocr.azure_endpoint and self.ocr.azure_key):
            errors.append("Azure OCR endpoint and key required when using Azure OCR")
            
        if self.vector_db.provider == "pinecone" and not (self.vector_db.pinecone_api_key and self.vector_db.pinecone_environment):
            errors.append("Pinecone API key and environment required when using Pinecone")
            
        # Check numeric ranges
        if not 0 <= self.retrieval.fusion_alpha <= 1:
            errors.append("Retrieval fusion_alpha must be between 0 and 1")
            
        if self.processing.chunk_overlap >= self.processing.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
            
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
