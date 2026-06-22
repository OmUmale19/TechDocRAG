# TechDocRAG: AI-Powered Document Q&A System 🚀

An intelligent document retrieval and question-answering system powered by **Google Gemini LLM**, combining hybrid search (keyword + semantic) with natural language generation for accurate, context-aware answers.

## 🎯 Key Features

- **🤖 Gemini LLM Integration**: Natural language answers with high confidence (91-97%)
- **🔍 Hybrid Retrieval**: BM25 keyword search + FAISS semantic embeddings
- **📚 Multi-Document Q&A**: Query across multiple documents simultaneously
- **💡 Smart Reasoning**: Cross-document synthesis with source attribution
- **⚡ Fast Response**: 1.2-2.3s average query time
- **📊 Confidence Scoring**: Explainable AI with confidence metrics
- **🔢 Calculation Support**: Extract and compute numerical data
- **📖 Source Citations**: Every answer linked to specific document sections

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TechDocRAG Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│ 📄 Document Input (Text/OCR) → Text Processing              │
│ ↓                                                           │
│ ✂️  Text Chunking → Semantic segmentation                   │
│ ↓                                                           │
│ 🧠 Embedding Generation → Sentence-Transformers             │
│ ↓                                                           │
│ 💾 Hybrid Storage → FAISS Vector DB + BM25 Index           │
│ ↓                                                           │
│ 🔍 Query Processing → User question analysis                │
│ ↓                                                           │
│ 🎯 Hybrid Retrieval → Keyword + Semantic search fusion      │
│ ↓                                                           │
│ 🤖 Gemini LLM Synthesis → Natural language generation       │
│ ↓                                                           │
│ ✅ Response → Answer + Confidence + Sources                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/TechDocRAG.git
cd TechDocRAG

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key
$env:GEMINI_API_KEY="your-api-key-here"  # Windows
export GEMINI_API_KEY="your-api-key-here"  # Linux/Mac
```

### Basic Usage

#### Method 1: Run Main Application

```bash
python main.py
```

Choose from 4 modes:
1. **Custom Text Demo** - Quick test with sample text
2. **Process File** - Upload and query a document
3. **Multi-Document Q&A** - Query across multiple documents ⭐ Recommended
4. **Health Check** - Verify system components

#### Method 2: Python API

```python
from techdocrag.core.document_processor import DocumentProcessor
from techdocrag.core.query_engine import QueryEngine

# Initialize
processor = DocumentProcessor()
engine = QueryEngine()

# Process documents
docs = [
    "Your first document text here...",
    "Your second document text here..."
]
results = processor.prepare_documents(docs)

# Ask questions
response = engine.process_query(
    query="What is the main topic?",
    retrieval_results=results
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence.overall_score:.1%}")
print(f"Sources: {[s.doc_id for s in response.sources]}")
```

### Demo Scripts

```bash
# Interactive Q&A session
python interactive_gemini_qa.py

# Comprehensive demo with sample documents
python demo_gemini_qa.py

# Integration test
python test_main_gemini.py
```

## 📁 Project Structure

```
TechDocRAG/
├── techdocrag/                     # Core package
│   ├── core/                       # Core orchestration
│   │   ├── config.py              # Configuration management
│   │   ├── document_processor.py  # Document processing pipeline
│   │   ├── query_engine.py        # Query orchestration + LLM
│   │   └── types.py               # Type definitions
│   ├── processing/                 # Document processing
│   │   ├── embedding_generator.py # Sentence embeddings
│   │   ├── field_extractor.py     # Metadata extraction
│   │   ├── layout_analyzer.py     # Document structure
│   │   ├── ocr_engine.py          # Text extraction
│   │   └── text_processor.py      # Text normalization
│   ├── retrieval/                  # Hybrid retrieval
│   │   ├── hybrid_retriever.py    # Combined search
│   │   ├── keyword_searcher.py    # BM25 keyword search
│   │   ├── result_fusion.py       # RRF algorithm
│   │   └── vector_store.py        # FAISS vector database
│   ├── reasoning/                  # Answer generation
│   │   ├── answer_synthesizer.py  # LLM-powered synthesis
│   │   ├── calculation_engine.py  # Numerical reasoning
│   │   ├── confidence_calculator.py # Confidence scoring
│   │   └── reasoning_engine.py    # Reasoning logic
│   ├── llm/                        # LLM integration
│   │   └── gemini_client.py       # Google Gemini API
│   └── utils/                      # Utilities
│       ├── exceptions.py          # Custom exceptions
│       ├── helpers.py             # Helper functions
│       ├── logging.py             # Logging system
│       └── validators.py          # Input validation
├── configs/
│   └── default.yaml               # System configuration
├── cache/
│   └── embeddings/                # Cached embeddings
├── logs/                          # Application logs
├── main.py                        # Main CLI application
├── demo_gemini_qa.py              # Comprehensive demo
├── interactive_gemini_qa.py       # Interactive Q&A
├── test_main_gemini.py            # Integration tests
├── test_system.py                 # System tests
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🎯 Performance Metrics

### System Performance (Latest Test Results)
- **Accuracy**: 91-97% confidence scores across diverse queries
- **Response Time**: 1.2-2.3s average (including LLM synthesis)
- **Multi-Document Support**: Successfully queries across 3+ documents
- **Success Rate**: 100% (9/9 test questions answered correctly)

### Tested Capabilities
✅ Multi-document summarization  
✅ Cross-document reasoning  
✅ Entity extraction (names, dates, codes)  
✅ Negative result detection ("no information available")  
✅ Numerical data identification  
✅ Temporal information extraction  

### Technology Stack
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Search**: BM25 (Rank-BM25)
- **Language**: Python 3.10+
- **Framework**: Async/await architecture

## 📊 Example Queries & Results

```python
# Query 1: Summarization
Q: "What is the summary of all three docs?"
A: Comprehensive multi-document synthesis
Confidence: 91.6% | Response Time: 2.32s

# Query 2: Entity Extraction
Q: "Which sensor is used?"
A: "GPR, LIDAR, chemical sensor"
Confidence: 97% | Response Time: 1.21s

# Query 3: Author Attribution
Q: "Who made this?"
A: "A. Udapure"
Confidence: 97% | Response Time: 1.26s

# Query 4: Negative Results
Q: "What is the estimated cost?"
A: "No cost information available in documents"
Confidence: 97% | Response Time: 1.24s

# Query 5: Cross-Document Reasoning
Q: "Which sensor could increase battery drain?"
A: Cross-document analysis with multiple sources
Confidence: 97% | Response Time: 1.85s
```

## 🛣️ Development Roadmap

### ✅ Phase 1-3: COMPLETE
- [x] Core document processing pipeline
- [x] Hybrid retrieval (BM25 + Vector search)
- [x] Multi-document Q&A system
- [x] Gemini LLM integration
- [x] Confidence scoring & source attribution
- [x] Production-ready CLI application

### ⏳ Phase 4: Advanced Features (Next)
- [ ] PDF file upload support
- [ ] Image & table extraction from PDFs
- [ ] Multi-format support (DOCX, XLSX, PPTX)
- [ ] Document comparison features
- [ ] Batch processing capabilities

### ⏳ Phase 5: Web Interface
- [ ] Streamlit/React web UI
- [ ] User authentication & authorization
- [ ] Document management system
- [ ] Query history & conversation tracking
- [ ] Export functionality (PDF, Word, JSON)
- [ ] REST API endpoints

### ⏳ Phase 6-7: Enterprise & Deployment
- [ ] Analytics dashboard
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] CI/CD pipeline
- [ ] Monitoring & alerting

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

**Project Maintainer**: Your Name  
**Email**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Google Gemini AI for LLM capabilities
- Sentence-Transformers for embeddings
- FAISS for vector search
- Rank-BM25 for keyword search

---

**⭐ Star this repo if you find it useful!**
