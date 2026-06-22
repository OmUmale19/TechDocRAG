# TechDocRAG Usage Guide 🚀

## Quick Start

### 1. Basic Usage (Simplest Way)

```python
from techdocrag.core.config import Config
from techdocrag.core.query_engine import QueryEngine
from techdocrag.core.types import Document

# Initialize the system
config = Config()
query_engine = QueryEngine(config)

# Add a document
document = Document(
    id="my_doc_1",
    title="My Document", 
    content="Your document content here...",
    source="document.pdf"
)

query_engine.add_document(document)

# Ask questions
response = query_engine.query("What is this document about?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence.overall}")
```

### 2. Web API Usage

Create a simple FastAPI server:

```python
from fastapi import FastAPI, UploadFile, File
from techdocrag.core.query_engine import QueryEngine
from techdocrag.core.config import Config

app = FastAPI()
config = Config()
query_engine = QueryEngine(config)

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    # Process and add document
    return {"status": "uploaded"}

@app.post("/ask-question")
async def ask_question(question: str):
    response = query_engine.query(question)
    return {
        "answer": response.answer,
        "confidence": response.confidence.overall,
        "sources": len(response.sources)
    }
```

### 3. Batch Processing

```python
# Process multiple documents at once
documents = [
    Document(id="doc1", title="Doc 1", content="..."),
    Document(id="doc2", title="Doc 2", content="..."),
    Document(id="doc3", title="Doc 3", content="...")
]

for doc in documents:
    query_engine.add_document(doc)

# Ask questions across all documents
response = query_engine.query("Find information about pricing")
```

## Configuration Options

### Vector Store Settings
```python
config.retrieval.semantic_weight = 0.7    # Semantic search weight
config.retrieval.keyword_weight = 0.3     # Keyword search weight  
config.retrieval.top_k = 10               # Number of results
```

### Embedding Settings
```python
config.embedding.model_name = "all-MiniLM-L6-v2"
config.embedding.cache_embeddings = True
config.embedding.batch_size = 32
```

### OCR Settings
```python
config.ocr.provider = "tesseract"
config.ocr.languages = ["eng"]
config.ocr.confidence_threshold = 60
```

## Supported Document Types

- **PDF files** - Extracted using OCR
- **Word documents** - Text extraction
- **Text files** - Direct processing
- **Images** - OCR text extraction
- **Structured data** - JSON, CSV parsing

## Common Use Cases

### 1. Invoice Processing
```python
# Add invoice document
invoice_doc = Document(
    id="invoice_001",
    content="INVOICE #123...",
    metadata={"type": "invoice"}
)

# Ask specific questions
questions = [
    "What is the total amount?",
    "Who is the vendor?", 
    "What is the due date?"
]
```

### 2. Technical Documentation
```python
# Process API documentation
api_doc = Document(
    id="api_docs",
    content="API Documentation...",
    metadata={"type": "technical"}
)

# Ask technical questions
questions = [
    "How do I authenticate?",
    "What are the rate limits?",
    "Show me the endpoint for user data"
]
```

### 3. Research Papers
```python
# Add research paper
paper = Document(
    id="research_001",
    content="Abstract: This paper discusses...",
    metadata={"type": "research", "year": "2025"}
)

# Ask research questions
questions = [
    "What is the main hypothesis?",
    "What methodology was used?",
    "What were the key findings?"
]
```

## Performance Tips

### 1. Optimize for Speed
- Use smaller `top_k` values for faster retrieval
- Cache embeddings for repeated documents
- Batch process multiple documents

### 2. Improve Accuracy  
- Include relevant metadata
- Use descriptive document titles
- Ask specific, well-formed questions

### 3. Memory Management
- Process large documents in chunks
- Clear cache periodically
- Use streaming for very large files

## Error Handling

```python
try:
    response = query_engine.query("Your question")
    if response.confidence.overall > 0.5:
        print(f"High confidence answer: {response.answer}")
    else:
        print("Low confidence - may need more context")
except Exception as e:
    print(f"Error: {e}")
```

## Advanced Features

### Custom Confidence Thresholds
```python
config.confidence.source_weight = 0.3
config.confidence.retrieval_weight = 0.4  
config.confidence.reasoning_weight = 0.3
```

### Hybrid Search Tuning
```python
# Favor semantic search for conceptual questions
config.retrieval.semantic_weight = 0.8
config.retrieval.keyword_weight = 0.2

# Favor keyword search for exact matches
config.retrieval.semantic_weight = 0.3
config.retrieval.keyword_weight = 0.7
```

### Custom Document Types
```python
class InvoiceDocument(Document):
    def __init__(self, invoice_data):
        super().__init__(
            id=invoice_data['invoice_id'],
            title=f"Invoice {invoice_data['invoice_number']}",
            content=self.format_invoice(invoice_data),
            metadata={"type": "invoice", "amount": invoice_data['total']}
        )
```

## Integration Examples

### Streamlit Web App
```python
import streamlit as st
from techdocrag.core.query_engine import QueryEngine

st.title("Document Q&A System")
uploaded_file = st.file_uploader("Upload document")
question = st.text_input("Ask a question")

if question:
    response = query_engine.query(question)
    st.write(f"Answer: {response.answer}")
    st.write(f"Confidence: {response.confidence.overall:.2f}")
```

### Discord Bot
```python
import discord
from techdocrag.core.query_engine import QueryEngine

class DocBot(discord.Client):
    async def on_message(self, message):
        if message.content.startswith('!ask'):
            question = message.content[4:]
            response = query_engine.query(question)
            await message.channel.send(f"Answer: {response.answer}")
```

### REST API with Authentication
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    # Implement your authentication logic
    if not validate_token(token.credentials):
        raise HTTPException(status_code=401)
    return token

@app.post("/query")
async def secure_query(question: str, token: str = Depends(verify_token)):
    response = query_engine.query(question)
    return {"answer": response.answer}
```

## Next Steps

1. **Run the examples**: `python usage_examples.py`
2. **Start with your documents**: Replace sample content with real files
3. **Build an interface**: Web app, API, or desktop application
4. **Customize for your domain**: Adjust settings for your specific use case
5. **Scale up**: Add database persistence, user management, etc.

---

**Need help?** Check the logs in `logs/` directory for detailed debugging information!