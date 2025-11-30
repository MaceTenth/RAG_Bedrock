# Neural RAG - Retrieval-Augmented Generation with Amazon Bedrock

A Flask-based RAG (Retrieval-Augmented Generation) application that uses **Amazon Bedrock Knowledge Base** for document retrieval and **Google Gemini** or **Amazon Bedrock Claude** for LLM-powered answer generation.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)

## Features

- 📄 **Document Upload** - Upload PDF, TXT, MD, CSV, and other text files
- 🔍 **Semantic Search** - Vector-based retrieval using Amazon Titan Embeddings
- 🔄 **Hybrid Search** - Combine vector + keyword search for better results
- 🤖 **Flexible LLM** - Use Google Gemini or Amazon Bedrock Claude for answer generation
- ⚡ **Real-time Sync** - Automatic document ingestion with status tracking
- 🎛️ **Advanced Settings** - Configure temperature, top_k, reranking, and more

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│   Frontend   │────►│  Flask API   │────►│  Amazon Bedrock KB      │
│   (HTML/JS)  │     │              │     │  (Vector Search)        │
└──────────────┘     └──────┬───────┘     └───────────┬─────────────┘
                            │                         │
                            ▼                         ▼
                     ┌──────────────┐     ┌─────────────────────────┐
                     │  Gemini or   │     │  Amazon OpenSearch      │
                     │  Bedrock LLM │     │  Serverless (Vectors)   │
                     └──────────────┘     └─────────────────────────┘
```

## Prerequisites

- Python 3.12+
- AWS Account with Bedrock access enabled
- Google Cloud account (for Gemini API) OR Bedrock Claude access
- AWS CLI configured (optional, for setup script)

## Quick Start

### 1. Clone and Setup

```bash
cd RAG_Bedrock/rag_web
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Bedrock Knowledge Base (populated by setup script or manually)
KNOWLEDGE_BASE_ID=your_knowledge_base_id
DATA_SOURCE_ID=your_data_source_id
S3_BUCKET_NAME=your_s3_bucket_name

# LLM Configuration
USE_BEDROCK_LLM=true  # Set to 'false' to use Gemini instead
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Gemini (required if USE_BEDROCK_LLM=false)
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=models/gemini-2.5-flash
```

### 3. Setup AWS Infrastructure (First Time Only)

Run the setup script to create required AWS resources:

```bash
python setup_aws_infrastructure.py
```

This creates:
- S3 bucket for document storage
- OpenSearch Serverless collection (vector store)
- IAM roles and policies
- Bedrock Knowledge Base with data source

**Note:** After running, update your `.env` with the generated `KNOWLEDGE_BASE_ID`, `DATA_SOURCE_ID`, and `S3_BUCKET_NAME`.

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:8000`

## Usage

### Web Interface

1. Open `http://localhost:8000` in your browser
2. **Upload Documents** - Use the upload section to add your documents
3. **Wait for Sync** - Documents are automatically indexed (check sync status)
4. **Ask Questions** - Type your question and get AI-powered answers

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/health` | GET | Health check and status |
| `/upload` | POST | Upload documents (multipart form) |
| `/sync` | POST | Manually trigger document sync |
| `/sync/status` | GET | Check sync job status |
| `/ingestion-status/<job_id>` | GET | Check specific job status |
| `/ask` | POST | Ask a question (JSON body) |

### Example API Usage

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "top_k": 4,
    "search_type": "SEMANTIC",
    "temperature": 1.0
  }'
```

**Upload a document:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf"
```

## Configuration Options

### Retrieval Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 4 | Number of document chunks to retrieve |
| `search_type` | SEMANTIC | Search mode: `SEMANTIC` or `HYBRID` |
| `reranking` | false | Use Amazon Rerank model for better results |
| `metadata_filter` | null | Filter documents by metadata |

### LLM Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Response randomness (0-2) |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `llm_top_k` | 40 | Token choice limit |
| `max_tokens` | 1024 | Maximum response length |

## Docker Deployment

```bash
# Build the image
docker build -t rag-bedrock .

# Run with environment file
docker run -p 8000:8000 --env-file .env rag-bedrock
```

## Project Structure

```
rag_web/
├── app.py                 # Main Flask application entry point
├── config.py              # Environment configuration
├── setup_aws_infrastructure.py  # AWS resource provisioning
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── routes/
│   └── api.py            # API route handlers
├── services/
│   ├── aws_clients.py    # AWS & Gemini client singletons
│   ├── ingestion.py      # S3 upload & sync logic
│   ├── retrieval.py      # Bedrock KB retrieval
│   └── llm.py            # LLM generation (Gemini/Claude)
├── templates/
│   ├── base.html         # Base template
│   └── index.html        # Main UI
└── static/
    └── css/main.css      # Styles
```

## Supported File Types

- `.txt` - Plain text
- `.pdf` - PDF documents
- `.md` - Markdown files
- `.csv` - CSV data
- `.html` / `.htm` - HTML pages
- `.doc` / `.docx` - Word documents

## Troubleshooting

### Common Issues

**"KNOWLEDGE_BASE_ID not configured"**
- Run `setup_aws_infrastructure.py` or manually set `KNOWLEDGE_BASE_ID` in `.env`

**"Missing GOOGLE_API_KEY"**
- If using Gemini, add your API key to `.env`
- Or set `USE_BEDROCK_LLM=true` to use Bedrock Claude instead

**Documents not appearing in search**
- Check `/sync/status` to ensure ingestion completed
- Ingestion may take 1-5 minutes for new documents

**AWS credential errors**
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct
- Ensure your IAM user has Bedrock, S3, and OpenSearch permissions

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
