# Neural RAG - Architecture Documentation

## Overview

This application implements a **Retrieval-Augmented Generation (RAG)** system using Amazon Bedrock Knowledge Base for document retrieval and Google Gemini for answer generation.

---

## Document Upload & Sync Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT UPLOAD FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

   USER                    FLASK APP                    AWS SERVICES
    │                          │                              │
    │  1. Upload .pdf/.txt     │                              │
    ├─────────────────────────►│                              │
    │     POST /upload         │                              │
    │                          │                              │
    │                          │  2. upload_to_s3()           │
    │                          ├─────────────────────────────►│ Amazon S3
    │                          │     s3.upload_fileobj()      │ (rag-bedrock-documents)
    │                          │                              │
    │                          │  3. start_ingestion_job()    │
    │                          ├─────────────────────────────►│ Bedrock Agent
    │                          │                              │
    │                          │                              ▼
    │                          │                    ┌─────────────────────┐
    │                          │                    │ INGESTION JOB       │
    │                          │                    │ - Read S3 docs      │
    │                          │                    │ - Chunk text        │
    │                          │                    │ - Titan Embeddings  │
    │                          │                    │ - Store vectors     │
    │                          │                    └─────────┬───────────┘
    │                          │                              │
    │                          │                              ▼
    │  4. Poll /sync/status    │                    ┌─────────────────────┐
    │◄─────────────────────────┤                    │ OpenSearch          │
    │    (every 5 seconds)     │                    │ Serverless          │
    │                          │                    │ (Vector Index)      │
    │                          │                    └─────────────────────┘
```

---

## Question & Answer Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUESTION FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

   USER                    FLASK APP                    AWS SERVICES
    │                          │                              │
    │  1. "What is X?"         │                              │
    ├─────────────────────────►│                              │
    │     POST /ask            │                              │
    │                          │                              │
    │                          │  2. retrieve_from_bedrock()  │
    │                          ├─────────────────────────────►│ Bedrock KB
    │                          │     client.retrieve()        │ (Vector Search)
    │                          │                              │
    │                          │◄─────────────────────────────┤
    │                          │  3. Returns chunks[]         │
    │                          │     (relevant text)          │
    │                          │                              │
    │                          │  4. ask_gemini()             │
    │                          ├─────────────────────────────►│ Google Gemini
    │                          │     "Context: ... Question"  │ (or Bedrock Claude)
    │                          │                              │
    │                          │◄─────────────────────────────┤
    │                          │  5. Generated answer         │
    │                          │                              │
    │◄─────────────────────────┤                              │
    │  6. Display answer       │                              │
    │     + context chunks     │                              │
```

---

## AWS Services Used

| Service | Purpose | Resource |
|---------|---------|----------|
| **Amazon S3** | Document storage | Configured via `S3_BUCKET_NAME` |
| **Amazon Bedrock Knowledge Base** | RAG orchestration & retrieval | Configured via `KNOWLEDGE_BASE_ID` |
| **Amazon OpenSearch Serverless** | Vector database | Auto-provisioned collection |
| **Amazon Titan Embeddings V2** | Text → Vector embeddings | 1024 dimensions |
| **Google Gemini 2.5 Flash** | LLM for answer generation | (or optional Bedrock Claude) |

---

## Key Code Locations

| Step | File | Function | Description |
|------|------|----------|-------------|
| **Upload Endpoint** | `app.py:260` | `POST /upload` | Receives files from frontend |
| **S3 Upload** | `app.py:187` | `upload_to_s3()` | Puts file in `s3://rag-bedrock-documents/documents/` |
| **Start Sync** | `app.py:198` | `start_ingestion_job()` | Tells Bedrock to process new docs |
| **Poll Status** | `app.py:211` | `get_ingestion_status()` | Checks if ingestion is complete |
| **Vector Retrieval** | `app.py:99` | `retrieve_from_bedrock()` | Semantic search for relevant chunks |
| **Answer Generation** | `app.py:156` | `ask_gemini()` | LLM synthesizes answer from chunks |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI page |
| `/upload` | POST | Upload documents to S3 & trigger sync |
| `/sync` | POST | Manually trigger Knowledge Base sync |
| `/sync/status` | GET | Check ingestion job status |
| `/ask` | POST | Ask a question (RAG query) |
| `/health` | GET | Health check & KB status |

---

## Configuration (`.env`)

```env
# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>

# Bedrock Knowledge Base
KNOWLEDGE_BASE_ID=<your-kb-id>
DATA_SOURCE_ID=<your-datasource-id>
S3_BUCKET_NAME=<your-bucket-name>

# LLM (Gemini by default)
GOOGLE_API_KEY=<your-gemini-key>
GEMINI_MODEL=models/gemini-2.5-flash

# Optional: Use Bedrock Claude instead
USE_BEDROCK_LLM=false
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

---

## Supported File Types

- `.txt` - Plain text
- `.pdf` - PDF documents
- `.md` - Markdown
- `.csv` - CSV files
- `.html` - HTML pages
- `.docx` / `.doc` - Word documents

---

## How RAG Works

1. **Document Ingestion**
   - Files uploaded to S3
   - Bedrock ingestion job reads files
   - Text is chunked into smaller pieces
   - Each chunk is converted to a 1024-dimension vector using Titan Embeddings
   - Vectors stored in OpenSearch Serverless

2. **Question Answering**
   - User's question is converted to a vector
   - OpenSearch finds the most similar document chunks (Top-K)
   - Retrieved chunks become the "context"
   - Context + Question sent to Gemini LLM
   - LLM generates a natural language answer

---

## Infrastructure Setup

The `setup_aws_infrastructure.py` script automatically provisions:

1. S3 bucket for documents
2. IAM role for Bedrock
3. OpenSearch Serverless collection
4. Vector index with proper field mappings
5. Bedrock Knowledge Base
6. Data source linking S3 to KB

Run with: `python setup_aws_infrastructure.py`
