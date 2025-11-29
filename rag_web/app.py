import os
import json
import boto3
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Config
# ---------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

# Optional: Use Bedrock Claude instead of Gemini
USE_BEDROCK_LLM = os.environ.get("USE_BEDROCK_LLM", "false").lower() == "true"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.md', '.csv', '.html', '.htm', '.doc', '.docx'}

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# AWS Clients
# ---------------------------
_bedrock_agent_runtime = None
_bedrock_runtime = None
_bedrock_agent = None
_s3_client = None
_gemini_client = None


def get_bedrock_agent_runtime():
    """Get Bedrock Agent Runtime client for retrieval."""
    global _bedrock_agent_runtime
    if _bedrock_agent_runtime is None:
        _bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            region_name=AWS_REGION
        )
    return _bedrock_agent_runtime


def get_bedrock_runtime():
    """Get Bedrock Runtime client for LLM inference."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION
        )
    return _bedrock_runtime


def get_bedrock_agent():
    """Get Bedrock Agent client for ingestion jobs."""
    global _bedrock_agent
    if _bedrock_agent is None:
        _bedrock_agent = boto3.client(
            "bedrock-agent",
            region_name=AWS_REGION
        )
    return _bedrock_agent


def get_s3_client():
    """Get S3 client for document uploads."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def get_gemini_client():
    """Get Gemini client (optional, if not using Bedrock LLM)."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# ---------------------------
# Bedrock RAG Functions
# ---------------------------
def retrieve_from_bedrock(query, k=3, search_type="SEMANTIC", reranking=False, metadata_filter=None):
    """Retrieve relevant documents from Bedrock Knowledge Base."""
    if not KNOWLEDGE_BASE_ID:
        raise RuntimeError("KNOWLEDGE_BASE_ID not configured. Run setup_aws_infrastructure.py first.")
    
    client = get_bedrock_agent_runtime()
    
    # Build vector search configuration
    vector_config = {
        "numberOfResults": k
    }
    
    # Add search type (SEMANTIC or HYBRID)
    if search_type == "HYBRID":
        vector_config["overrideSearchType"] = "HYBRID"
    else:
        vector_config["overrideSearchType"] = "SEMANTIC"
    
    # Add metadata filter if provided
    if metadata_filter:
        vector_config["filter"] = metadata_filter
    
    retrieval_config = {
        "vectorSearchConfiguration": vector_config
    }
    
    # Add reranking if enabled
    if reranking:
        retrieval_config["vectorSearchConfiguration"]["rerankingConfiguration"] = {
            "type": "BEDROCK_RERANKING_MODEL",
            "modelConfiguration": {
                "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.rerank-v1:0"
            }
        }
    
    response = client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration=retrieval_config
    )
    
    results = []
    for result in response.get("retrievalResults", []):
        content = result.get("content", {}).get("text", "")
        if content:
            results.append(content)
    
    return results


def ask_bedrock_llm(context, question, temperature=1.0, top_p=0.95, top_k=40, max_tokens=1024):
    """Use Bedrock Claude to generate an answer."""
    client = get_bedrock_runtime()
    
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    
    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"].strip()


def ask_gemini(context, question, temperature=1.0, top_p=0.95, top_k=40, max_tokens=1024):
    """Use Gemini to generate an answer."""
    from google.genai import types
    
    client = get_gemini_client()
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )
    )
    return (resp.text or "").strip()


def generate_answer(context, question, temperature=1.0, top_p=0.95, top_k=40, max_tokens=1024):
    """Generate answer using configured LLM (Bedrock or Gemini)."""
    if USE_BEDROCK_LLM:
        return ask_bedrock_llm(context, question, temperature, top_p, top_k, max_tokens)
    else:
        return ask_gemini(context, question, temperature, top_p, top_k, max_tokens)


def upload_to_s3(file):
    """Upload a file to S3 bucket."""
    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME not configured.")
    
    s3 = get_s3_client()
    key = f"documents/{file.filename}"
    s3.upload_fileobj(file, S3_BUCKET_NAME, key)
    return key


def start_ingestion_job():
    """Start a Knowledge Base ingestion job to sync documents."""
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        raise RuntimeError("KNOWLEDGE_BASE_ID or DATA_SOURCE_ID not configured.")
    
    client = get_bedrock_agent()
    response = client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID
    )
    return response["ingestionJob"]["ingestionJobId"]


def get_ingestion_status():
    """Get the status of the latest ingestion job."""
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        return None
    
    client = get_bedrock_agent()
    response = client.list_ingestion_jobs(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID,
        maxResults=1
    )
    
    jobs = response.get("ingestionJobSummaries", [])
    if jobs:
        return jobs[0]
    return None


def get_ingestion_job_by_id(job_id):
    """Get the status of a specific ingestion job by ID."""
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        return None
    
    client = get_bedrock_agent()
    try:
        response = client.get_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID,
            ingestionJobId=job_id
        )
        return response.get("ingestionJob", {})
    except Exception:
        return None

# ---------------------------
# Pages
# ---------------------------
@app.get("/")
def home():
    return render_template("index.html")


# ---------------------------
# APIs
# ---------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    # Count documents in S3
    doc_count = 0
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="")
        doc_count = response.get("KeyCount", 0)
    except Exception:
        pass
    
    status = {
        "status": "healthy",
        "document_count": doc_count,
        "knowledge_base_configured": KNOWLEDGE_BASE_ID is not None,
        "s3_bucket_configured": S3_BUCKET_NAME is not None,
        "llm": "bedrock" if USE_BEDROCK_LLM else "gemini"
    }
    
    # Check latest ingestion status
    ingestion = get_ingestion_status()
    if ingestion:
        status["last_ingestion"] = {
            "status": ingestion.get("status"),
            "started_at": str(ingestion.get("startedAt", "")),
        }
    
    return jsonify(status)


@app.post("/upload")
def upload():
    """Upload documents to S3 and trigger ingestion."""
    if "files" not in request.files:
        return jsonify({"error": "Send file(s) under 'files' form field."}), 400
    
    files = request.files.getlist("files")
    uploaded = []
    
    for f in files:
        if f.filename:
            ext = os.path.splitext(f.filename.lower())[1]
            if ext in ALLOWED_EXTENSIONS:
                try:
                    key = upload_to_s3(f)
                    uploaded.append(f.filename)  # Return original filename for display
                except Exception as e:
                    return jsonify({"error": f"Failed to upload {f.filename}: {str(e)}"}), 500
    
    if not uploaded:
        return jsonify({"error": f"No valid files uploaded. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Start ingestion job to sync new documents
    try:
        job_id = start_ingestion_job()
        return jsonify({
            "uploaded": uploaded,
            "ingestion_job_id": job_id,
            "message": "Documents uploaded. Ingestion started - this may take a few minutes."
        })
    except Exception as e:
        return jsonify({
            "uploaded": uploaded,
            "warning": f"Documents uploaded but ingestion failed: {str(e)}"
        })


@app.post("/sync")
def sync():
    """Manually trigger a Knowledge Base sync."""
    try:
        job_id = start_ingestion_job()
        return jsonify({
            "ingestion_job_id": job_id,
            "message": "Ingestion job started."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/ingestion-status/<job_id>")
def ingestion_status(job_id):
    """Get the status of a specific ingestion job."""
    job = get_ingestion_job_by_id(job_id)
    if job:
        return jsonify({
            "status": job.get("status"),
            "started_at": str(job.get("startedAt", "")),
            "updated_at": str(job.get("updatedAt", "")),
            "failure_reasons": job.get("failureReasons", [])
        })
    return jsonify({"status": "not_found", "error": "Job not found"}), 404


@app.get("/sync/status")
def sync_status():
    """Get the status of the latest ingestion job."""
    ingestion = get_ingestion_status()
    if ingestion:
        return jsonify({
            "status": ingestion.get("status"),
            "started_at": str(ingestion.get("startedAt", "")),
            "updated_at": str(ingestion.get("updatedAt", ""))
        })
    return jsonify({"status": "no_jobs_found"})


@app.post("/ask")
def ask():
    """Answer a question using RAG."""
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    
    # Retrieval parameters
    k = int(data.get("top_k", 4))
    search_type = data.get("search_type", "SEMANTIC")
    reranking = data.get("reranking", False)
    metadata_filter = data.get("metadata_filter", None)
    
    # LLM generation parameters
    temperature = float(data.get("temperature", 1.0))
    top_p = float(data.get("top_p", 0.95))
    llm_top_k = int(data.get("llm_top_k", 40))
    max_tokens = int(data.get("max_tokens", 1024))
    
    if not question:
        return jsonify({"error": "Provide 'question' in JSON body."}), 400
    
    try:
        # Retrieve relevant chunks from Bedrock Knowledge Base
        chunks = retrieve_from_bedrock(
            question, 
            k=k, 
            search_type=search_type, 
            reranking=reranking, 
            metadata_filter=metadata_filter
        )
        
        if not chunks:
            return jsonify({
                "question": question,
                "error": "No relevant documents found. Upload documents and sync first."
            }), 404
        
        # Generate answer using LLM
        context = "\n".join(chunks)
        answer = generate_answer(
            context, 
            question, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=llm_top_k, 
            max_tokens=max_tokens
        )
        
        return jsonify({
            "question": question,
            "top_k": k,
            "search_type": search_type,
            "reranking": reranking,
            "context": chunks,
            "answer": answer
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Start
# ---------------------------
if __name__ == "__main__":
    print(f"🚀 Starting RAG application")
    print(f"   Knowledge Base ID: {KNOWLEDGE_BASE_ID or 'Not configured'}")
    print(f"   S3 Bucket: {S3_BUCKET_NAME or 'Not configured'}")
    print(f"   LLM: {'Bedrock ' + BEDROCK_MODEL_ID if USE_BEDROCK_LLM else 'Gemini'}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8001)), debug=True)