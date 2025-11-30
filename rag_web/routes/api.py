"""
API routes for the RAG application.
"""
import os
from flask import Blueprint, request, jsonify
from config import KNOWLEDGE_BASE_ID, S3_BUCKET_NAME, USE_BEDROCK_LLM, ALLOWED_EXTENSIONS
from services import (
    retrieve_from_bedrock,
    generate_answer,
    upload_to_s3,
    start_ingestion_job,
    get_ingestion_status,
    get_ingestion_job_by_id,
)
from services.ingestion import get_document_count

api = Blueprint('api', __name__)


@api.get("/health")
def health():
    """Health check endpoint."""
    doc_count = get_document_count()
    
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


@api.post("/upload")
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
                    upload_to_s3(f)
                    uploaded.append(f.filename)
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


@api.post("/sync")
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


@api.get("/ingestion-status/<job_id>")
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


@api.get("/sync/status")
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


@api.post("/ask")
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
