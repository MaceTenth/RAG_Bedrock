"""
Document ingestion service for S3 uploads and Knowledge Base syncing.
"""
from config import KNOWLEDGE_BASE_ID, DATA_SOURCE_ID, S3_BUCKET_NAME
from .aws_clients import get_s3_client, get_bedrock_agent


def upload_to_s3(file):
    """
    Upload a file to S3 bucket.
    
    Args:
        file: Flask file object from request.files
    
    Returns:
        The S3 object key
    """
    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME not configured.")
    
    s3 = get_s3_client()
    key = f"documents/{file.filename}"
    s3.upload_fileobj(file, S3_BUCKET_NAME, key)
    return key


def start_ingestion_job():
    """
    Start a Knowledge Base ingestion job to sync documents.
    
    Returns:
        The ingestion job ID
    """
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        raise RuntimeError("KNOWLEDGE_BASE_ID or DATA_SOURCE_ID not configured.")
    
    client = get_bedrock_agent()
    response = client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID
    )
    return response["ingestionJob"]["ingestionJobId"]


def get_ingestion_status():
    """
    Get the status of the latest ingestion job.
    
    Returns:
        Dict with job status or None if no jobs found
    """
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
    """
    Get the status of a specific ingestion job by ID.
    
    Args:
        job_id: The ingestion job ID
    
    Returns:
        Dict with job details or None if not found
    """
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


def get_document_count():
    """
    Count documents in the S3 bucket's documents/ folder.
    
    Returns:
        Number of documents or 0 on error
    """
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="documents/")
        
        # Filter out folder markers (keys ending with /) and count only actual files
        if "Contents" in response:
            count = sum(1 for obj in response["Contents"] if not obj["Key"].endswith("/"))
            return count
        return 0
    except Exception:
        return 0
