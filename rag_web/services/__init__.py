"""
Services package initialization.
"""
from .aws_clients import (
    get_bedrock_agent_runtime,
    get_bedrock_runtime,
    get_bedrock_agent,
    get_s3_client,
    get_gemini_client
)
from .retrieval import retrieve_from_bedrock
from .llm import generate_answer
from .ingestion import (
    upload_to_s3,
    start_ingestion_job,
    get_ingestion_status,
    get_ingestion_job_by_id,
    get_document_count
)

__all__ = [
    # AWS Clients
    'get_bedrock_agent_runtime',
    'get_bedrock_runtime',
    'get_bedrock_agent',
    'get_s3_client',
    'get_gemini_client',
    # Retrieval
    'retrieve_from_bedrock',
    # LLM
    'generate_answer',
    # Ingestion
    'upload_to_s3',
    'start_ingestion_job',
    'get_ingestion_status',
    'get_ingestion_job_by_id',
    'get_document_count',
]
