"""
AWS client singletons for boto3 and Gemini.
"""
import boto3
from config import AWS_REGION, GEMINI_API_KEY

# Singleton clients
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
