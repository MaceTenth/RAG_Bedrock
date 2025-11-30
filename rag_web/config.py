"""
Configuration settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------
# AWS Configuration
# ---------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

# ---------------------------
# LLM Configuration
# ---------------------------
# Gemini (default)
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

# Bedrock Claude (optional)
USE_BEDROCK_LLM = os.environ.get("USE_BEDROCK_LLM", "false").lower() == "true"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# ---------------------------
# App Configuration
# ---------------------------
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.md', '.csv', '.html', '.htm', '.doc', '.docx'}
PORT = int(os.environ.get("PORT", 8001))
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
