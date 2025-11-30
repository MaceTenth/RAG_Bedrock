"""
LLM service for generating answers using Bedrock Claude or Gemini.
"""
import json
from config import USE_BEDROCK_LLM, BEDROCK_MODEL_ID, GEMINI_MODEL
from .aws_clients import get_bedrock_runtime, get_gemini_client


def _build_prompt(context, question):
    """Build the RAG prompt."""
    return f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""


def ask_bedrock_llm(context, question, temperature=1.0, top_p=0.95, top_k=40, max_tokens=1024):
    """
    Use Bedrock Claude to generate an answer.
    
    Args:
        context: Retrieved document context
        question: User's question
        temperature: Randomness (0-2)
        top_p: Nucleus sampling threshold
        top_k: Limit token choices
        max_tokens: Maximum response length
    
    Returns:
        Generated answer text
    """
    client = get_bedrock_runtime()
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "messages": [
            {"role": "user", "content": _build_prompt(context, question)}
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
    """
    Use Gemini to generate an answer.
    
    Args:
        context: Retrieved document context
        question: User's question
        temperature: Randomness (0-2)
        top_p: Nucleus sampling threshold
        top_k: Limit token choices
        max_tokens: Maximum response length
    
    Returns:
        Generated answer text
    """
    from google.genai import types
    
    client = get_gemini_client()
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(context, question),
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )
    )
    return (resp.text or "").strip()


def generate_answer(context, question, temperature=1.0, top_p=0.95, top_k=40, max_tokens=1024):
    """
    Generate answer using configured LLM (Bedrock or Gemini).
    
    Uses USE_BEDROCK_LLM config to determine which LLM to use.
    """
    if USE_BEDROCK_LLM:
        return ask_bedrock_llm(context, question, temperature, top_p, top_k, max_tokens)
    else:
        return ask_gemini(context, question, temperature, top_p, top_k, max_tokens)
