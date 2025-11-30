"""
Bedrock Knowledge Base retrieval service.
"""
from config import AWS_REGION, KNOWLEDGE_BASE_ID
from .aws_clients import get_bedrock_agent_runtime


def retrieve_from_bedrock(query, k=3, search_type="SEMANTIC", reranking=False, metadata_filter=None):
    """
    Retrieve relevant documents from Bedrock Knowledge Base.
    
    Args:
        query: The search query text
        k: Number of results to return (1-100)
        search_type: "SEMANTIC" or "HYBRID"
        reranking: Whether to use Amazon Rerank model
        metadata_filter: Optional filter dict for document metadata
    
    Returns:
        List of text chunks from matching documents
    """
    if not KNOWLEDGE_BASE_ID:
        raise RuntimeError("KNOWLEDGE_BASE_ID not configured. Run setup_aws_infrastructure.py first.")
    
    client = get_bedrock_agent_runtime()
    
    # Build vector search configuration
    vector_config = {
        "numberOfResults": k,
        "overrideSearchType": search_type
    }
    
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
