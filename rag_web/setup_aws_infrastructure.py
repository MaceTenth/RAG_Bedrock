#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script for RAG with Bedrock Knowledge Base

This script provisions:
1. S3 bucket for document storage
2. OpenSearch Serverless collection (vector store)
3. IAM roles and policies
4. Bedrock Knowledge Base
5. Data source linking S3 to Knowledge Base

Usage:
    python setup_aws_infrastructure.py

Prerequisites:
    - AWS credentials in .env file
    - boto3 installed
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path

# Load environment variables from .env file
def load_env(env_path=".env"):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"❌ Error: {env_path} not found.")
        print(f"   Please copy .env.example to .env and fill in your credentials.")
        sys.exit(1)
    
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
    
    print(f"✅ Loaded environment from {env_path}")

# Load .env first
load_env()

import boto3
from botocore.exceptions import ClientError

# Configuration from environment
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", f"rag-bedrock-docs-{uuid.uuid4().hex[:8]}")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag-vector-collection")
KNOWLEDGE_BASE_NAME = os.environ.get("KNOWLEDGE_BASE_NAME", "rag-knowledge-base")

# AWS Clients
session = boto3.Session(
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

s3_client = session.client("s3")
iam_client = session.client("iam")
oss_client = session.client("opensearchserverless")
bedrock_agent_client = session.client("bedrock-agent")
sts_client = session.client("sts")

# Get AWS Account ID
ACCOUNT_ID = sts_client.get_caller_identity()["Account"]
print(f"📋 AWS Account ID: {ACCOUNT_ID}")
print(f"📋 Region: {AWS_REGION}")


def create_s3_bucket():
    """Create S3 bucket for document storage."""
    print(f"\n🪣 Creating S3 bucket: {S3_BUCKET_NAME}")
    
    try:
        if AWS_REGION == "us-east-1":
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )
        print(f"   ✅ Bucket created: {S3_BUCKET_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            print(f"   ℹ️  Bucket already exists: {S3_BUCKET_NAME}")
        else:
            raise e
    
    # Create a folder for documents
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key="documents/")
    print(f"   ✅ Created documents/ folder")
    
    return S3_BUCKET_NAME


def create_bedrock_execution_role():
    """Create IAM role for Bedrock Knowledge Base."""
    role_name = "BedrockKnowledgeBaseRole"
    print(f"\n🔐 Creating IAM role: {role_name}")
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:knowledge-base/*"
                    }
                }
            }
        ]
    }
    
    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for Bedrock Knowledge Base to access S3 and OpenSearch"
        )
        role_arn = response["Role"]["Arn"]
        print(f"   ✅ Role created: {role_arn}")
        
        # Wait for role to propagate
        time.sleep(10)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"
            print(f"   ℹ️  Role already exists: {role_arn}")
        else:
            raise e
    
    # Attach policies for S3, Bedrock, and OpenSearch access
    policies = [
        {
            "name": "BedrockKnowledgeBaseS3Policy",
            "document": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            f"arn:aws:s3:::{S3_BUCKET_NAME}",
                            f"arn:aws:s3:::{S3_BUCKET_NAME}/*"
                        ]
                    }
                ]
            }
        },
        {
            "name": "BedrockKnowledgeBaseFoundationModelPolicy",
            "document": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["bedrock:InvokeModel"],
                        "Resource": [
                            f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v1",
                            f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0",
                            f"arn:aws:bedrock:{AWS_REGION}::foundation-model/cohere.embed-english-v3",
                            f"arn:aws:bedrock:{AWS_REGION}::foundation-model/cohere.embed-multilingual-v3"
                        ]
                    }
                ]
            }
        },
        {
            "name": "BedrockKnowledgeBaseOSSPolicy",
            "document": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["aoss:APIAccessAll"],
                        "Resource": [
                            f"arn:aws:aoss:{AWS_REGION}:{ACCOUNT_ID}:collection/*"
                        ]
                    }
                ]
            }
        }
    ]
    
    for policy in policies:
        try:
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=policy["name"],
                PolicyDocument=json.dumps(policy["document"])
            )
            print(f"   ✅ Attached policy: {policy['name']}")
        except ClientError as e:
            print(f"   ⚠️  Policy error: {e}")
    
    return f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"


def create_opensearch_serverless_collection(role_arn):
    """Create OpenSearch Serverless collection for vector storage."""
    print(f"\n🔍 Creating OpenSearch Serverless collection: {COLLECTION_NAME}")
    
    # 1. Create encryption policy
    encryption_policy_name = "rag-vec-enc-policy"
    encryption_policy = {
        "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}],
        "AWSOwnedKey": True
    }
    
    try:
        oss_client.create_security_policy(
            name=encryption_policy_name,
            type="encryption",
            policy=json.dumps(encryption_policy),
            description="Encryption policy for RAG vector collection"
        )
        print(f"   ✅ Encryption policy created")
    except ClientError as e:
        if "ConflictException" in str(e):
            print(f"   ℹ️  Encryption policy already exists")
        else:
            raise e
    
    # 2. Create network policy
    network_policy_name = "rag-vec-net-policy"
    network_policy = [
        {
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AllowFromPublic": True
        },
        {
            "Rules": [{"ResourceType": "dashboard", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AllowFromPublic": True
        }
    ]
    
    try:
        oss_client.create_security_policy(
            name=network_policy_name,
            type="network",
            policy=json.dumps(network_policy),
            description="Network policy for RAG vector collection"
        )
        print(f"   ✅ Network policy created")
    except ClientError as e:
        if "ConflictException" in str(e):
            print(f"   ℹ️  Network policy already exists")
        else:
            raise e
    
    # 3. Create data access policy
    data_access_policy_name = "rag-vec-data-policy"
    data_access_policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{COLLECTION_NAME}"],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:UpdateCollectionItems",
                        "aoss:DescribeCollectionItems",
                        "aoss:DeleteCollectionItems"
                    ]
                },
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{COLLECTION_NAME}/*"],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:DeleteIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument"
                    ]
                }
            ],
            "Principal": [
                role_arn,
                f"arn:aws:iam::{ACCOUNT_ID}:root"
            ],
            "Description": "Data access policy for RAG"
        }
    ]
    
    try:
        oss_client.create_access_policy(
            name=data_access_policy_name,
            type="data",
            policy=json.dumps(data_access_policy),
            description="Data access policy for RAG vector collection"
        )
        print(f"   ✅ Data access policy created")
    except ClientError as e:
        if "ConflictException" in str(e):
            print(f"   ℹ️  Data access policy already exists")
        else:
            raise e
    
    # 4. Create the collection
    try:
        response = oss_client.create_collection(
            name=COLLECTION_NAME,
            type="VECTORSEARCH",
            description="Vector collection for RAG application"
        )
        collection_id = response["createCollectionDetail"]["id"]
        print(f"   ✅ Collection created: {collection_id}")
    except ClientError as e:
        if "ConflictException" in str(e):
            # Get existing collection
            response = oss_client.batch_get_collection(names=[COLLECTION_NAME])
            collection_id = response["collectionDetails"][0]["id"]
            print(f"   ℹ️  Collection already exists: {collection_id}")
        else:
            raise e
    
    # 5. Wait for collection to be active
    print(f"   ⏳ Waiting for collection to become active...")
    while True:
        response = oss_client.batch_get_collection(ids=[collection_id])
        status = response["collectionDetails"][0]["status"]
        if status == "ACTIVE":
            collection_endpoint = response["collectionDetails"][0]["collectionEndpoint"]
            print(f"   ✅ Collection is ACTIVE")
            print(f"   📍 Endpoint: {collection_endpoint}")
            break
        elif status == "FAILED":
            raise Exception("Collection creation failed!")
        print(f"      Status: {status}...")
        time.sleep(30)
    
    return collection_id, collection_endpoint


def create_vector_index(collection_endpoint, force_recreate=False):
    """Create the vector index in OpenSearch Serverless collection."""
    print(f"\n📊 Creating vector index in OpenSearch collection...")
    
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth
    
    # Get credentials for signing requests
    credentials = session.get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        AWS_REGION,
        "aoss",
        session_token=credentials.token
    )
    
    # Parse the endpoint (remove https://)
    host = collection_endpoint.replace("https://", "")
    
    # Create OpenSearch client
    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    
    index_name = "bedrock-knowledge-base-index"
    
    # Check if index exists and has correct mappings
    try:
        if client.indices.exists(index=index_name):
            if force_recreate:
                print(f"   🗑️  Deleting existing index to recreate with correct mappings...")
                client.indices.delete(index=index_name)
                print(f"   ✅ Old index deleted")
                time.sleep(5)
            else:
                # Check if mappings are correct
                mappings = client.indices.get_mapping(index=index_name)
                props = mappings.get(index_name, {}).get("mappings", {}).get("properties", {})
                if "vector" in props and "text" in props:
                    print(f"   ℹ️  Index already exists with correct mappings: {index_name}")
                    return index_name
                else:
                    print(f"   ⚠️  Index exists but has wrong mappings. Recreating...")
                    client.indices.delete(index=index_name)
                    print(f"   ✅ Old index deleted")
                    time.sleep(5)
    except Exception as e:
        print(f"   ⚠️  Could not check index: {e}")
    
    # Create the vector index with proper mappings for Bedrock
    # Using field names that match AWS sample code
    index_body = {
        "settings": {
            "index.knn": "true",
            "number_of_shards": 1,
            "knn.algo_param.ef_search": 512,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,  # Titan Embeddings v2 dimension
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "l2"
                    }
                },
                "text": {
                    "type": "text"
                },
                "text-metadata": {
                    "type": "text"
                }
            }
        }
    }
    
    try:
        response = client.indices.create(index=index_name, body=index_body)
        print(f"   ✅ Vector index created: {index_name}")
        # Wait for index to be fully available
        print(f"   ⏳ Waiting 60 seconds for index to be available...")
        time.sleep(60)
        return index_name
    except Exception as e:
        if "resource_already_exists_exception" in str(e).lower():
            print(f"   ℹ️  Index already exists: {index_name}")
            return index_name
        else:
            raise e


def create_bedrock_knowledge_base(role_arn, collection_arn):
    """Create Bedrock Knowledge Base."""
    print(f"\n🧠 Creating Bedrock Knowledge Base: {KNOWLEDGE_BASE_NAME}")
    
    try:
        response = bedrock_agent_client.create_knowledge_base(
            name=KNOWLEDGE_BASE_NAME,
            description="RAG Knowledge Base for document Q&A",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0"
                }
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": "bedrock-knowledge-base-index",
                    "fieldMapping": {
                        "vectorField": "vector",
                        "textField": "text",
                        "metadataField": "text-metadata"
                    }
                }
            }
        )
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        print(f"   ✅ Knowledge Base created: {kb_id}")
    except ClientError as e:
        if "ConflictException" in str(e):
            # List existing knowledge bases to find ours
            response = bedrock_agent_client.list_knowledge_bases()
            for kb in response["knowledgeBaseSummaries"]:
                if kb["name"] == KNOWLEDGE_BASE_NAME:
                    kb_id = kb["knowledgeBaseId"]
                    print(f"   ℹ️  Knowledge Base already exists: {kb_id}")
                    break
            else:
                raise Exception("Could not find existing knowledge base")
        else:
            raise e
    
    # Wait for KB to be active
    print(f"   ⏳ Waiting for Knowledge Base to become active...")
    while True:
        response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
        status = response["knowledgeBase"]["status"]
        if status == "ACTIVE":
            print(f"   ✅ Knowledge Base is ACTIVE")
            break
        elif status == "FAILED":
            failure_reasons = response["knowledgeBase"].get("failureReasons", [])
            raise Exception(f"Knowledge Base creation failed: {failure_reasons}")
        print(f"      Status: {status}...")
        time.sleep(10)
    
    return kb_id


def create_data_source(kb_id):
    """Create S3 data source for the Knowledge Base."""
    data_source_name = "s3-documents-source"
    print(f"\n📂 Creating Data Source: {data_source_name}")
    
    try:
        response = bedrock_agent_client.create_data_source(
            knowledgeBaseId=kb_id,
            name=data_source_name,
            description="S3 bucket containing documents for RAG",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{S3_BUCKET_NAME}",
                    "inclusionPrefixes": ["documents/"]
                }
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {
                    "chunkingStrategy": "FIXED_SIZE",
                    "fixedSizeChunkingConfiguration": {
                        "maxTokens": 512,
                        "overlapPercentage": 20
                    }
                }
            }
        )
        ds_id = response["dataSource"]["dataSourceId"]
        print(f"   ✅ Data Source created: {ds_id}")
    except ClientError as e:
        if "ConflictException" in str(e):
            # List existing data sources
            response = bedrock_agent_client.list_data_sources(knowledgeBaseId=kb_id)
            for ds in response["dataSourceSummaries"]:
                if ds["name"] == data_source_name:
                    ds_id = ds["dataSourceId"]
                    print(f"   ℹ️  Data Source already exists: {ds_id}")
                    break
            else:
                raise Exception("Could not find existing data source")
        else:
            raise e
    
    return ds_id


def update_env_file(kb_id, ds_id):
    """Update .env file with created resource IDs."""
    print(f"\n📝 Updating .env file with resource IDs")
    
    env_path = Path(".env")
    if not env_path.exists():
        # Copy from example
        with open(".env.example", "r") as src:
            content = src.read()
        with open(".env", "w") as dst:
            dst.write(content)
    
    # Read current .env
    with open(".env", "r") as f:
        lines = f.readlines()
    
    # Update values
    updated_lines = []
    for line in lines:
        if line.startswith("KNOWLEDGE_BASE_ID="):
            updated_lines.append(f"KNOWLEDGE_BASE_ID={kb_id}\n")
        elif line.startswith("DATA_SOURCE_ID="):
            updated_lines.append(f"DATA_SOURCE_ID={ds_id}\n")
        elif line.startswith("S3_BUCKET_NAME="):
            updated_lines.append(f"S3_BUCKET_NAME={S3_BUCKET_NAME}\n")
        else:
            updated_lines.append(line)
    
    # Write updated .env
    with open(".env", "w") as f:
        f.writelines(updated_lines)
    
    print(f"   ✅ Updated .env with:")
    print(f"      KNOWLEDGE_BASE_ID={kb_id}")
    print(f"      DATA_SOURCE_ID={ds_id}")
    print(f"      S3_BUCKET_NAME={S3_BUCKET_NAME}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 AWS Infrastructure Setup for RAG with Bedrock")
    print("=" * 60)
    
    try:
        # Step 1: Create S3 bucket
        bucket_name = create_s3_bucket()
        
        # Step 2: Create IAM role
        role_arn = create_bedrock_execution_role()
        
        # Step 3: Create OpenSearch Serverless collection
        collection_id, collection_endpoint = create_opensearch_serverless_collection(role_arn)
        collection_arn = f"arn:aws:aoss:{AWS_REGION}:{ACCOUNT_ID}:collection/{collection_id}"
        
        # Step 4: Create vector index in the collection
        create_vector_index(collection_endpoint)
        
        # Step 5: Create Bedrock Knowledge Base
        kb_id = create_bedrock_knowledge_base(role_arn, collection_arn)
        
        # Step 6: Create Data Source
        ds_id = create_data_source(kb_id)
        
        # Step 7: Update .env file
        update_env_file(kb_id, ds_id)
        
        print("\n" + "=" * 60)
        print("✅ Setup Complete!")
        print("=" * 60)
        print(f"""
📋 Resources Created:
   • S3 Bucket: {bucket_name}
   • OpenSearch Collection: {COLLECTION_NAME} ({collection_id})
   • Knowledge Base ID: {kb_id}
   • Data Source ID: {ds_id}

📝 Next Steps:
   1. Upload documents to S3:
      aws s3 cp your-document.txt s3://{bucket_name}/documents/

   2. Sync the Knowledge Base:
      aws bedrock-agent start-ingestion-job \\
          --knowledge-base-id {kb_id} \\
          --data-source-id {ds_id}

   3. Update your app.py to use Bedrock retrieval
   
   4. Run your Flask app:
      python app.py
""")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
