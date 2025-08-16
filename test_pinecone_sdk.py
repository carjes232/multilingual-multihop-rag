#!/usr/bin/env python3
"""
Test Pinecone SDK to verify integrated inference functionality.
"""

import os
import dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
dotenv.load_dotenv()

def test_pinecone_index_creation():
    """Test Pinecone index creation with source_model."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not set")
        return False
        
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Delete existing test index if it exists
        index_name = "test-integrated-inference"
        try:
            pc.delete_index(index_name)
            print(f"Deleted existing index '{index_name}'")
        except Exception:
            pass  # Index doesn't exist, which is fine
            
        # Create index with source_model
        print("Creating index with source_model...")
        pc.create_index(
            name=index_name,
            source_model="multilingual-e5-large",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully!")
        
        # Describe the index to verify it has source_model
        print("Describing index...")
        desc = pc.describe_index(index_name)
        print(f"Index description: {desc}")
        
        # Check if source_model is present
        source_model = None
        if hasattr(desc, 'source_model'):
            source_model = desc.source_model
        elif hasattr(desc, 'sourceModel'):
            source_model = desc.sourceModel
        elif isinstance(desc, dict):
            source_model = desc.get('source_model') or desc.get('sourceModel')
            
        print(f"Source model: {source_model}")
        
        if source_model:
            print("SUCCESS: Index created with source_model")
            # Clean up
            pc.delete_index(index_name)
            print(f"Cleaned up test index '{index_name}'")
            return True
        else:
            print("FAILURE: Index created without source_model")
            # Clean up
            pc.delete_index(index_name)
            print(f"Cleaned up test index '{index_name}'")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pinecone_index_creation()
    if success:
        print("Pinecone SDK test PASSED")
    else:
        print("Pinecone SDK test FAILED")