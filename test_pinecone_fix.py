#!/usr/bin/env python3
"""
Simple test script to verify Pinecone integrated inference fix.
"""

import os
import dotenv
from pinecone import Pinecone

# Load environment variables
dotenv.load_dotenv()

def test_pinecone_index():
    """Test Pinecone index configuration and upsert."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not set")
        return False
        
    index_name = os.getenv("PINECONE_INDEX", "rag")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Describe the index
        print(f"Describing index: {index_name}")
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
        
        # Test upsert with a simple record
        index = pc.Index(index_name)
        
        if hasattr(index, "upsert_records"):
            print("Testing upsert_records with integrated inference...")
            test_records = [
                {
                    "id": "test-record-1",
                    "text": "This is a test document for Pinecone integrated inference.",
                    "metadata": {
                        "title": "Test Document",
                        "doc_id": "test-doc-1"
                    }
                }
            ]
            
            namespace = os.getenv("PINECONE_NAMESPACE", None)
            print(f"Using namespace: {namespace}")
            
            # Try upsert with namespace handling
            try:
                if namespace:
                    result = index.upsert_records(namespace=namespace, records=test_records)
                else:
                    result = index.upsert_records(records=test_records)
                print("Upsert successful!")
                print(f"Result: {result}")
                return True
            except Exception as e:
                print(f"Upsert failed: {e}")
                return False
        else:
            print("upsert_records method not available")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pinecone_index()
    if success:
        print("Pinecone integration test PASSED")
    else:
        print("Pinecone integration test FAILED")