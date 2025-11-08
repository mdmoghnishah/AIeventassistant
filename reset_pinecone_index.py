import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def reset_pinecone_index():
    """Delete and recreate Pinecone index with correct dimensions"""
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    print(f"🔍 Checking index: {index_name}")
    
    # Check if index exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"⚠️  Index '{index_name}' exists. Deleting...")
        pc.delete_index(index_name)
        print("✅ Index deleted")
    
    # Create new index with correct dimensions
    print(f"📊 Creating new index with 1024 dimensions...")
    
    pc.create_index(
        name=index_name,
        dimension=1024,  # For BAAI/bge-large-en-v1.5
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    print(f"✅ Index '{index_name}' created successfully!")
    print(f"📝 Dimension: 1024")
    print(f"📝 Metric: cosine")
    print(f"\n🚀 Now you can upload your PDF!")

if __name__ == "__main__":
    try:
        reset_pinecone_index()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. Your .env file has PINECONE_API_KEY and PINECONE_INDEX_NAME")
        print("2. Your API key is valid")
        print("3. You have permission to delete/create indexes")