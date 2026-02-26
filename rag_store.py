# rag_store.py

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from load_kb import load_knowledge_base

# 1. Load your data
documents = load_knowledge_base("kb.txt")

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Create Qdrant in-memory database
client = QdrantClient(":memory:")

# 4. Create collection
client.recreate_collection(
    collection_name="knowledge_base",
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    }
)

# 5. Generate embeddings and store
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)

client.upload_collection(
    collection_name="knowledge_base",
    vectors=embeddings,
    payload=documents,
    ids=[doc["id"] for doc in documents]
)

print("✅ Embeddings stored in Qdrant")