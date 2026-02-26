# rag_query.py

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from load_kb import load_knowledge_base

# Load data again
documents = load_knowledge_base("kb.txt")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant (in-memory)
client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name="knowledge_base",
    vectors_config={"size": 384, "distance": "Cosine"}
)

texts = [doc["text"] for doc in documents]
embeddings = embed_model.encode(texts)

client.upload_collection(
    collection_name="knowledge_base",
    vectors=embeddings,
    payload=documents,
    ids=[doc["id"] for doc in documents]
)

# Connect to Llamafile
llm = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# User query
query = input("Ask a question: ")

query_vector = embed_model.encode(query)

search_result = client.query_points(
    collection_name="knowledge_base",
    query=query_vector.tolist(),
    limit=3
).points

context = "\n".join([point.payload["text"] for point in search_result])

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

response = llm.chat.completions.create(
    model="LLaMA_CPP",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:\n", response.choices[0].message.content)