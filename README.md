# Retrieval Augmented Generation using Llamafile and Qdrant

## Overview
This project implements a full Retrieval Augmented Generation (RAG) pipeline using
custom text data. The system retrieves relevant information from a vector database
(Qdrant) and generates answers using a locally hosted Large Language Model via Llamafile.

## Components Used
- Sentence Transformers (`all-MiniLM-L6-v2`) for text embeddings
- Qdrant (in-memory) as the vector database
- Phi-2 Llamafile as a local LLM (OpenAI-compatible API)
- OpenAI Python client for LLM interaction

## Data
The knowledge base is stored in a text file (`kb.txt`) and is split into chunks for
embedding and retrieval.

## How It Works
1. Text data is loaded and chunked
2. Embeddings are created using Sentence Transformers
3. Embeddings are stored in Qdrant
4. User query is embedded and matched against stored vectors
5. Retrieved context is passed to the LLM
6. LLM generates an answer grounded in retrieved data

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Start the Llamafile server

./phi-2.llamafile

## Run the RAG system:

python rag_query.py
