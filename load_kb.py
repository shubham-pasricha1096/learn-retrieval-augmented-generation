# load_kb.py

def load_knowledge_base(file_path="kb.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("\n\n")  # split by paragraphs

    documents = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk:
            documents.append({
                "id": idx,
                "text": chunk
            })

    return documents


if __name__ == "__main__":
    docs = load_knowledge_base()
    print(f"Loaded {len(docs)} documents")
    print(docs[:2])