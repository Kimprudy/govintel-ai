import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from govintel.config import PROCESSED_DIR, OPENAI_API_KEY

CHROMA_DIR = "chroma_db"
COLLECTION = "ecfr"
EMBED_MODEL = "text-embedding-3-small"


def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)


def get_store():
    """Open the existing vector store without re-indexing."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def load_documents(title_number: int) -> list:
    path = os.path.join(PROCESSED_DIR, f"title-{title_number}-chunks.jsonl")
    docs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            docs.append(Document(
                page_content=c["text"],
                metadata={
                    "chunk_id": c["chunk_id"],
                    "citation": c["citation"],
                    "title": c["title"],
                    "part": c["part"],
                    "section": c["section"],
                    "heading": c["heading"],
                    "snapshot_date": c["snapshot_date"],
                },
            ))
    return docs


def build_index(title_number: int = 2, batch_size: int = 100):
    docs = load_documents(title_number)
    print(f"Indexing {len(docs)} chunks...")

    store = get_store()

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)}")

    print(f"Done. Persisted to {CHROMA_DIR}/")
    return store


if __name__ == "__main__":
    build_index(2)
