import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Must match the model build_qdrant.py used to create the collection.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_vectorstore = None
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=os.environ.get("QDRANT_COLLECTION", "divergent_children"),
            embedding=get_embeddings(),
        )
    return _vectorstore


def rag_retrieve(query: str, k: int = 5) -> str:
    """Single-pass similarity search over child chunks, expanded to each
    child's parent text for real context — mirrors the parent/child
    hierarchy your ingestion script (build_qdrant.py) built: only children
    are embedded, but each child stores its parent's full text in metadata.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found in Qdrant."

    seen_parents = set()
    parts = []
    for d in docs:
        parent_id = d.metadata.get("parent_id")
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        chapter = d.metadata.get("chapter", "unknown chapter")
        page_start = d.metadata.get("chapter_page_start")
        page_end = d.metadata.get("chapter_page_end")
        text = d.metadata.get("parent_text", d.page_content)

        parts.append(f"[{chapter}, pages {page_start}-{page_end}] {text[:800]}")

    return "\n\n".join(parts)