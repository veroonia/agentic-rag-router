import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# Must match the embedding model used when
# the Qdrant collection was created.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_vectorstore = None
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore

    if _vectorstore is None:

        qdrant_url = os.environ.get(
            "QDRANT_URL",
            "http://localhost:6333"
        )

        collection_name = os.environ.get(
            "QDRANT_COLLECTION",
            "divergent_children"
        )

        qdrant_api_key = os.environ.get(
            "QDRANT_API_KEY"
        )

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
        )

        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=get_embeddings(),
        )

    return _vectorstore


def rag_retrieve(
    query: str,
    k: int = 5,
) -> str:
    """
    Perform a single similarity search against Qdrant.

    The existing collection contains child chunks with metadata
    pointing to their parent chunks.
    """

    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=k,
    )

    if not docs:
        return "No relevant documents found in Qdrant."

    seen_parents = set()
    parts = []

    for doc in docs:

        parent_id = doc.metadata.get(
            "parent_id"
        )

        # Avoid returning the same parent multiple times.
        if parent_id in seen_parents:
            continue

        seen_parents.add(parent_id)

        chapter = doc.metadata.get(
            "chapter",
            "unknown chapter",
        )

        page_start = doc.metadata.get(
            "chapter_page_start"
        )

        page_end = doc.metadata.get(
            "chapter_page_end"
        )

        text = doc.metadata.get(
            "parent_text",
            doc.page_content,
        )

        parts.append(
            f"[{chapter}, pages {page_start}-{page_end}]\n"
            f"{text[:1200]}"
        )

    return "\n\n".join(parts)