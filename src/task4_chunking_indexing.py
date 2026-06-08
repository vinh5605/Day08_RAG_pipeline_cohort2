"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store

Lựa chọn & giải thích:
    - Chunking: RecursiveCharacterTextSplitter (an toàn, phổ biến). Văn bản pháp
      luật có cấu trúc Điều/Khoản dài, bài báo có đoạn văn ngắn — splitter này
      thử tách theo "\\n\\n" → "\\n" → ". " → " " trước khi cắt cứng, nên giữ được
      ranh giới đoạn/câu trong phần lớn trường hợp.
    - CHUNK_SIZE = 500 ký tự: đủ lớn để 1 chunk chứa trọn 1 đoạn/khoản ngắn
      (giữ ngữ cảnh để semantic search "hiểu" được câu hỏi liên quan), đủ nhỏ để
      embedding model encode nhanh và để reranker không bị loãng thông tin.
    - CHUNK_OVERLAP = 50 ký tự (10% chunk size): giữ một phần ngữ cảnh giữa 2
      chunk liền kề, tránh trường hợp 1 câu/điều khoản quan trọng bị cắt đúng
      ở ranh giới chunk và mất nghĩa.
    - Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      (xem src/_shared.py) — multilingual, hỗ trợ tiếng Việt, 384 chiều, nhẹ
      (~470MB) nên chạy tốt trên máy cá nhân không cần GPU/API key.
    - Vector store: ChromaDB — PersistentClient lưu thẳng xuống đĩa
      (data/chroma_db/), không cần Docker/server/cloud account, phù hợp chạy
      local cho bài tập cá nhân.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src._shared import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    STANDARDIZED_DIR as _STANDARDIZED_DIR,
    embed_texts,
    get_chroma_collection,
)

STANDARDIZED_DIR = _STANDARDIZED_DIR


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

CHUNK_SIZE = 500        # Đủ chứa 1 đoạn/khoản, nhỏ gọn cho embedding & rerank
CHUNK_OVERLAP = 50      # 10% chunk size — giữ ngữ cảnh ở ranh giới các chunk
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_MODEL = EMBEDDING_MODEL_NAME  # multilingual MiniLM — nhẹ, hỗ trợ tiếng Việt
EMBEDDING_DIM = EMBEDDING_DIM

VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss" — local, không cần server


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn (xem src._shared.embed_texts).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB collection (persist xuống data/chroma_db/).
    Idempotent: xoá dữ liệu cũ trong collection trước khi ghi lại từ đầu.
    """
    collection = get_chroma_collection()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{c['metadata']['source']}::{c['metadata']['chunk_index']}" for c in chunks]
    collection.add(
        ids=ids,
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} -> {CHROMA_DIR} (collection={COLLECTION_NAME})")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
