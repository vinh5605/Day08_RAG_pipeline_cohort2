"""
Cấu hình & helpers dùng chung giữa Task 4 (indexing), Task 5 (semantic search)
và Task 6 (lexical search) — tránh lặp lại đường dẫn / tên collection / model
ở nhiều nơi rồi bị lệch nhau.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "DrugLawDocs"

# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2:
#   - Multilingual (50+ ngôn ngữ, gồm tiếng Việt), 384 chiều, ~470MB.
#   - Nhẹ hơn nhiều so với BAAI/bge-m3 (~2.3GB) nhưng vẫn cho chất lượng tốt
#     cho retrieval tiếng Việt — phù hợp chạy trên máy cá nhân không có GPU mạnh.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

_model = None
_client = None


def get_embedding_model():
    """Trả về SentenceTransformer đã load (cache lại để khỏi load nhiều lần)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một danh sách text, trả về list vector đã normalize (cosine-ready)."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def get_chroma_collection():
    """Trả về (hoặc tạo mới) ChromaDB collection lưu trên đĩa tại CHROMA_DIR."""
    global _client
    import chromadb

    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
