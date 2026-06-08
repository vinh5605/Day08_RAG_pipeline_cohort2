"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Cách triển khai:
    Dùng đúng embedding model + ChromaDB collection đã tạo ở Task 4
    (qua src._shared, để đảm bảo tương thích chiều vector & tên collection).
    Chroma collection được cấu hình với "hnsw:space": "cosine" và lưu các
    embedding đã normalize → khoảng cách trả về là cosine distance, nên
    similarity = 1 - distance (càng gần 1 càng giống).
"""

from src._shared import embed_texts, get_chroma_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]
    for content, metadata, distance in zip(documents, metadatas, distances):
        results.append({
            "content": content,
            "score": 1.0 - distance,
            "metadata": dict(metadata),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
