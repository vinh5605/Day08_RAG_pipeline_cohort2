"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song (+ HyDE, xem bên dưới)
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

Bonus (+5) — HyDE (Hypothetical Document Embeddings):
    Câu hỏi của user (ngắn, dạng nghi vấn) thường khác văn phong với các đoạn
    văn bản luật/bài báo trong corpus (dài, dạng khẳng định) → embedding của
    câu hỏi và embedding của đoạn văn bản liên quan có thể "lệch" nhau trong
    không gian vector dù nội dung liên quan tới nhau (vocabulary mismatch).

    HyDE giải quyết việc này bằng cách: dùng LLM sinh ra một đoạn văn bản
    "giả định" (hypothetical document) — văn phong gần giống tài liệu thật,
    TRẢ LỜI THẲNG cho câu hỏi (kể cả khi không chắc đúng) — rồi dùng CHÍNH
    embedding của đoạn giả định đó để truy vấn dense retrieval thay vì (hoặc
    cùng với) embedding của câu hỏi gốc. Vì đoạn giả định gần văn phong với
    tài liệu thật hơn câu hỏi gốc, similarity với chunk liên quan thường cao
    và ổn định hơn.

    Triển khai ở đây (`generate_hypothetical_document`):
        - Có LLM (OpenAI/Ollama qua `task10_generation._call_llm`) → sinh đoạn
          văn giả định thật bằng LLM theo `_HYDE_SYSTEM_PROMPT`.
        - Không có LLM khả dụng → `_local_hypothetical_document` tự ghép câu
          hỏi vào khung câu văn phong "khẳng định kiểu văn bản luật/tin tức"
          (đơn giản, không cần model, vẫn giữ tinh thần "đổi văn phong câu hỏi
          → câu khẳng định" cốt lõi của HyDE).
    Sau đó `semantic_search(hypothetical_document, ...)` được merge (RRF) cùng
    semantic_search(query, ...) và lexical_search(query, ...) — xem `retrieve`.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"   # "mmr" | "rrf" | "cross_encoder" — mmr: tự code, chạy local
USE_HYDE = True         # Bonus: bật HyDE query expansion cho dense retrieval


# =============================================================================
# HyDE — Hypothetical Document Embeddings (Bonus +5)
# =============================================================================

_HYDE_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật và báo chí Việt Nam về chủ đề
ma tuý. Với câu hỏi của người dùng, hãy viết MỘT đoạn văn ngắn (3-5 câu, tiếng
Việt) đóng vai trò "tài liệu giả định" trả lời trực tiếp câu hỏi đó — văn phong
giống một trích đoạn trong văn bản pháp luật (Điều/Khoản) hoặc bài báo, dùng
thuật ngữ chuyên ngành phù hợp.

Lưu ý: đoạn văn KHÔNG cần chính xác tuyệt đối về số liệu/điều khoản — mục đích
chỉ là tạo ra một đoạn có văn phong/từ vựng gần với tài liệu thật để hỗ trợ tìm
kiếm ngữ nghĩa (kỹ thuật HyDE), không phải để trả lời chính thức cho người dùng.
Chỉ trả về đoạn văn, không thêm lời dẫn hay giải thích."""


def _local_hypothetical_document(query: str) -> str:
    """
    HyDE fallback không cần LLM.

    Câu hỏi gốc thường ở dạng nghi vấn, ngắn ("Điều 249 quy định hình phạt gì?")
    — khác văn phong khẳng định, dài của văn bản luật/tin tức trong corpus.
    Hàm này "đổi văn phong" một cách đơn giản: bỏ dấu hỏi, ghép câu hỏi vào các
    khung câu khẳng định kiểu văn bản pháp luật/báo chí thường gặp trong corpus
    — giữ đúng tinh thần cốt lõi của HyDE (tạo văn bản giả định gần phong cách
    tài liệu mục tiêu hơn câu hỏi gốc) mà không cần gọi LLM.
    """
    statement = query.strip().rstrip("?！？").strip()
    if not statement:
        return ""
    lowered = statement[0].lower() + statement[1:] if len(statement) > 1 else statement.lower()

    return (
        f"Theo quy định của pháp luật Việt Nam về phòng, chống ma tuý và các "
        f"nguồn tin tức liên quan, {lowered}. Nội dung cụ thể được nêu rõ trong "
        f"các điều khoản của Luật Phòng, chống ma tuý, Bộ luật Hình sự, các nghị "
        f"định hướng dẫn thi hành, hoặc trong các bài báo phản ánh vụ việc liên "
        f"quan, bao gồm chi tiết về hành vi, mức xử phạt và bối cảnh cụ thể."
    )


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh "tài liệu giả định" (hypothetical document) cho HyDE.

    Ưu tiên dùng LLM thật (`task10_generation._call_llm` — OpenAI nếu có
    OPENAI_API_KEY, ngược lại Ollama/Qwen cục bộ nếu server đang chạy); nếu
    không có LLM khả dụng thì dùng `_local_hypothetical_document` (tự code,
    không cần API/model — đúng tinh thần "local/free" của dự án).

    Import `task10_generation` được thực hiện LAZY (bên trong hàm) để tránh
    circular import — task10_generation cũng `from .task9_retrieval_pipeline
    import retrieve`.
    """
    try:
        from .task10_generation import _call_llm

        generated, _backend = _call_llm(_HYDE_SYSTEM_PROMPT, query)
        if generated and generated.strip():
            return generated.strip()
    except Exception as e:
        print(f"  [!] HyDE: không gọi được LLM ({e}) — dùng template cục bộ")

    return _local_hypothetical_document(query)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = USE_HYDE,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search (query gốc)        → results_dense
          ├→ Semantic Search (HyDE doc, bonus)  → results_hyde     [nếu use_hyde]
          ├→ Lexical Search                     → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không
        use_hyde: Có sinh "tài liệu giả định" (HyDE) để mở rộng dense retrieval
                  hay không (bonus +5 — xem `generate_hypothetical_document`)

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    dense_results = semantic_search(query, top_k=top_k * 2)

    dense_lists = [dense_results]
    if use_hyde:
        hypothetical_doc = generate_hypothetical_document(query)
        if hypothetical_doc:
            hyde_results = semantic_search(hypothetical_doc, top_k=top_k * 2)
            if hyde_results:
                dense_lists.append(hyde_results)

    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf(dense_lists + [sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item.setdefault("source", "hybrid")
    else:
        final_results = merged[:top_k]

    # Lưu ý: KHÔNG dùng final_results[0]["score"] để so threshold — score ở đây
    # tuỳ thuộc bước cuối cùng (RRF fusion ~0.01-0.05, MMR/cross-encoder ~0-1,
    # ...) nên không có scale cố định để so với score_threshold (phát hiện khi
    # đánh giá A/B use_reranking=True/False trong group_project/evaluation).
    # Dùng best similarity score của dense/semantic search (luôn ở scale cosine
    # 0-1) làm tín hiệu "độ tin cậy" nhất quán cho quyết định fallback.
    confidence = dense_results[0]["score"] if dense_results else 0.0

    if not final_results or confidence < score_threshold:
        print(f"  [fallback] Best semantic confidence ({confidence:.3f}) < threshold ({score_threshold}) "
              f"-> PageIndex vectorless")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
