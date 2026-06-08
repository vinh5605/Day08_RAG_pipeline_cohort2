"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lựa chọn cho bài này: MMR là phương pháp mặc định (`rerank`).
    - Tự code, không cần API key/model bên ngoài (đúng tinh thần "local/free"):
      chỉ cần embedding model đã có sẵn từ Task 4/5 (src._shared.embed_texts).
    - MMR vừa xét độ liên quan tới query (relevance), vừa xét độ đa dạng giữa
      các kết quả đã chọn (diversity) — tránh tình trạng top-k toàn các chunk
      gần giống nhau (vd: 3 khoản liên tiếp của cùng 1 điều luật) mà bỏ sót
      góc nhìn khác (vd: 1 chunk từ bài báo liên quan).
      MMR(d) = λ·sim(query, d) − (1−λ)·max_{s ∈ selected} sim(d, s)
    - RRF cũng được code đầy đủ — dùng khi cần gộp nhiều ranked list (vd: kết
      quả từ semantic_search + lexical_search) trước khi đưa vào MMR/LLM.
    - rerank_cross_encoder dùng 1 model cross-encoder multilingual chạy local
      (lazy-load, chỉ tải khi thực sự gọi tới — không ảnh hưởng pipeline mặc định).
"""

import numpy as np

from src._shared import embed_texts


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# =============================================================================
# Cross-encoder (local, multilingual, lazy-loaded — không cần API key)
# =============================================================================

_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (chạy local, multilingual).

    Cross-encoder encode CHUNG (query, document) qua 1 model — chính xác hơn
    bi-encoder (encode riêng rồi so cosine) vì model "thấy" cả 2 văn bản cùng
    lúc, nhưng đổi lại chậm hơn nên chỉ dùng để rerank top candidates.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [[query, c["content"]] for c in candidates]
    scores = model.predict(pairs)

    rescored = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda r: r["score"], reverse=True)
    return rescored[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance (tự implement)
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    relevance = [_cosine_sim(query_embedding, c["embedding"]) for c in candidates]

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = {k: v for k, v in candidates[idx].items() if k != "embedding"}
        item["score"] = relevance[idx]
        results.append(item)
    return results


# =============================================================================
# RRF — Reciprocal Rank Fusion (tự implement)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Mỗi ranker (semantic, lexical, ...) cho ra 1 thứ hạng riêng cho cùng 1
    document. RRF cộng dồn "điểm nghịch đảo hạng" từ mọi ranker — document
    nào được nhiều ranker xếp hạng cao sẽ có tổng điểm cao, mà không cần các
    ranker dùng chung 1 thang điểm (BM25 score và cosine similarity vốn không
    so sánh trực tiếp được). k=60 làm mượt chênh lệch giữa các hạng đầu.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = {kk: vv for kk, vv in content_map[content].items() if kk != "embedding"}
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "mmr" | "rrf" | "cross_encoder"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking (mặc định "mmr" — tự code, chạy local)

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "mmr":
        # Candidates không nhất thiết có sẵn 'embedding' (vd: tới từ BM25) nên
        # embed query + content ngay tại đây bằng cùng model ở Task 4/5.
        texts = [query] + [c["content"] for c in candidates]
        vectors = embed_texts(texts)
        query_embedding, doc_embeddings = vectors[0], vectors[1:]
        enriched = [{**c, "embedding": vec} for c, vec in zip(candidates, doc_embeddings)]
        return rerank_mmr(query_embedding, enriched, top_k)

    if method == "rrf":
        return rerank_rrf([candidates], top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
