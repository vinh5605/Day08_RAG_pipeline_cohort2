"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Tokenization:
    Tiếng Việt không có khoảng trắng giữa các "từ ghép" (vd: "ma túy" là 1 từ
    gồm 2 âm tiết), nhưng việc tách theo âm tiết (.split() đơn giản) vẫn cho
    BM25 đủ tín hiệu để match — n-gram âm tiết trùng nhau giữa query và
    document là đủ để tính TF/IDF có ý nghĩa, mà không cần thêm dependency
    phân đoạn từ (underthesea/pyvi). Lowercase trước khi tách để so khớp
    không phân biệt hoa/thường.

Bonus (+5) — Lexical alternative khác BM25: TF-IDF + cosine similarity
    (xem `tfidf_search` / `lexical_search(method="tfidf")`).
    - TF (Term Frequency): tần suất từ trong document, chuẩn hoá theo độ dài
      document — document dài không bị "lợi thế" vì lặp từ nhiều lần.
    - IDF (Inverse Document Frequency, smoothed): idf(t) = ln((1+N)/(1+df(t))) + 1
      — từ xuất hiện ở càng ít document thì trọng số càng cao (cùng công thức
      "smooth_idf" mà scikit-learn TfidfVectorizer dùng mặc định).
    - Mỗi document/query → 1 vector thưa {term: tf*idf}, chuẩn hoá L2; độ liên
      quan = cosine similarity giữa 2 vector (tích vô hướng vì đã chuẩn hoá).
    - Khác biệt với BM25: BM25 có "term saturation" (tần suất từ càng cao thì
      điểm tăng càng chậm, kiểm soát bởi k1) và length-normalization phi tuyến
      (tham số b); TF-IDF tuyến tính theo tần suất và không có cơ chế bão hoà
      — đơn giản, dễ giải thích hơn, nhưng dễ bị chi phối bởi document lặp từ
      khoá nhiều lần hơn so với BM25.
"""

import math
import re
from collections import Counter

from src._shared import STANDARDIZED_DIR
from src.task4_chunking_indexing import chunk_documents

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_corpus() -> list[dict]:
    """Đọc & chunk toàn bộ markdown trong data/standardized/ thành corpus cho BM25."""
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return chunk_documents(documents)


CORPUS: list[dict] = _load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# =============================================================================
# TF-IDF + Cosine Similarity (Bonus +5 — lexical alternative khác BM25)
# =============================================================================

def _tfidf_vectorize(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Biến 1 danh sách token thành vector thưa {term: tf*idf}, đã chuẩn hoá L2."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    length = len(tokens)
    vector = {term: (count / length) * idf.get(term, 0.0) for term, count in counts.items()}
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0:
        return {}
    return {term: weight / norm for term, weight in vector.items()}


def build_tfidf_index(corpus: list[dict]) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Xây dựng TF-IDF index: bảng IDF (smoothed, kiểu scikit-learn) + vector
    TF-IDF (đã chuẩn hoá L2) cho từng document trong corpus.

        idf(t) = ln((1 + N) / (1 + df(t))) + 1

    Returns:
        (idf_table, document_vectors) — document_vectors[i] tương ứng corpus[i]
    """
    tokenized_docs = [_tokenize(doc["content"]) for doc in corpus]

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    n_docs = len(corpus)
    idf = {
        term: math.log((1 + n_docs) / (1 + df)) + 1.0
        for term, df in document_frequency.items()
    }

    document_vectors = [_tfidf_vectorize(tokens, idf) for tokens in tokenized_docs]
    return idf, document_vectors


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity giữa 2 vector thưa đã chuẩn hoá L2 (= tích vô hướng)."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


_BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None
_TFIDF_IDF, _TFIDF_DOC_VECTORS = build_tfidf_index(CORPUS) if CORPUS else ({}, [])


def reload_corpus() -> int:
    """
    Nạp lại corpus từ data/standardized/ (vd: sau khi runtime-ingestion thêm
    tài liệu mới) và xây lại cả BM25 lẫn TF-IDF index.

    Returns:
        Số chunks trong corpus sau khi reload.
    """
    global CORPUS, _BM25_INDEX, _TFIDF_IDF, _TFIDF_DOC_VECTORS

    CORPUS = _load_corpus()
    _BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None
    _TFIDF_IDF, _TFIDF_DOC_VECTORS = build_tfidf_index(CORPUS) if CORPUS else ({}, [])
    return len(CORPUS)


def _bm25_search(query: str, top_k: int) -> list[dict]:
    if _BM25_INDEX is None:
        return []

    tokenized_query = _tokenize(query)
    scores = _BM25_INDEX.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical search bằng TF-IDF + cosine similarity — phương pháp khác BM25
    (xem giải thích cơ chế ở docstring đầu file → bonus +5).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by cosine similarity descending (score trong khoảng [0, 1]).
    """
    if not _TFIDF_DOC_VECTORS:
        return []

    query_vector = _tfidf_vectorize(_tokenize(query), _TFIDF_IDF)
    if not query_vector:
        return []

    scored = []
    for idx, doc_vector in enumerate(_TFIDF_DOC_VECTORS):
        if not doc_vector:
            continue
        similarity = _cosine_sparse(query_vector, doc_vector)
        if similarity > 0:
            scored.append((similarity, idx))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "content": CORPUS[idx]["content"],
            "score": float(similarity),
            "metadata": CORPUS[idx]["metadata"],
        }
        for similarity, idx in scored[:top_k]
    ]


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    """
    Tìm kiếm từ khóa (lexical search).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) | "tfidf" — phương pháp khác BM25, xem
                `tfidf_search` để biết cơ chế TF-IDF cosine similarity
                (đủ điều kiện bonus +5 "giải thích lexical search khác BM25").

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score hoặc TF-IDF cosine similarity
            'metadata': dict
        }
        Sorted by score descending.
    """
    if method == "tfidf":
        return tfidf_search(query, top_k=top_k)
    if method == "bm25":
        return _bm25_search(query, top_k=top_k)
    raise ValueError(f"Unknown lexical search method: {method!r} (expected 'bm25' or 'tfidf')")


if __name__ == "__main__":
    # Test — so sánh BM25 vs TF-IDF trên cùng 1 query
    query = "Điều 248 tàng trữ trái phép chất ma tuý"

    print("--- BM25 ---")
    for r in lexical_search(query, top_k=5, method="bm25"):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print("\n--- TF-IDF (cosine similarity) ---")
    for r in lexical_search(query, top_k=5, method="tfidf"):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
