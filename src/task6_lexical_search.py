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
"""

import re

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


_BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
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


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
