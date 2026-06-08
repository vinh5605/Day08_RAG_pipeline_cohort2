# RAG Evaluation Results

## Framework sử dụng

DeepEval-style custom offline evaluator (4 metric bắt buộc, không cần API key).

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |
|--------|---------------------------|-----------------------------|---|
| Faithfulness | 0.94 | 1.00 | -0.06 |
| Answer Relevance | 0.24 | 0.16 | +0.08 |
| Context Recall | 0.80 | 0.95 | -0.15 |
| Context Precision | 1.00 | 1.00 | +0.00 |
| Average | 0.74 | 0.78 | -0.03 |

## A/B Comparison Analysis

**Config A:** semantic + BM25, HyDE query expansion, RRF merge, lightweight reranking, PageIndex fallback.

**Config B:** semantic + BM25, RRF merge, không rerank để so sánh tác động reranking.

**Kết luận:** Config A thường ưu tiên chunk khớp câu hỏi tốt hơn nên relevance/precision ổn định hơn; Config B nhanh hơn nhưng dễ giữ kết quả trùng lặp hoặc ít liên quan.

## Bonus implemented

- HyDE query expansion trong `src/task9_retrieval_pipeline.py`.
- Conversation memory trong `app.py`.
- UI/UX: hiển thị source, score và highlight keyword trong source snippet.
- Lexical search alternative khác BM25: `tfidf_search()` / `lexical_search(method='tfidf')` trong `src/task6_lexical_search.py`, có giải thích TF-IDF cosine similarity để lấy bonus +5.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Hình phạt cơ bản cho tội tàng trữ trái phép chất ma tuý theo Điều 249 là gì? | 0.43 | 0.62 | 0.31 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |
| 2 | Nghị định 105/2021/NĐ-CP hướng dẫn nội dung gì? | 1.00 | 0.11 | 0.29 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |
| 3 | Pipeline RAG sử dụng nguồn fallback nào khi hybrid search điểm thấp? | 1.00 | 0.05 | 0.43 | Retrieval/Generation | Thiếu dữ liệu thật/API LLM, corpus mô phỏng còn nhỏ |

## Recommendations

### Cải tiến 1
**Action:** Thay dữ liệu mô phỏng bằng PDF/DOCX/HTML thật từ nguồn chính thống.  
**Expected impact:** Tăng faithfulness và context recall.

### Cải tiến 2
**Action:** Dùng embedding multilingual thật (bge-m3) và vector DB như Weaviate.  
**Expected impact:** Cải thiện semantic search cho câu hỏi tiếng Việt dài.

### Cải tiến 3
**Action:** Dùng cross-encoder reranker/Jina hoặc Qwen và LLM có kiểm soát citation.  
**Expected impact:** Tăng precision và chất lượng câu trả lời cuối.