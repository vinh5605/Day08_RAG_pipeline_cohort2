"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Generation backend:
    - Nếu có OPENAI_API_KEY trong .env → gọi OpenAI thật theo đúng SYSTEM_PROMPT.
    - Nếu không (giải pháp local/free mặc định) → `_generate_local_answer` tự
      tổng hợp câu trả lời trực tiếp từ các chunk đã retrieve, MỖI đoạn đều
      kèm "[Nguồn: ...]" — vẫn tuân thủ yêu cầu cốt lõi của task (mọi câu trả
      lời đều có citation, và nói rõ khi không có evidence) mà không cần gọi
      API trả phí.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence (thường phủ 1-2 điều luật + 1-2 bài báo liên quan)
# mà không quá dài khiến prompt loãng / gây "lost in the middle".
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ tự nhiên về diễn đạt nhưng không quá ngẫu nhiên/lan man.
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần câu trả lời bám sát evidence (factual), hạn chế "sáng tạo".
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Rank lẻ (1st, 3rd, 5th, ...) → đưa lên đầu, theo đúng thứ tự ưu tiên
    head = chunks[0::2]
    # Rank chẵn (2nd, 4th, ...) → đưa xuống cuối, đảo ngược (4th trước 2nd)
    # để rank cao hơn nằm gần cuối hơn (tận dụng "recency" attention)
    tail = chunks[1::2][::-1]

    return head + tail


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _generate_local_answer(query: str, reordered: list[dict]) -> str:
    """
    Local/free fallback (không gọi LLM API): tổng hợp câu trả lời trực tiếp
    từ các chunk đã retrieve, mỗi đoạn trích đều kèm "[Nguồn: ...]" — vẫn giữ
    đúng tinh thần "mọi claim phải có citation / nói rõ khi thiếu evidence"
    của SYSTEM_PROMPT mà không cần OPENAI_API_KEY.
    """
    if not reordered:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines = [f'Tổng hợp các đoạn trích liên quan tới câu hỏi: "{query}"\n']
    for chunk in reordered:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "không rõ nguồn")
        snippet = " ".join(chunk["content"].split())
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {snippet} [Nguồn: {source}]")

    lines.append(
        "\n(Chế độ tổng hợp cục bộ — không gọi LLM API. Câu trả lời là trích dẫn "
        "trực tiếp từ các nguồn trên; vui lòng đối chiếu văn bản gốc trước khi dùng chính thức.)"
    )
    return "\n".join(lines)


def _generate_openai_answer(query: str, context: str) -> str | None:
    """Gọi OpenAI thật nếu có OPENAI_API_KEY; trả None nếu lỗi/không có key."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [!] OpenAI API lỗi ({e}) — fallback sang local citation generator")
        return None


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM (OpenAI nếu có key, ngược lại fallback local)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _generate_openai_answer(query, context)
    if answer is None:
        answer = _generate_local_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
