"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Generation backend (theo thứ tự ưu tiên — xem `_call_llm`):
    1. OpenAI (`gpt-4o-mini`) nếu có OPENAI_API_KEY trong .env.
    2. Ollama cục bộ (model Qwen, xem OLLAMA_MODEL) nếu server Ollama đang
       chạy ở OLLAMA_BASE_URL — đúng kiến trúc "Generation triển khai cục bộ
       bằng Ollama (model Qwen)" của nhóm, miễn phí & không cần API key.
    3. `_generate_local_answer` (giải pháp local/free cuối cùng): tự tổng hợp
       câu trả lời trực tiếp từ các chunk đã retrieve, MỖI đoạn đều kèm
       "[Nguồn: ...]" — vẫn tuân thủ yêu cầu cốt lõi của task (mọi câu trả lời
       đều có citation, và nói rõ khi không có evidence) mà không cần gọi LLM.
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


def _generate_openai_answer(system_prompt: str, user_message: str) -> str | None:
    """Gọi OpenAI thật nếu có OPENAI_API_KEY; trả None nếu lỗi/không có key."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [!] OpenAI API lỗi ({e})")
        return None


# =============================================================================
# OLLAMA — Local generation backend (model Qwen)
# =============================================================================
#
# Theo kiến trúc của nhóm: "Generation triển khai cục bộ bằng Ollama (model
# Qwen)" — đây là backend miễn phí, chạy hoàn toàn local, không cần API key.
# Cài đặt: https://ollama.com → `ollama pull qwen2.5:3b` (hoặc model Qwen khác)
# rồi chạy `ollama serve` (mặc định lắng nghe ở http://localhost:11434).

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
_OLLAMA_PING_TIMEOUT = 1.5   # giây — chỉ "ping" xem server có chạy không, không chờ generation
_OLLAMA_GENERATE_TIMEOUT = 180


def _ollama_available() -> bool:
    """Kiểm tra Ollama server cục bộ có đang chạy ở OLLAMA_BASE_URL không."""
    try:
        import requests

        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=_OLLAMA_PING_TIMEOUT)
        return response.ok
    except Exception:
        return False


def _generate_ollama_answer(system_prompt: str, user_message: str) -> str | None:
    """Gọi Ollama (model Qwen, chạy local) qua REST API; trả None nếu lỗi."""
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
            },
            timeout=_OLLAMA_GENERATE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content")
    except Exception as e:
        print(f"  [!] Ollama lỗi ({e})")
        return None


def _call_llm(system_prompt: str, user_message: str) -> tuple[str | None, str]:
    """
    Gọi LLM theo thứ tự ưu tiên — dùng chung cho generation (Task 10) lẫn
    HyDE (Task 9, bonus): OpenAI (nếu có OPENAI_API_KEY) → Ollama/Qwen cục bộ
    (nếu server đang chạy) → None (caller tự fallback cục bộ phù hợp với mình,
    vd: `_generate_local_answer` ở đây hay `_local_hypothetical_document` ở
    task9_retrieval_pipeline).

    Returns:
        (text, backend) — backend ∈ {"openai", "ollama", "local"}.
        `text` là None khi cả OpenAI lẫn Ollama đều không khả dụng/đều lỗi.
    """
    answer = _generate_openai_answer(system_prompt, user_message)
    if answer:
        return answer, "openai"

    if _ollama_available():
        answer = _generate_ollama_answer(system_prompt, user_message)
        if answer:
            return answer, "ollama"

    return None, "local"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (hybrid + HyDE, xem task9_retrieval_pipeline)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM: OpenAI → Ollama/Qwen cục bộ → tổng hợp local (xem `_call_llm`)
        6. Return answer + sources + backend đã dùng

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str, # 'hybrid' hoặc 'pageindex'
            'llm': str               # Backend sinh câu trả lời: openai|ollama|local
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    answer, llm = _call_llm(SYSTEM_PROMPT, user_message)
    if answer is None:
        answer = _generate_local_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "llm": llm,
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
