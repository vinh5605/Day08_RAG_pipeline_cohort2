"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Triển khai trong file này:
    1. Đường SDK thật (upload_documents / nhánh trong pageindex_search):
       dùng khi có PAGEINDEX_API_KEY trong .env — gọi đúng API của PageIndex.
    2. Local fallback `_local_structural_search` (chạy mặc định, không cần
       API key, đúng tinh thần "local/free"): tự xây "page tree" — tách mỗi
       file markdown thành các section theo heading (#, ##, ###...) và giữ
       "structural path" (vd: "Nghị định 105 > Chương III > Điều 14"), rồi
       so khớp query với section bằng độ phủ token trên (tiêu đề + nội dung).
       Đây chính là "vectorless": không hề dùng embedding/vector similarity,
       mà dựa vào CẤU TRÚC tài liệu (outline) + lexical match — cùng nguyên
       lý cốt lõi mà PageIndex quảng cáo (structural retrieval thay vì dense
       vector search), chỉ khác là PageIndex dùng LLM để duyệt cây còn ở đây
       dùng token-overlap cho đơn giản & không cần model/API.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _build_section_index() -> list[dict]:
    """
    Tách mỗi markdown file thành các "structural section" theo heading,
    mô phỏng "page tree" của PageIndex — mỗi section giữ outline path đầy đủ
    từ root tới heading hiện tại (vd: "... > Chương III > Điều 14: ...").
    """
    sections = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"

        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            content = text.strip()
            if len(content) >= 20:
                sections.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "type": doc_type, "path": md_file.stem},
                })
            continue

        path_stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            path_stack = [(lv, h) for lv, h in path_stack if lv < level]
            path_stack.append((level, heading))
            structural_path = " > ".join(h for _, h in path_stack)

            content = f"{structural_path}\n{body}" if body else structural_path
            if len(content.strip()) < 20:
                continue
            sections.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type, "path": structural_path},
            })
    return sections


_SECTION_INDEX: list[dict] = _build_section_index()


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex (cần PAGEINDEX_API_KEY).
    """
    from pageindex import PageIndex

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        pi.upload(
            content=content,
            metadata={"filename": md_file.name, "type": md_file.parent.name},
        )
        print(f"  - Uploaded: {md_file.name}")


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    """Vectorless fallback: so khớp query với structural sections bằng token overlap."""
    if not _SECTION_INDEX:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for section in _SECTION_INDEX:
        section_tokens = _tokenize(section["content"])
        overlap = query_tokens & section_tokens
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens)
        scored.append((score, section))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "content": section["content"],
            "score": score,
            "metadata": section["metadata"],
            "source": "pageindex",
        }
        for score, section in scored[:top_k]
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Nếu có PAGEINDEX_API_KEY → gọi SDK thật; nếu không (hoặc SDK lỗi) →
    dùng `_local_structural_search` (tự code, không cần API key).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception as e:
            print(f"  [!] PageIndex API lỗi ({e}) — fallback sang local structural search")

    return _local_structural_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chưa được set trong .env — dùng local structural search.")
        print("(Đăng ký tại https://pageindex.ai/ để dùng SDK thật)")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
