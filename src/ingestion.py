"""
Runtime ingestion — nạp tài liệu mới (file upload / URL) vào hệ thống khi
đang chạy, để chatbot có thể trả lời dựa trên tri thức vừa thêm ngay từ câu
hỏi tiếp theo (không cần dừng app / chạy lại pipeline thủ công).

Dùng bởi `group_project/app.py` (Streamlit) và `group_project/web_app.py`
(panel "Nạp thêm tri thức" / "Quản lý Tri thức").

Luồng xử lý (tái sử dụng đúng các bước Task 1 → 4 → 6):
    1. Lưu file/trang web gốc vào data/landing/uploads/          (như Task 1 & 2)
    2. Chuẩn hoá sang markdown, lưu vào data/standardized/uploads/ (như Task 3)
    3. refresh_index(): chunk → embed → index lại ChromaDB (Task 4), nạp lại
       BM25/TF-IDF corpus (Task 6) và ghi snapshot data/local_chunks.json —
       để cả semantic_search lẫn lexical_search đều "thấy" tài liệu mới.

Định dạng hỗ trợ khi upload:
    - PDF/DOCX/DOC → MarkItDown (Task 3)
    - MD/TXT       → đọc thẳng / bọc trong header nguồn
    - JSON         → theo format crawl ở Task 2 (title/url/content_markdown)
    - HTML/HTM     → trích tiêu đề & nội dung chính bằng BeautifulSoup

ingest_url dùng requests + BeautifulSoup — cùng nguyên lý crawl ở Task 2
(xem src/task2_crawl_news.py) nhưng tổng quát hơn (không cần CSS selector
khai báo trước cho từng báo) vì URL có thể là bất kỳ trang nào.
"""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src._shared import PROJECT_DIR, STANDARDIZED_DIR
from src.task4_chunking_indexing import (
    chunk_documents,
    embed_chunks,
    index_to_vectorstore,
    load_documents,
)
from src.task6_lexical_search import reload_corpus

LANDING_UPLOADS_DIR = PROJECT_DIR / "data" / "landing" / "uploads"
STANDARDIZED_UPLOADS_DIR = STANDARDIZED_DIR / "uploads"
LOCAL_CHUNKS_PATH = PROJECT_DIR / "data" / "local_chunks.json"

_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DrugLawIntelBot/1.0)"}
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


# =============================================================================
# Helpers — đặt tên file an toàn, không ghi đè
# =============================================================================

def _slugify(name: str, fallback: str = "document") -> str:
    """Chuẩn hoá tên file/tiêu đề thành tên file an toàn (ASCII, không khoảng trắng)."""
    name = name.strip().replace(" ", "-")
    slug = _SLUG_RE.sub("-", name).strip("-._")
    return slug or fallback


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Tránh ghi đè tài liệu cũ: thêm hậu tố -2, -3, ... nếu trùng tên."""
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# =============================================================================
# Convert → Markdown (mở rộng nguyên lý Task 3 cho nhiều định dạng hơn)
# =============================================================================

def _extract_html(html: str) -> tuple[str, str]:
    """Trích tiêu đề & nội dung chính từ HTML (cùng nguyên lý Task 2: requests + BeautifulSoup,
    nhưng dùng heuristic chung — h1/title + các thẻ văn bản — vì URL có thể là trang bất kỳ)."""
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.find("h1") or soup.find("title")
    title = title_el.get_text(" ", strip=True) if title_el else "Untitled"

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["p", "li", "h2", "h3"])]
    blocks = [b for b in blocks if len(b) > 30]
    body = "\n\n".join(blocks) if blocks else soup.get_text(" ", strip=True)[:5000]
    return title, body


def _convert_upload_to_markdown(filename: str, content: bytes) -> tuple[str, str]:
    """
    Chuyển nội dung 1 file upload sang markdown chuẩn hoá (giữ nguồn để citation).

    Returns:
        (title, markdown_text)
    """
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if suffix in (".pdf", ".docx", ".doc"):
        from markitdown import MarkItDown

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            text = MarkItDown().convert(str(tmp_path)).text_content
        finally:
            tmp_path.unlink(missing_ok=True)

        header = f"# {stem}\n\n**Source:** {filename}\n**Ingested:** {_now()}\n\n---\n\n"
        return stem, header + text

    if suffix in (".md", ".markdown"):
        return stem, content.decode("utf-8", errors="replace")

    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
        header = f"# {stem}\n\n**Source:** {filename}\n**Ingested:** {_now()}\n\n---\n\n"
        return stem, header + text

    if suffix == ".json":
        # Cùng format với output crawl của Task 2 (title/url/content_markdown);
        # nếu không khớp format đó thì giữ nguyên JSON làm nội dung tham khảo.
        data = json.loads(content.decode("utf-8", errors="replace"))
        title = data.get("title", stem)
        body = data.get("content_markdown") or json.dumps(data, ensure_ascii=False, indent=2)
        header = (
            f"# {title}\n\n"
            f"**Source:** {data.get('url', filename)}\n"
            f"**Crawled:** {data.get('date_crawled', _now())}\n\n---\n\n"
        )
        return title, header + body

    if suffix in (".html", ".htm"):
        title, body = _extract_html(content.decode("utf-8", errors="replace"))
        header = f"# {title}\n\n**Source:** {filename}\n**Ingested:** {_now()}\n\n---\n\n"
        return title, header + body

    # Định dạng không xác định: cố gắng decode như văn bản thuần.
    text = content.decode("utf-8", errors="replace")
    header = f"# {stem}\n\n**Source:** {filename}\n**Ingested:** {_now()}\n\n---\n\n"
    return stem, header + text


def _save_markdown(markdown_text: str, stem_hint: str) -> Path:
    STANDARDIZED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _unique_path(STANDARDIZED_UPLOADS_DIR, _slugify(stem_hint), ".md")
    output_path.write_text(markdown_text, encoding="utf-8")
    return output_path


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_DIR)).replace("\\", "/")


# =============================================================================
# Public ingestion entry points
# =============================================================================

def ingest_uploaded_file(filename: str, content: bytes, refresh: bool = True) -> dict:
    """
    Nạp 1 file người dùng upload (PDF/DOCX/DOC/MD/TXT/JSON/HTML) vào hệ thống.

    Args:
        filename: Tên file gốc — dùng để suy ra định dạng & đặt tên output
        content: Nội dung file dạng bytes
        refresh: True → rebuild index ngay (mặc định). Đặt False khi ingest
                 nhiều file liên tiếp rồi gọi `refresh_index()` một lần ở cuối
                 để tránh rebuild lặp lại tốn thời gian.

    Returns:
        {'title': str, 'markdown_path': str, 'raw_path': str, 'stats': dict | None}
    """
    LANDING_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify(filename, fallback="upload")
    raw_path = _unique_path(
        LANDING_UPLOADS_DIR, Path(safe_name).stem, Path(safe_name).suffix or ".bin"
    )
    raw_path.write_bytes(content)

    title, markdown_text = _convert_upload_to_markdown(filename, content)
    markdown_path = _save_markdown(markdown_text, Path(filename).stem)

    return {
        "title": title,
        "markdown_path": _relative(markdown_path),
        "raw_path": _relative(raw_path),
        "stats": refresh_index() if refresh else None,
    }


def ingest_url(url: str, refresh: bool = True) -> dict:
    """
    Tải 1 trang web, trích nội dung chính, chuẩn hoá thành markdown và nạp
    vào hệ thống — cùng nguyên lý crawl với Task 2 (requests + BeautifulSoup),
    nhưng tổng quát hơn vì URL người dùng nhập có thể là trang bất kỳ (không
    có sẵn CSS selector khai báo trước như danh sách báo ở task2_crawl_news).

    Args:
        url: Đường dẫn trang web (bài báo, văn bản pháp luật online, ...)
        refresh: True → rebuild index ngay sau khi ingest

    Returns:
        {'title': str, 'markdown_path': str, 'raw_path': str, 'stats': dict | None}
    """
    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    html = response.text
    title, body = _extract_html(html)

    LANDING_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    raw_stem = _slugify(title) if title and title != "Untitled" else _slugify(url)
    raw_path = _unique_path(LANDING_UPLOADS_DIR, raw_stem, ".html")
    raw_path.write_text(html, encoding="utf-8")

    markdown_text = (
        f"# {title}\n\n**Source:** {url}\n**Crawled:** {_now()}\n\n---\n\n{body}"
    )
    markdown_path = _save_markdown(markdown_text, raw_stem)

    return {
        "title": title,
        "markdown_path": _relative(markdown_path),
        "raw_path": _relative(raw_path),
        "stats": refresh_index() if refresh else None,
    }


# =============================================================================
# Index refresh — chunk/embed/index lại (Task 4) + nạp lại lexical corpus (Task 6)
# =============================================================================

def _write_local_chunks_snapshot(chunks: list[dict]) -> None:
    """
    Ghi snapshot toàn bộ chunks (content + metadata, KHÔNG kèm embedding — để
    file gọn & người dùng có thể đọc trực tiếp) ra data/local_chunks.json.

    Đây là "bản sao cục bộ dạng JSON" — đúng tinh thần kiến trúc High-
    availability của nhóm ("Fallback quay về truy vấn cục bộ JSON/TF-IDF khi
    mất kết nối Cloud Vector DB / API"), đồng thời cho người dùng xem nhanh
    corpus hiện có (kể cả tài liệu vừa upload) mà không cần mở ChromaDB.
    """
    snapshot = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    LOCAL_CHUNKS_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def refresh_index() -> dict:
    """
    Rebuild toàn bộ index từ data/standardized/ (gồm cả tài liệu vừa ingest):

        1. load_documents → chunk_documents → embed_chunks → index_to_vectorstore
           (đúng pipeline Task 4 — ChromaDB collection được xoá & ghi lại từ
           đầu nên luôn đồng bộ, idempotent)
        2. task6_lexical_search.reload_corpus() — nạp lại BM25 + TF-IDF index
           để lexical_search cũng thấy tài liệu mới
        3. Ghi snapshot data/local_chunks.json

    Có thể gọi lại nhiều lần an toàn (idempotent). Trả về thống kê để UI hiển
    thị cho người dùng biết corpus hiện có bao nhiêu document/chunk.

    Returns:
        {'documents': int, 'chunks': int}
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = embed_chunks(chunks)
    index_to_vectorstore(chunks)

    reload_corpus()
    _write_local_chunks_snapshot(chunks)

    return {"documents": len(documents), "chunks": len(chunks)}


if __name__ == "__main__":
    print("Refreshing index from data/standardized/ ...")
    stats = refresh_index()
    print(f"✓ {stats['documents']} documents -> {stats['chunks']} chunks indexed")
    print(f"✓ Snapshot written to {LOCAL_CHUNKS_PATH}")
