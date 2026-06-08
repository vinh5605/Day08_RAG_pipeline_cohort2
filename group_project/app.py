"""Professional Streamlit RAG Chatbot UI.

Run:
    streamlit run app.py

Features:
- Domain-specific chatbot UI for Vietnamese drug law + news about artists/drug cases.
- Conversation memory.
- Citation-oriented answers.
- Source cards with score, type, path/source and keyword highlighting.
- Runtime ingestion: upload PDF/DOCX/DOC/MD/TXT/JSON/HTML or add a web URL.
- Ingested documents are saved to data/landing/uploads and data/standardized/uploads,
  then data/local_chunks.json is rebuilt so the next retrieval can use the new docs.
"""

from __future__ import annotations

try:
    import streamlit as st
except Exception:  # Allows tests/import in environments without streamlit.
    st = None

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.task10_generation import generate_with_citation
from src.ingestion import ingest_uploaded_file, ingest_url, refresh_index


APP_TITLE = "DrugLaw Intel Chatbot"
APP_SUBTITLE = "Trợ lý RAG chuyên về pháp luật ma túy Việt Nam và tin tức nghệ sĩ liên quan ma túy"


CUSTOM_CSS = """
<style>
:root {
  --primary: #7c2d12;
  --primary-soft: #fff7ed;
  --accent: #2563eb;
  --border: #e5e7eb;
  --muted: #6b7280;
}
.main .block-container { padding-top: 1.5rem; max-width: 1180px; }
.hero {
  border: 1px solid var(--border);
  background: linear-gradient(135deg, #111827 0%, #7c2d12 55%, #f97316 130%);
  color: white;
  padding: 22px 26px;
  border-radius: 22px;
  box-shadow: 0 14px 40px rgba(17,24,39,.18);
  margin-bottom: 16px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2rem; letter-spacing: .2px; }
.hero p { margin: 0; opacity: .92; font-size: 1.02rem; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 7px 10px; border-radius: 999px;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  font-size: .85rem;
}
.metric-card {
  border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px;
  background: white; box-shadow: 0 6px 18px rgba(17,24,39,.04);
}
.metric-card .label { color: var(--muted); font-size: .82rem; }
.metric-card .value { font-size: 1.15rem; font-weight: 700; margin-top: 3px; }
.source-card {
  border: 1px solid var(--border); border-radius: 14px; padding: 13px 15px;
  background: #ffffff; margin-bottom: 12px;
}
.source-title { font-weight: 700; color: #111827; margin-bottom: 5px; }
.source-meta { color: var(--muted); font-size: .86rem; margin-bottom: 8px; }
.source-snippet { background: #f9fafb; border-radius: 10px; padding: 10px; }
.ingest-box {
  border: 1px dashed #fb923c; background: var(--primary-soft); padding: 14px;
  border-radius: 16px; margin-bottom: 12px;
}
.small-muted { color: var(--muted); font-size: .88rem; }
.stChatMessage { border-radius: 18px; }
</style>
"""


def highlight(text: str, query: str, limit: int = 1200) -> str:
    """Highlight query keywords inside a source snippet."""
    import re

    snippet = (text or "")[:limit]
    for tok in sorted(set(re.findall(r"\w+", (query or "").lower())), key=len, reverse=True):
        if len(tok) < 3:
            continue
        snippet = re.sub(f"({re.escape(tok)})", r"**\1**", snippet, flags=re.I)
    return snippet


def answer(question: str, history: list[dict] | None = None, top_k: int = 5) -> dict:
    """Generate an answer with simple follow-up support."""
    if history:
        previous_questions = " ".join(
            m["content"] for m in history[-6:] if m.get("role") == "user"
        )
        question_for_retrieval = (previous_questions + " " + question).strip()
    else:
        question_for_retrieval = question
    return generate_with_citation(question_for_retrieval, top_k=top_k)


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero">
          <h1>⚖️ {APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
          <div class="badge-row">
            <span class="badge">🔎 Hybrid Retrieval</span>
            <span class="badge">📚 Citation-first Answer</span>
            <span class="badge">🧠 Conversation Memory</span>
            <span class="badge">📥 Upload PDF/DOCX/MD/Web</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Corpus", "Legal + News + Uploads"),
        ("Retrieval", "Semantic + BM25 + RRF"),
        ("Fallback", "PageIndex-style local"),
        ("Bonus", "HyDE + TF-IDF + UI"),
    ]
    for col, (label, value) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def render_source_cards(sources: list[dict], query: str) -> None:
    if not sources:
        st.info("Không có source documents được trả về.")
        return
    for idx, source in enumerate(sources, 1):
        metadata = source.get("metadata", {}) or {}
        source_name = metadata.get("source") or metadata.get("path") or "unknown-source"
        doc_type = metadata.get("type", "unknown")
        retrieval_source = source.get("source", "hybrid")
        score = float(source.get("score", 0.0))
        st.markdown(
            f"""
            <div class="source-card">
              <div class="source-title">#{idx} — {source_name}</div>
              <div class="source-meta">type: <b>{doc_type}</b> · retrieval: <b>{retrieval_source}</b> · score: <b>{score:.3f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Xem đoạn nội dung / highlighted snippet", expanded=(idx == 1)):
            st.markdown(highlight(source.get("content", ""), query))
            st.caption(str(metadata))


def render_ingestion_panel() -> None:
    st.sidebar.markdown("## 📥 Nạp thêm tri thức")
    st.sidebar.markdown(
        "<div class='small-muted'>Upload file hoặc nhập URL. Hệ thống sẽ lưu, chuẩn hoá markdown và rebuild local DB để truy xuất ngay ở câu hỏi tiếp theo.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Upload PDF/DOCX/MD/TXT/JSON/HTML", expanded=True):
        uploaded_files = st.file_uploader(
            "Chọn file tài liệu",
            type=["pdf", "docx", "doc", "md", "txt", "json", "html", "htm"],
            accept_multiple_files=True,
            help="Tài liệu mới sẽ được lưu vào data/landing/uploads và data/standardized/uploads.",
        )
        if st.button("➕ Ingest uploaded files", use_container_width=True, disabled=not uploaded_files):
            progress = st.progress(0)
            logs = []
            for i, file in enumerate(uploaded_files or [], 1):
                try:
                    result = ingest_uploaded_file(file.name, file.getvalue(), refresh=False)
                    logs.append(f"✅ {file.name} → {result['markdown_path']}")
                except Exception as exc:  # noqa: BLE001 - UI should show user-friendly errors.
                    logs.append(f"❌ {file.name}: {exc}")
                progress.progress(i / max(1, len(uploaded_files)))
            stats = refresh_index()
            st.success(f"Đã ingest {len(uploaded_files)} file. DB hiện có {stats.get('chunks')} chunks.")
            for log in logs:
                st.write(log)

    with st.sidebar.expander("Thêm link web / bài báo", expanded=True):
        url = st.text_input("URL", placeholder="https://tuoitre.vn/...")
        if st.button("🌐 Ingest URL", use_container_width=True, disabled=not url.strip()):
            try:
                with st.spinner("Đang tải và xử lý trang web..."):
                    result = ingest_url(url.strip(), refresh=True)
                st.success(f"Đã ingest: {result['title']}")
                st.caption(result["markdown_path"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Không ingest được URL: {exc}")

    if st.sidebar.button("🔄 Rebuild local vector DB", use_container_width=True):
        stats = refresh_index()
        st.sidebar.success(f"Rebuilt: {stats.get('documents')} docs, {stats.get('chunks')} chunks")


def render_sidebar_settings() -> tuple[int, bool]:
    st.sidebar.markdown("## ⚙️ Cấu hình trả lời")
    top_k = st.sidebar.slider("Số source chunks", min_value=3, max_value=10, value=5, step=1)
    show_sources = st.sidebar.toggle("Hiển thị source documents", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Gợi ý câu hỏi")
    examples = [
        "Điều 249 quy định hình phạt tàng trữ ma túy như thế nào?",
        "Chi Dân và An Tây liên quan chuyên án ma túy nào?",
        "Các hình thức cai nghiện theo Luật Phòng chống ma túy 2021?",
        "Tóm tắt tài liệu tôi vừa upload và trích nguồn.",
    ]
    for example in examples:
        if st.sidebar.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
    return top_k, show_sources


def render_chat(top_k: int, show_sources: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    chat_tab, sources_tab, guide_tab = st.tabs(["💬 Chat", "📌 Sources", "📖 Hướng dẫn"])

    with chat_tab:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending = st.session_state.pop("pending_prompt", "") if "pending_prompt" in st.session_state else ""
        prompt = pending or st.chat_input("Hỏi về pháp luật ma túy, vụ việc nghệ sĩ, hoặc tài liệu vừa upload...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Đang truy xuất nguồn, rerank và tạo câu trả lời có citation..."):
                    result = answer(prompt, st.session_state.messages[:-1], top_k=top_k)
                st.markdown(result["answer"])
                st.session_state.last_sources = result.get("sources", [])
                if show_sources:
                    with st.expander("Nguồn đã sử dụng", expanded=True):
                        render_source_cards(st.session_state.last_sources, prompt)

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    with sources_tab:
        st.subheader("Source documents của lượt trả lời gần nhất")
        render_source_cards(st.session_state.last_sources, " ".join(m.get("content", "") for m in st.session_state.messages[-2:]))

    with guide_tab:
        st.markdown(
            """
            ### Cách dùng nhanh

            1. Hỏi trực tiếp về luật ma túy hoặc tin tức nghệ sĩ liên quan ma túy.
            2. Muốn bổ sung tri thức mới: dùng sidebar **Nạp thêm tri thức**.
            3. Upload `PDF/DOCX/MD/TXT/JSON/HTML` hoặc nhập link web.
            4. Sau khi ingest xong, hỏi tiếp — tài liệu mới đã được lưu và index trong `data/local_chunks.json`.

            ### Nơi dữ liệu mới được lưu

            - Raw files: `data/landing/uploads/`
            - Markdown chuẩn hóa: `data/standardized/uploads/`
            - Local DB/chunks: `data/local_chunks.json`
            """
        )


def main() -> None:
    if st is None:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    render_header()
    render_metrics()
    render_ingestion_panel()
    top_k, show_sources = render_sidebar_settings()
    st.markdown("")
    render_chat(top_k=top_k, show_sources=show_sources)


if __name__ == "__main__":
    main()