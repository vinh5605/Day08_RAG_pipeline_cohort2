"""Minimal non-Streamlit web app for the RAG chatbot.

Run:
    python web_app.py
Then open:
    http://localhost:8000

Features:
- Simple custom HTML page: web name, chat box, upload box, URL ingest box.
- Backend APIs for chat, upload, URL ingest.
- Uses src.task10_generation, which calls Ollama/Qwen if available.
"""

from __future__ import annotations

from pathlib import Path
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
import cgi

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.task10_generation import generate_with_citation, OLLAMA_MODEL, _ollama_available
from src.ingestion import ingest_uploaded_file, ingest_url, refresh_index

# Cloud platforms (Render / Hugging Face Spaces Docker / ...) chỉ định cổng
# qua biến môi trường PORT và yêu cầu bind 0.0.0.0 (không phải 127.0.0.1) để
# truy cập được từ bên ngoài container. Mặc định vẫn là localhost:8000 cho
# máy cá nhân — không cần đặt biến môi trường khi chạy local.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

INDEX_HTML = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DrugLaw Intel Chatbot</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    body { 
        margin: 0; 
        font-family: 'Inter', sans-serif; 
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%); 
        color: #1e293b; 
        min-height: 100vh;
    }
    header { 
        background: rgba(255, 255, 255, 0.8); 
        backdrop-filter: blur(12px);
        color: #0f172a; 
        padding: 20px 32px; 
        border-bottom: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    header h1 { margin: 0; font-size: 28px; font-weight: 700; color: #1e293b; background: linear-gradient(to right, #2563eb, #7c3aed); -webkit-background-clip: text; color: transparent; }
    header p { margin: 8px 0 0; opacity: .8; font-size: 15px; color: #475569; }
    .wrap { max-width: 1200px; margin: 30px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 360px; gap: 24px; }
    .card { 
        background: rgba(255, 255, 255, 0.85); 
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.6); 
        border-radius: 20px; 
        padding: 24px; 
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01); 
    }
    .card h2 { margin-top: 0; font-size: 20px; font-weight: 600; margin-bottom: 20px; color: #0f172a; display: flex; align-items: center; gap: 8px;}
    #chat { 
        height: 550px; 
        overflow-y: auto; 
        border: 1px solid #e2e8f0; 
        border-radius: 16px; 
        padding: 20px; 
        background: rgba(248, 250, 252, 0.7); 
        display: flex; flex-direction: column; gap: 16px;
        scroll-behavior: smooth;
    }
    #chat::-webkit-scrollbar { width: 6px; }
    #chat::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
    .msg { 
        padding: 14px 18px; 
        border-radius: 18px; 
        line-height: 1.5; 
        white-space: pre-wrap; 
        font-size: 15px;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .user { 
        background: linear-gradient(135deg, #2563eb, #3b82f6); 
        color: white; 
        align-self: flex-end; 
        border-bottom-right-radius: 4px;
    }
    .bot { 
        background: white; 
        color: #1e293b; 
        align-self: flex-start; 
        border-bottom-left-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    .row { display: flex; gap: 12px; margin-top: 20px; align-items: center;}
    input[type=text] { 
        flex: 1; 
        padding: 16px; 
        border-radius: 14px; 
        border: 1px solid #cbd5e1; 
        font-family: inherit; font-size: 15px;
        transition: all 0.2s ease;
        background: rgba(255,255,255,0.9);
    }
    input[type=text]:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15); }
    button { 
        border: 0; 
        background: linear-gradient(135deg, #2563eb, #1d4ed8); 
        color: white; 
        border-radius: 14px; 
        padding: 16px 24px; 
        cursor: pointer; 
        font-weight: 600; 
        font-size: 15px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3); }
    button:active { transform: translateY(1px); box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2); }
    button.secondary { 
        background: white; 
        color: #334155; 
        border: 1px solid #cbd5e1;
        width: 100%; 
        margin-top: 10px; 
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    button.secondary:hover { background: #f8fafc; border-color: #94a3b8; }
    .source { 
        border-top: 1px solid #f1f5f9; 
        padding-top: 12px; 
        margin-top: 12px; 
        font-size: 13px; 
        color: #64748b;
        background: #f8fafc;
        padding: 10px;
        border-radius: 8px;
    }
    .source b { color: #334155; }
    .small { color: #64748b; font-size: 13px; line-height: 1.5; margin-bottom: 16px; display: block;}
    h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; margin: 24px 0 10px 0; }
    input[type=file] { 
        width: 100%; 
        padding: 10px;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        font-size: 14px;
        color: #475569;
        cursor: pointer;
    }
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 8px 12px; border-radius: 999px;
        background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;
        font-weight: 500; font-size: 13px; margin-top: 10px; width: 100%; justify-content: center;
    }
    @media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } #chat { height: 480px; } }
  </style>
</head>
<body>
  <header>
    <h1>⚖️ DrugLaw Intel</h1>
    <p>Hỏi đáp pháp luật ma túy & tin tức mở rộng với AI</p>
  </header>
  <main class="wrap">
    <section class="card">
      <h2>✨ Trợ lý Chatbot</h2>
      <div id="chat">
        <div class="msg bot">Chào bạn! Tôi là trợ lý AI chuyên về luật phòng chống ma túy. Bạn cần tìm hiểu thông tin gì?</div>
      </div>
      <div class="row">
        <input id="question" type="text" placeholder="Nhập câu hỏi... (VD: Điều 249 quy định hình phạt gì?)" />
        <button onclick="sendChat()">Gửi đi</button>
      </div>
    </section>
    <aside class="card">
      <h2>📚 Quản lý Tri thức</h2>
      <span class="small">Hệ thống sẽ lưu dữ liệu vào cơ sở dữ liệu để tìm kiếm ở các câu hỏi sau.</span>
      
      <h3>Upload File</h3>
      <input id="file" type="file" accept=".pdf,.docx,.doc,.md,.txt,.json,.html,.htm" />
      <button class="secondary" onclick="uploadFile()">Upload & Đưa vào DB</button>
      
      <h3>Nhập Link Web</h3>
      <input id="url" type="text" placeholder="https://..." />
      <button class="secondary" onclick="ingestUrl()">Phân tích URL</button>
      
      <h3>Trạng thái Hệ thống</h3>
      <div id="status" class="status-badge">Sẵn sàng.</div>
    </aside>
  </main>
<script>
const chat = document.getElementById('chat');
const history = [];
function addMsg(role, text, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  div.textContent = text;
  if (sources && sources.length) {
    const src = document.createElement('div'); src.className='source';
    src.innerHTML = '<b>📌 Nguồn tham khảo:</b><br>' + sources.map((s,i)=> `${i+1}. ${s.source || 'unknown'} (Độ tin cậy: ${Number(s.score||0).toFixed(2)})`).join('<br>');
    div.appendChild(src);
  }
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
}
async function sendChat() {
  const input = document.getElementById('question');
  const q = input.value.trim(); if (!q) return;
  input.value = ''; addMsg('user', q); history.push({role:'user', content:q});
  
  const botDiv = document.createElement('div');
  botDiv.className = 'msg bot'; botDiv.innerHTML = '<span style="opacity:0.6">Đang suy nghĩ...</span>';
  chat.appendChild(botDiv); chat.scrollTop = chat.scrollHeight;
  
  try {
    const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q, history})});
    const data = await res.json();
    botDiv.remove();
    addMsg('bot', data.answer || data.error || 'Không có câu trả lời.', data.sources || []);
    history.push({role:'assistant', content:data.answer || ''});
  } catch(e) { botDiv.remove(); addMsg('bot', 'Lỗi: '+e); }
}
async function uploadFile() {
  const f = document.getElementById('file').files[0]; if (!f) return alert('Vui lòng chọn file trước');
  const fd = new FormData(); fd.append('file', f);
  const status = document.getElementById('status');
  status.textContent='Đang xử lý file...'; status.style.background='#fef3c7'; status.style.color='#d97706';
  
  const res = await fetch('/api/upload', {method:'POST', body:fd}); const data = await res.json();
  status.textContent = data.error ? ('Lỗi: '+data.error) : ('Đã thêm: '+data.title);
  status.style.background = data.error ? '#fee2e2' : '#ecfdf5';
  status.style.color = data.error ? '#dc2626' : '#059669';
}
async function ingestUrl() {
  const url = document.getElementById('url').value.trim(); if (!url) return alert('Vui lòng nhập URL');
  const status = document.getElementById('status');
  status.textContent='Đang đọc URL...'; status.style.background='#fef3c7'; status.style.color='#d97706';
  
  const res = await fetch('/api/ingest_url', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})});
  const data = await res.json();
  status.textContent = data.error ? ('Lỗi: '+data.error) : ('Đã thêm: '+data.title);
  status.style.background = data.error ? '#fee2e2' : '#ecfdf5';
  status.style.color = data.error ? '#dc2626' : '#059669';
}
document.getElementById('question').addEventListener('keydown', e => { if(e.key==='Enter') sendChat(); });
fetch('/api/status').then(r=>r.json()).then(d=>{
    const status = document.getElementById('status');
    status.textContent = `Model: ${d.ollama_available ? d.model : 'Rule-based'}`;
});
</script>
</body>
</html>"""


def json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/status":
            json_response(self, {
                "ollama_available": _ollama_available(),
                "model": OLLAMA_MODEL,
                "message": "Ollama/Qwen sẵn sàng" if _ollama_available() else "Chưa thấy Ollama. Cài Ollama và pull model để trả lời mượt hơn.",
            })
            return
        self.send_error(404)

    def do_POST(self):
        try:
            if self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                question = data.get("question", "")
                result = generate_with_citation(question, top_k=2)
                sources = []
                for s in result.get("sources", [])[:2]:
                    md = s.get("metadata", {}) or {}
                    sources.append({
                        "source": md.get("source") or md.get("path") or "unknown",
                        "score": s.get("score", 0),
                        "type": md.get("type", "unknown"),
                    })
                json_response(self, {"answer": result.get("answer", ""), "sources": sources, "llm": result.get("llm")})
                return

            if self.path == "/api/ingest_url":
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                result = ingest_url(data.get("url", ""), refresh=True)
                json_response(self, result)
                return

            if self.path == "/api/upload":
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
                file_item = form["file"] if "file" in form else None
                if file_item is None or not getattr(file_item, "filename", ""):
                    json_response(self, {"error": "No file uploaded"}, status=400)
                    return
                content = file_item.file.read()
                result = ingest_uploaded_file(file_item.filename, content, refresh=True)
                json_response(self, result)
                return

            self.send_error(404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def log_message(self, fmt, *args):
        print("[web] " + fmt % args)


if __name__ == "__main__":
    refresh_index()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    print(f"Ollama available: {_ollama_available()} | model: {OLLAMA_MODEL}")
    server.serve_forever()