# Báo cáo Đồ án Nhóm: Hệ thống RAG Pháp luật Ma tuý

## Danh sách thành viên và Phân công nhiệm vụ

Đồ án được thực hiện bởi nhóm gồm 6 thành viên. Các công việc trong quá trình xây dựng RAG Pipeline được phân chia đều đặn nhằm tối ưu thế mạnh của từng cá nhân.

| STT | Mã Sinh viên | Họ và Tên | Vai trò & Phân công nhiệm vụ |
| --- | --- | --- | --- |
| 1 | 2A202600864 | **Vũ Ngọc Vinh** | **Data Engineer & Processing:** Đảm nhiệm việc thu thập văn bản luật, crawl tin tức (Task 1 & 2). Thực hiện chuẩn hóa markdown, làm sạch dữ liệu (loại bỏ quảng cáo, boilerplate) và thiết lập luồng xử lý Data Ingestion (Task 3). |
| 2 | 2A202600948 | **Hoàng Hải** | **Vector DB & Embedding:** Phụ trách Task 4 & Task 5. Thực hiện chiến lược chia nhỏ văn bản (Chunking) theo ngữ nghĩa. Tích hợp Weaviate Cloud Database và cài đặt OpenAI Embeddings (`text-embedding-3-small`) để triển khai Semantic Search. |
| 3 | 2A202600632 | **Vũ Hải Dương** | **Lexical & Vectorless RAG:** Phát triển bộ máy tìm kiếm từ khóa cục bộ BM25 (Task 6). Đồng thời tích hợp và phát triển hệ thống PageIndex Vectorless API (Task 8) nhằm xây dựng hệ thống fallback mạnh mẽ khi Semantic Search không tìm thấy thông tin. |
| 4 | 2A202600684 | **Vũ Thành Lộc** | **Reranking & Hybrid Pipeline:** Chịu trách nhiệm cấu hình Jina Reranker AI (Task 7) để xếp hạng lại độ liên quan ngữ nghĩa. Xây dựng bộ điều phối Retrieval Pipeline (Task 9) áp dụng RRF (Reciprocal Rank Fusion) và HyDE (Hypothetical Document Embeddings). |
| 5 | 2A202600772 | **Vũ Tuấn Phương** | **Generation & Prompt Engineering:** Thiết kế luồng sinh câu trả lời bằng LLM (Ollama - Qwen) (Task 10). Tối ưu Prompt để đảm bảo trích dẫn (citation) chính xác nguồn. Viết cơ chế Extractive Fallback (hệ thống tự trích xuất khi LLM bị lỗi hoặc không có khả năng sinh). |
| 6 | 2A202600581 | **Nguyễn Danh Thành** | **Full-stack UI/UX & Evaluation:** Thiết kế và code giao diện Web App hiện đại (Glassmorphism, mượt mà và trực quan). Đồng thời phát triển bộ khung chấm điểm Offline RAG Evaluation (`eval_pipeline.py`) để đo lường tự động chất lượng của hệ thống. |

---

## Kiến trúc Hệ thống
Dự án được thiết kế theo cấu trúc mô-đun hoá cao với 10 task chính tuân thủ theo lý thuyết RAG (Retrieval-Augmented Generation). 
- **Trích xuất thông tin (Retrieval):** Sử dụng Hybrid Search, kết hợp giữa Vector Search (Weaviate + OpenAI) và Lexical Search (BM25).
- **Reranker (Xếp hạng):** Sử dụng `jina-reranker-v2-base-multilingual` qua API.
- **Generation (Sinh văn bản):** Triển khai cục bộ bằng Ollama (model Qwen).
- **High-availability (Dự phòng rủi ro):** Toàn bộ các kết nối ra ngoài (Cloud Vector DB, Jina AI, PageIndex) đều có Fallback quay về truy vấn cục bộ (JSON/TF-IDF) nhằm đảm bảo app không bao giờ bị dừng do mất mạng hoặc hết Quota API.

## Hướng dẫn cài đặt và chạy ứng dụng

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình API Keys:**
   Mở (hoặc tạo) file `.env` ở thư mục gốc và điền các API Key tương ứng:
   ```env
   OPENAI_API_KEY=sk-...
   JINA_API_KEY=jina_...
   PAGEINDEX_API_KEY=...
   WEAVIATE_URL=https://...
   WEAVIATE_API_KEY=...
   ```

3. **Chạy Web App:**
   ```bash
   python group_project/web_app.py
   ```
   Sau đó truy cập `http://localhost:8000` trên trình duyệt.

4. **Chạy Evaluation (Đánh giá RAG tự động):**
   ```bash
   python group_project/evaluation/eval_pipeline.py
   ```
   Kết quả đánh giá sẽ được ghi vào file `group_project/evaluation/results.md`.

## Triển khai Online (Deploy) — Bonus

Repo đã có sẵn `Dockerfile` (ở thư mục gốc) đóng gói toàn bộ chatbot
(`group_project/web_app.py`) thành 1 container — chạy được trên bất kỳ nền
tảng hỗ trợ Docker nào. `web_app.py` đọc `HOST`/`PORT` từ biến môi trường
(mặc định `127.0.0.1:8000` khi chạy local), nên không cần sửa code khi deploy.

### Cách 1 — Hugging Face Spaces (SDK: Docker)

1. Tạo Space mới tại huggingface.co/new-space → chọn **Docker** làm SDK.
2. Đẩy code lên Space (Spaces là 1 git repo):
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
3. Vào tab **Settings → Variables and secrets** của Space, thêm các biến cần
   thiết (xem bảng bên dưới) — ít nhất nên có `OLLAMA_BASE_URL` trỏ tới một
   Ollama server có thể truy cập từ Internet, hoặc `OPENAI_API_KEY` nếu dùng
   OpenAI; nếu bỏ trống cả hai, hệ thống vẫn chạy được nhờ cơ chế Extractive
   Fallback cục bộ (xem Task 10).
4. Spaces tự build `Dockerfile`, expose cổng `7860` và set biến `PORT=7860`
   — `web_app.py` sẽ tự bind đúng cổng đó.

### Cách 2 — Render (Web Service: Docker)

1. Push repo lên GitHub, sau đó vào render.com → **New → Blueprint** → chọn
   repo này. Render tự nhận diện file `render.yaml` ở gốc repo và build
   `Dockerfile`.
   *(Hoặc: New → Web Service → Runtime: Docker, không cần render.yaml.)*
2. Khai báo các biến môi trường trong tab **Environment** (xem bảng bên dưới).
   Render tự đặt `PORT`; `web_app.py` đọc giá trị này nên không cần khai báo.
3. Sau khi build xong, Render cấp 1 URL public dạng
   `https://druglaw-rag-chatbot.onrender.com`.

### Biến môi trường (Secrets) nên cấu hình khi deploy

| Biến | Bắt buộc? | Vai trò |
| --- | --- | --- |
| `OPENAI_API_KEY` | Không | Sinh câu trả lời chất lượng cao (Task 10) |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Không | Dùng LLM cục bộ Qwen qua Ollama (cần server Ollama public) |
| `JINA_API_KEY` | Không | Reranking (Task 7) — thiếu thì fallback MMR/cross-encoder cục bộ |
| `PAGEINDEX_API_KEY` | Không | Vectorless RAG (Task 8) — thiếu thì fallback structural search cục bộ |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | Không | Vector DB cloud (Task 4) — thiếu thì dùng ChromaDB cục bộ đã đóng gói sẵn trong image |

Nhờ kiến trúc "High-availability" (mọi API ngoài đều có fallback cục bộ —
xem mục Kiến trúc Hệ thống), chatbot vẫn hoạt động đầy đủ kể cả khi **không**
cấu hình bất kỳ biến nào ở trên — phù hợp để demo nhanh trên free-tier.