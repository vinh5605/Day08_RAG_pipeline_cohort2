# Dockerfile triển khai chatbot RAG (group_project/web_app.py) lên các nền
# tảng container — Hugging Face Spaces (SDK: Docker) hoặc Render (Web Service:
# Docker). Đáp ứng tiêu chí Bonus "Deploy chatbot online" (README.md gốc).
#
# Build & chạy cục bộ để kiểm thử trước khi deploy:
#   docker build -t druglaw-rag .
#   docker run --rm -p 8000:8000 --env-file .env druglaw-rag
# rồi mở http://localhost:8000
#
# Lưu ý: HF Spaces (Docker) mặc định expose cổng 7860 và đặt biến PORT=7860;
# Render cũng tự đặt PORT. web_app.py đọc HOST/PORT từ biến môi trường nên
# không cần sửa gì khi deploy — xem group_project/web_app.py.

FROM python:3.12-slim

# libgomp1: runtime cần cho onnxruntime/sentence-transformers (CPU inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải sẵn embedding model vào image lúc build — tránh việc mỗi lần container
# khởi động lại phải tải ~470MB từ Hugging Face Hub (làm chậm cold start và
# tốn băng thông trên các nền tảng free-tier).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

# HF Spaces (Docker SDK) mong đợi service lắng nghe ở cổng 7860; Render tự
# đặt biến PORT khi chạy. web_app.py ưu tiên biến môi trường PORT/HOST nếu
# có, nên hai giá trị mặc định dưới đây chỉ áp dụng khi nền tảng không tự đặt.
ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["python", "group_project/web_app.py"]
