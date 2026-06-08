"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# Văn bản pháp luật đã thu thập (lưu trong data/landing/legal/):
#
#   1. luat-phong-chong-ma-tuy-2021.pdf
#      Luật Phòng, chống ma túy 2021 (Luật số 73/2021/QH14)
#      Tải trực tiếp bản gốc (có lớp text) từ Cổng TTĐT Chính phủ:
#      https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf
#
#   2. nghi-dinh-105-2021.docx
#      Nghị định 105/2021/NĐ-CP — hướng dẫn thi hành Luật PCMT 2021
#      Bản PDF "signed" trên chinhphu.vn chỉ là ảnh scan (không có lớp text,
#      MarkItDown/markitdown không trích xuất được nội dung), nên nội dung được
#      biên soạn lại (giữ nguyên văn) từ bản công khai trên Thư viện Pháp luật:
#      https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-496664.aspx
#
#   3. bo-luat-hinh-su-2015.docx
#      Bộ luật Hình sự 2015 (số 100/2015/QH13) — gồm Chương XX:
#      Các tội phạm về ma túy (Điều 247-259) và các điều khoản liên quan.
#      Cùng lý do như trên (PDF gốc là bản scan), nội dung lấy từ:
#      https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-hinh-su-2015-296661.aspx
#
#   4. nghi-dinh-57-2022-danh-muc-chat-ma-tuy.docx
#      Nghị định 57/2022/NĐ-CP — quy định các danh mục chất ma túy và tiền chất
#      https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-57-2022-ND-CP-danh-muc-chat-ma-tuy-va-tien-chat-527507.aspx
#
# Script tạo file (2)-(4) ở dạng .docx có thể chạy lại bằng download_legal_docs()
# bên dưới — fetch trang HTML (giữ nguyên cấu trúc Chương/Điều/Khoản), trích xuất
# nội dung và ghi ra .docx có lớp text thật (để Task 3 convert markdown thành công).

import re

import requests
from bs4 import BeautifulSoup
from docx import Document

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# (title, subtitle, source_url, output_filename)
DOCX_SOURCES = [
    (
        "Nghị định 105/2021/NĐ-CP",
        "Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Phòng, chống ma túy",
        "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-496664.aspx",
        "nghi-dinh-105-2021.docx",
    ),
    (
        "Bộ luật Hình sự 2015 (số 100/2015/QH13)",
        "Chương XX và các quy định liên quan đến tội phạm về ma túy",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-hinh-su-2015-296661.aspx",
        "bo-luat-hinh-su-2015.docx",
    ),
    (
        "Nghị định 57/2022/NĐ-CP",
        "Quy định các danh mục chất ma túy và tiền chất",
        "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-57-2022-ND-CP-danh-muc-chat-ma-tuy-va-tien-chat-527507.aspx",
        "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.docx",
    ),
]

# URL bản PDF gốc (có lớp text) của Luật Phòng, chống ma túy 2021
LAW_73_PDF_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf"

WHITESPACE_RE = re.compile(r"[ \t]+")


def _extract_paragraphs(html: str) -> list[str]:
    """Lấy text theo đúng cấu trúc Chương/Mục/Điều từ trang Thư viện Pháp luật."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div[id*=NoiDung]") or soup.select_one(".content1")
    if container is None:
        raise RuntimeError("Không tìm thấy khối nội dung văn bản trên trang")

    lines, seen = [], set()
    for el in container.find_all(["p", "li", "h1", "h2", "h3", "h4", "td"]):
        text = WHITESPACE_RE.sub(" ", el.get_text(" ", strip=True)).strip()
        if text and text not in seen:
            seen.add(text)
            lines.append(text)
    return lines


def _build_docx(title: str, subtitle: str, source_url: str, out_path: Path):
    html = requests.get(source_url, headers=HEADERS, timeout=60).text
    lines = _extract_paragraphs(html)

    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_heading(subtitle, level=2)
    doc.add_paragraph(f"Nguồn: {source_url}")
    doc.add_paragraph(
        "Văn bản được biên soạn lại (giữ nguyên văn) từ nội dung công khai trên "
        "Thư viện Pháp luật, phục vụ mục đích học tập / xây dựng RAG pipeline cá nhân."
    )
    doc.add_paragraph("=" * 40)
    for line in lines:
        if re.match(r"^(Chương|Mục|Phần)\s+[IVXLCDM\d]", line, re.IGNORECASE):
            doc.add_heading(line, level=2)
        elif re.match(r"^Điều\s+\d+", line):
            doc.add_heading(line, level=3)
        else:
            doc.add_paragraph(line)

    doc.save(out_path)
    print(f"  ✓ Đã tạo: {out_path} ({len(lines)} đoạn, {sum(len(l) for l in lines)} ký tự)")


def download_legal_docs():
    """Tải / dựng lại 4 văn bản pháp luật vào DATA_DIR (chạy lại được, idempotent)."""
    setup_directory()

    pdf_path = DATA_DIR / "luat-phong-chong-ma-tuy-2021.pdf"
    if not pdf_path.exists():
        resp = requests.get(LAW_73_PDF_URL, headers=HEADERS, timeout=60)
        pdf_path.write_bytes(resp.content)
        print(f"  ✓ Đã tải: {pdf_path}")
    else:
        print(f"  • Đã tồn tại: {pdf_path}")

    for title, subtitle, source_url, filename in DOCX_SOURCES:
        out_path = DATA_DIR / filename
        if out_path.exists():
            print(f"  • Đã tồn tại: {out_path}")
            continue
        _build_docx(title, subtitle, source_url, out_path)


if __name__ == "__main__":
    download_legal_docs()
