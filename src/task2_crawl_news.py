"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma túy.

Lưu ý lựa chọn thư viện:
    README khuyến nghị Crawl4AI (cần Playwright + tải browser binaries — khá nặng
    cho máy cá nhân). Ở đây dùng `requests` + `BeautifulSoup` — cùng nguyên lý
    "fetch HTML rồi trích nội dung chính sang markdown", chỉ khác là dùng
    selector CSS thủ công thay vì engine crawl tự động của Crawl4AI. Cách này
    nhẹ, không cần browser, và vẫn tạo ra output đúng format yêu cầu
    (1 file JSON / bài, có url, title, date_crawled, content).

Cài đặt:
    pip install requests beautifulsoup4 lxml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách bài báo về nghệ sĩ Việt Nam liên quan tới ma túy (VnExpress, Tuổi Trẻ).
# Mỗi entry khai báo CSS selector cho title/sapo/content vì mỗi báo có cấu trúc
# HTML khác nhau (không có 1 selector chung như Crawl4AI tự suy luận).
ARTICLE_URLS = [
    {
        "url": "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
        "site": "vnexpress",
        "title_sel": "h1.title-detail",
        "sapo_sel": "p.description",
        "content_sel": ".fck_detail",
    },
    {
        "url": "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
        "site": "vnexpress",
        "title_sel": "h1.title-detail",
        "sapo_sel": "p.description",
        "content_sel": ".fck_detail",
    },
    {
        "url": "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
        "site": "vnexpress",
        "title_sel": "h1.title-detail",
        "sapo_sel": "p.description",
        "content_sel": ".fck_detail",
    },
    {
        "url": "https://vnexpress.net/son-ngoc-minh-hot-boy-vpop-mot-thoi-vuong-lao-ly-5076096.html",
        "site": "vnexpress",
        "title_sel": "h1.title-detail",
        "sapo_sel": "p.description",
        "content_sel": ".fck_detail",
    },
    {
        "url": "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
        "site": "tuoitre",
        "title_sel": "h1.detail-title",
        "sapo_sel": "h2.detail-sapo",
        "content_sel": ".detail-content",
    },
    {
        "url": "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
        "site": "tuoitre",
        "title_sel": "h1.detail-title",
        "sapo_sel": "h2.detail-sapo",
        "content_sel": ".detail-content",
    },
]


def crawl_article(spec: dict) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "site": str,
            "content_markdown": str,
        }
    """
    resp = requests.get(spec["url"], headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    title_el = soup.select_one(spec["title_sel"])
    sapo_el = soup.select_one(spec["sapo_sel"])
    content_el = soup.select_one(spec["content_sel"])

    title = title_el.get_text(" ", strip=True) if title_el else "Unknown"
    sapo = sapo_el.get_text(" ", strip=True) if sapo_el else ""

    paragraphs = []
    if content_el is not None:
        for tag in content_el.select("script, style, .annonce-zone, .author_mail"):
            tag.decompose()
        for p in content_el.find_all(["p", "h2", "h3", "li"]):
            text = p.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

    body = "\n\n".join(paragraphs)
    content_markdown = f"# {title}\n\n**{sapo}**\n\n{body}" if sapo else f"# {title}\n\n{body}"

    return {
        "url": spec["url"],
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "site": spec["site"],
        "content_markdown": content_markdown,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS và lưu mỗi bài thành 1 file JSON."""
    setup_directory()

    for i, spec in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {spec['url']}")
        try:
            article = crawl_article(spec)
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            continue

        filename = f"article_{i:02d}_{spec['site']}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}  ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    crawl_all()
