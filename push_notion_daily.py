# push_notion_daily.py
import os, json, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID")

assert NOTION_TOKEN, "NOTION_TOKEN(.env)이 필요합니다."
assert NOTION_PARENT_PAGE_ID, "NOTION_PARENT_PAGE_ID(.env)이 필요합니다."

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def create_daily_page(date_str: str, articles: list[dict]):
    """상위 페이지 아래에 날짜별 서브 페이지 생성 후 기사 요약 삽입"""
    create_url = "https://api.notion.com/v1/pages"

    # 기사 블록 구성: 제목(H3) + 요약(문단) + 원문 링크(문단 링크)
    children = []
    for art in articles:
        short_title = art.get("short_title") or art.get("title") or "제목 없음"
        summary     = art.get("summary") or "요약 없음"
        url         = art.get("url")

        children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type":"text","text":{"content": short_title[:200]}}]}
        })
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type":"text","text":{"content": summary[:1800]}}]
            }
        })
        if url:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type":"text","text":{"content":"원문 보기","link":{"url": url}}}]
                }
            })

    data = {
        "parent": { "page_id": NOTION_PARENT_PAGE_ID },
        "properties": {
            "title": {
                "title": [{"type":"text","text":{"content": f"경제 뉴스 요약 - {date_str}"}}]
            }
        },
        "children": children
    }

    res = requests.post(create_url, headers=HEADERS, data=json.dumps(data))
    if res.status_code != 200:
        print("⚠️ Notion 업로드 실패:", res.status_code, res.text)
    else:
        print(f"✅ Notion 페이지 생성 완료: 경제 뉴스 요약 - {date_str}")

if __name__ == "__main__":
    # 1) 요약 결과 로드
    with open("econ_popular_multi.json","r",encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("top", [])

    # 2) 오늘 날짜 제목으로 생성
    today = datetime.now().strftime("%Y-%m-%d")
    create_daily_page(today, articles)