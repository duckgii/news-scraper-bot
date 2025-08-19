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

def _txt(text):  # 안전하게 자르는 헬퍼
    return (text or "")[:1800]

def _bullets(items):
    return [{"object":"block","type":"bulleted_list_item",
             "bulleted_list_item":{"rich_text":[{"type":"text","text":{"content": _txt(x)}}]}} for x in items or []]

def create_daily_page(date_str: str, articles: list[dict], invest_struct: dict | None = None):
    create_url = "https://api.notion.com/v1/pages"
    children = []

    # ===== 상단: 오늘의 투자 브리핑(구조화) =====
    if invest_struct:
        # 헤더
        children.append({"object":"block","type":"heading_2",
                         "heading_2":{"rich_text":[{"type":"text","text":{"content":"오늘의 투자 브리핑"}}]}})
        # 핵심 테마
        themes = (invest_struct or {}).get("themes", [])
        if themes:
            children.append({"object":"block","type":"heading_3",
                             "heading_3":{"rich_text":[{"type":"text","text":{"content":"핵심 테마"}}]}})
            children.extend(_bullets(themes))

        # 이슈 설명서(토글로 학습형)
        issues = (invest_struct or {}).get("issues", [])
        if issues:
            children.append({"object":"block","type":"heading_3",
                             "heading_3":{"rich_text":[{"type":"text","text":{"content":"이슈 설명서"}}]}})
            for iss in issues:
                toggle_children = []
                if iss.get("what_happened"):
                    toggle_children.append({"object":"block","type":"paragraph",
                        "paragraph":{"rich_text":[{"type":"text","text":{"content":"무슨 일: "+_txt(iss["what_happened"])}}]}})
                if iss.get("why_it_matters"):
                    toggle_children.append({"object":"block","type":"paragraph",
                        "paragraph":{"rich_text":[{"type":"text","text":{"content":"의미: "+_txt(iss["why_it_matters"])}}]}})
                if iss.get("transmission"):
                    toggle_children.append({"object":"block","type":"paragraph",
                        "paragraph":{"rich_text":[{"type":"text","text":{"content":"전파 경로"}}]}})
                    toggle_children.extend(_bullets(iss["transmission"]))
                if iss.get("monitor"):
                    toggle_children.append({"object":"block","type":"paragraph",
                        "paragraph":{"rich_text":[{"type":"text","text":{"content":"관찰 포인트"}}]}})
                    toggle_children.extend(_bullets(iss["monitor"]))
                if iss.get("scenarios"):
                    sc = iss["scenarios"]
                    for name in ["base","bull","bear"]:
                        node = sc.get(name)
                        if node:
                            toggle_children.append({"object":"block","type":"paragraph",
                                "paragraph":{"rich_text":[{"type":"text","text":{"content":f"{name.upper()}({node.get('prob','')}) - { _txt(node.get('text','')) }"}}]}})
                if iss.get("actions"):
                    toggle_children.append({"object":"block","type":"paragraph",
                        "paragraph":{"rich_text":[{"type":"text","text":{"content":"행동 가이드(일반론)"}}]}})
                    toggle_children.extend(_bullets(iss["actions"]))

                children.append({
                    "object":"block","type":"toggle",
                    "toggle":{"rich_text":[{"type":"text","text":{"content":_txt(iss.get("title","이슈") )}}], "children":toggle_children}
                })

        # 자산군 플레이북
        play = (invest_struct or {}).get("playbook", {})
        if play:
            children.append({"object":"block","type":"heading_3",
                             "heading_3":{"rich_text":[{"type":"text","text":{"content":"자산군 플레이북"}}]}})

            def _section(title, data):
                block = [{"object":"block","type":"paragraph",
                          "paragraph":{"rich_text":[{"type":"text","text":{"content":"바이어스: "+_txt(data.get("bias",""))}}]}}]
                for k,label in [("tactical_1_2w","전술(1~2주)"),("strategic_1_3m","전략(1~3개월)"),
                                ("triggers","트리거"),("hedges","헤지/분산"),("invalidations","무효화 조건")]:
                    if data.get(k):
                        block.append({"object":"block","type":"paragraph",
                                      "paragraph":{"rich_text":[{"type":"text","text":{"content":label}}]}})
                        block.extend(_bullets(data[k]))
                return {
                    "object":"block","type":"toggle",
                    "toggle":{"rich_text":[{"type":"text","text":{"content":title}}], "children":block}
                }

            if play.get("equities"):   children.append(_section("주식",     play["equities"]))
            if play.get("bonds"):      children.append(_section("채권",     play["bonds"]))
            if play.get("fx_dollar"):  children.append(_section("달러/FX",  play["fx_dollar"]))

            # 원자재 묶음
            com = play.get("commodities", {})
            if com:
                com_children = []
                for k,label in [("oil","원유"),("gold","금"),("industrial_metals","산업금속")]:
                    if com.get(k):
                        node = com[k]
                        notes = node.get("notes", [])
                        sub = [{"object":"block","type":"paragraph",
                                "paragraph":{"rich_text":[{"type":"text","text":{"content":"바이어스: "+_txt(node.get("bias",""))}}]}}]
                        if notes: sub.extend(_bullets(notes))
                        com_children.append({"object":"block","type":"toggle",
                                             "toggle":{"rich_text":[{"type":"text","text":{"content":label}}], "children":sub}})
                children.append({"object":"block","type":"toggle",
                                 "toggle":{"rich_text":[{"type":"text","text":{"content":"원자재"}}], "children":com_children}})

        # 관찰 포인트/면책
        cal = (invest_struct or {}).get("calendar", [])
        if cal:
            children.append({"object":"block","type":"heading_3",
                             "heading_3":{"rich_text":[{"type":"text","text":{"content":"관찰 포인트"}}]}})
            children.extend(_bullets(cal))

        disclaimer = (invest_struct or {}).get("disclaimer")
        if disclaimer:
            children.append({"object":"block","type":"callout","callout":{
                "icon":{"emoji":"⚠️"},
                "rich_text":[{"type":"text","text":{"content":_txt(disclaimer)}}]
            }})
        children.append({"object":"block","type":"divider","divider":{}})

    # ===== 이하: 기사 10개(기존 섹션) =====
    for art in articles:
        short_title = art.get("short_title") or art.get("title") or "제목 없음"
        summary     = art.get("summary") or "요약 없음"
        url         = art.get("url")

        children.append({"object":"block","type":"heading_3",
                         "heading_3":{"rich_text":[{"type":"text","text":{"content": short_title[:200]}}]}})
        children.append({"object":"block","type":"paragraph",
                         "paragraph":{"rich_text":[{"type":"text","text":{"content": _txt(summary)}}]}})
        if url:
            children.append({"object":"block","type":"paragraph",
                             "paragraph":{"rich_text":[{"type":"text","text":{"content":"원문 보기","link":{"url": url}}}]}})

    data = {
        "parent": {"page_id": NOTION_PARENT_PAGE_ID},
        "properties": {"title": {"title": [{"type":"text","text":{"content": f"경제 뉴스 요약 - {date_str}"}}]}},
        "children": children
    }
    res = requests.post(create_url, headers=HEADERS, data=json.dumps(data))
    if res.status_code != 200:
        print("⚠️ Notion 업로드 실패:", res.status_code, res.text)
    else:
        print(f"✅ Notion 페이지 생성 완료: 경제 뉴스 요약 - {date_str}")

if __name__ == "__main__":
    with open("econ_popular_multi.json","r",encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("top", [])
    invest_struct = data.get("invest_structured")   # ★ 추가
    today = datetime.now().strftime("%Y-%m-%d")

    create_daily_page(today, articles, invest_struct)