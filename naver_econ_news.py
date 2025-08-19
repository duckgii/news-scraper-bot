import os, re, json, time, requests, math
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs, urlunparse
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from rapidfuzz import fuzz

# GPT
from openai import OpenAI
from newspaper import Article

# ── 환경
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert NAVER_CLIENT_ID and NAVER_CLIENT_SECRET, "환경변수 NAVER_CLIENT_ID/SECRET 필요(.env)"
assert OPENAI_API_KEY, "환경변수 OPENAI_API_KEY 필요(.env)"

client = OpenAI(api_key=OPENAI_API_KEY)
KST = timezone(timedelta(hours=9))

# ── 키워드 세트 (여러 키워드로 수집 → 통합)
KEYWORDS = [
    "경제", "금리", "물가", "환율", "무역", "수출", "수입",
    "GDP", "성장률", "경기", "기준금리", "연준", "한국은행",
    "증시", "코스피", "코스닥", "주가", "지수", "채권", "국채", "수익률",
    "유가", "국제유가", "원자재", "금 가격", "구리", "니켈",
    "고용", "실업률", "임금", "부동산", "전세", "매매가격", "주택시장"
]

# ── 제외 키워드(잡음 줄이기)
EXCLUDE = [
    r"연예|아이돌|배우|예능|가요", r"스포츠|야구|축구|농구|배구",
    r"사건사고|범죄", r"날씨", r"게임|e스포츠",
    r"쿠폰|세일|이벤트", r"출시|리뷰", r"블로그"
]

# ── 정규화 도우미
TAG_RE = re.compile(r"<[^>]+>|&\w+;")
QUOTES = re.compile(r"[\"'“”‘’]")
PUNCT = re.compile(r"[^\w가-힣\s]")
NUMS = re.compile(r"\d{2,}")
DROP_QS_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","ncid","fbclid","igshid","ref"}

# === (추가) 투자 브리핑 생성 ===
def build_invest_prompt(articles):
    """
    articles: [{'rank', 'size', 'title', 'url', 'source', 'latest_pubDate', 'short_title', 'summary'}...]
    """
    bullets = []
    for a in articles:
        bullets.append(f"- [{a['rank']}] {a['short_title']}\n  요약: {a['summary']}\n  링크: {a['url']}")
    joined = "\n".join(bullets)

    prompt = f"""아래는 오늘의 경제 기사 요약 10개다. 내용을 종합해 **일반적 정보 제공용** '오늘의 투자 브리핑'을 작성하라.
단, 개인 맞춤 투자자문을 하지 말고, 과도한 확신/수익 보장을 금지한다. 구체 종목 추천은 피하고, 섹터/자산군/테마 수준으로 설명하라.
형식은 아래를 따르라(각 항목은 간결하게, 필요시 bullet로):

[오늘의 핵심 테마 3~5개]
[기회 요인]
[리스크 요인]
[관찰 포인트(경제지표/이벤트/발표)]
[전략적 시사점(포트폴리오·헤지 관점의 일반적 제안)]
[면책 문구 1줄: '본 내용은 정보 제공 목적이며 투자판단의 최종 책임은 본인에게 있습니다.']

기사 요약:
{joined}"""
    return prompt

def gpt_invest_brief(articles):
    prompt = build_invest_prompt(articles)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"너는 과도한 확신을 피하고 리스크를 명확히 밝히는 투자 브리핑 작성가다."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] 투자 브리핑 생성 실패: {e}")
        return "투자 브리핑 생성 실패(요약 데이터 부족 또는 API 제한)."

def norm_html(text:str)->str:
    return re.sub(r"\s+"," ", TAG_RE.sub(" ", text or "")).strip()

def clean_url(u: str) -> str:
    if not u: return u
    try:
        p = urlparse(u)
        qs = parse_qs(p.query)
        # 네이버 경유 링크면 원문으로 복원
        if "url" in qs and qs["url"]:
            return qs["url"][0]
        q = {k:v for k,v in qs.items() if k not in DROP_QS_KEYS}
        cleaned = p._replace(query="&".join(f"{k}={v[0]}" for k,v in q.items()))
        return urlunparse(cleaned)
    except Exception:
        return u

def to_kst(dt):
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)

def is_today_kst(dt):
    now = datetime.now(KST)
    start = now.replace(hour=0,minute=0,second=0,microsecond=0)
    end = start + timedelta(days=1)
    return start <= dt < end

def match_any(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def norm_title(s: str) -> str:
    s = TAG_RE.sub(" ", s or "")
    s = QUOTES.sub(" ", s)
    s = NUMS.sub("#", s)
    s = PUNCT.sub(" ", s)
    s = re.sub(r"\s+"," ", s).strip().lower()
    return s

# ── API 호출 (키워드별 + 페이지네이션)
def fetch_news_for_keyword(keyword, per_page=50, max_pages=2, sleep_sec=0.2):
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    base = "https://openapi.naver.com/v1/search/news.json"
    collected = []
    for p in range(max_pages):
        start = 1 + p*per_page
        params = {"query": keyword, "display": per_page, "start": start, "sort": "date"}
        r = requests.get(base, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items: break
        collected.extend(items)
        time.sleep(sleep_sec)     # 쿼터 보호
    return collected

def fetch_all_keywords(keywords, per_page=50, max_pages=2):
    all_items = []
    for kw in keywords:
        try:
            all_items.extend(fetch_news_for_keyword(kw, per_page=per_page, max_pages=max_pages))
        except Exception as e:
            print(f"[WARN] '{kw}' 요청 실패: {e}")
    return all_items

# ── 필터링(‘오늘’ + 제외어), 그리고 URL 정규화/기본 중복 제거
def filter_and_normalize(raw_items):
    out = []
    seen_urls = set()
    for it in raw_items:
        title = norm_html(it.get("title"))
        desc  = norm_html(it.get("description"))
        url   = clean_url(it.get("originallink") or it.get("link") or "")
        if not url: 
            continue

        # 발행일 → 오늘만
        try:
            pub_dt = parsedate_to_datetime(it.get("pubDate"))
            pub_kst = to_kst(pub_dt)
            if not is_today_kst(pub_kst):
                continue
        except Exception:
            continue

        # 제외 키워드
        text = f"{title} {desc}"
        if match_any(EXCLUDE, text):
            continue

        # URL 중복 제거
        if url in seen_urls:
            continue
        seen_urls.add(url)

        out.append({
            "title": title,
            "desc": desc,
            "url": url,
            "source": urlparse(url).netloc,
            "pubDate": pub_kst.isoformat()
        })
    # 최신순
    out.sort(key=lambda x: x["pubDate"], reverse=True)
    return out

# ── 제목 유사 중복 제거(사전)
def dedup_by_title(items, threshold=93):
    kept, canon = [], []
    for it in items:
        t = norm_title(it["title"])
        if any(fuzz.token_set_ratio(t, c) >= threshold for c in canon):
            continue
        kept.append(it); canon.append(t)
    return kept

# ── 군집화(인기 추정)
def cluster_by_title(items, eps=0.65, min_samples=2):
    if not items: return []
    titles = [it["title"] for it in items]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4))
    X = vec.fit_transform(titles)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(X)

    clusters = {}
    for idx, label in enumerate(labels):
        key = label if label != -1 else f"single-{idx}"
        clusters.setdefault(key, []).append(idx)

    groups = []
    for key, idxs in clusters.items():
        members = [items[i] for i in idxs]
        members_sorted = sorted(members, key=lambda x: x["pubDate"], reverse=True)
        rep = members_sorted[0]
        groups.append({
            "cluster_id": str(key),
            "size": len(members),
            "rep_title": rep["title"],
            "rep_url": rep["url"],
            "rep_source": rep["source"],
            "latest_pubDate": rep["pubDate"],
            "members": members_sorted,  # 요약시 참고 가능
        })
    groups.sort(key=lambda g:(g["size"], g["latest_pubDate"]), reverse=True)
    return groups

# ── TopN with diversity (마지막 안전장치)
def top_n_with_diversity(groups, items, n=10, sim_threshold=93):
    picked, seen = [], []
    def add(rep):
        t = norm_title(rep["rep_title"])
        if any(fuzz.token_set_ratio(t, s) >= sim_threshold for s in seen):
            return False
        picked.append(rep); seen.append(t); return True

    for g in groups:
        if len(picked) >= n: break
        add(g)

    # 부족하면 싱글 기사로 채우기
    if len(picked) < n:
        used = {p["rep_url"] for p in picked}
        for it in items:
            if it["url"] in used: 
                continue
            fake = {
                "cluster_id": "single-fill",
                "size": 1,
                "rep_title": it["title"],
                "rep_url": it["url"],
                "rep_source": it["source"],
                "latest_pubDate": it["pubDate"]
            }
            if add(fake): 
                used.add(it["url"])
            if len(picked) >= n:
                break
    return picked[:n]

# ── 본문 추출
def extract_article_text(url):
    try:
        art = Article(url, language="ko")
        art.download()
        art.parse()
        text = (art.text or "").strip()
        # newspaper가 짧게 주면 대비
        return text
    except Exception as e:
        print(f"[WARN] 본문 추출 실패: {url} ({e})")
        return ""

# ── GPT 요약 (짧은 제목 + 5문장), 긴 본문은 map→reduce 방식
def gpt_summarize(text, url, chunk_chars=2500):
    if not text.strip():
        return {"short_title": "본문 추출 실패", "summary": "본문 없음"}

    # 1) 긴 본문 쪼개기
    chunks = []
    t = text
    while t:
        chunks.append(t[:chunk_chars])
        t = t[chunk_chars:]

    # 2) chunk별 2~3문장 요약
    partial_summaries = []
    for i, ch in enumerate(chunks):
        prompt = f"""아래 기사 일부를 2~3문장으로 요약해줘.
- 과장 금지, 사실 위주
- 수치/기관명/날짜 보존

부분 {i+1}/{len(chunks)}:
{ch}"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"너는 한국어 경제 기사 요약가야."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
            )
            partial_summaries.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[WARN] 부분 요약 실패: {e}")
            partial_summaries.append("")

    # 3) 합성 요약(최종): 짧은 제목 + 정확히 5문장
    merge_input = "\n\n".join(ps for ps in partial_summaries if ps)
    if not merge_input:
        merge_input = text[:2000]

    final_prompt = f"""아래 내용을 바탕으로 기사를 최종 요약해줘.
형식:
1) 맨 첫 줄: 아주 짧은 제목(한 줄)
2) 그 다음: 한국어로 정확히 5문장 요약

출처 URL: {url}
요약 재료:
{merge_input}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"너는 한국어 경제 기사 요약가야."},
                {"role":"user","content":final_prompt}
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        lines = [ln for ln in content.split("\n") if ln.strip()]
        if not lines:
            return {"short_title": "요약 실패", "summary": ""}
        short_title = lines[0].strip()
        summary = "\n".join(lines[1:]).strip()
        return {"short_title": short_title, "summary": summary}
    except Exception as e:
        print(f"[WARN] 최종 요약 실패: {e}")
        return {"short_title": "요약 실패", "summary": ""}


# === 투자 브리핑(구조화 JSON) 프롬프트 생성 ===
def build_invest_prompt_v2(articles):
    # GPT에 넘길 요약만 간결 JSON으로 정리
    items = [
        {
            "rank": a["rank"],
            "title": a["short_title"] or a["title"],
            "summary": a["summary"],
            "url": a["url"]
        } for a in articles
    ]
    seed = json.dumps(items, ensure_ascii=False)

    # ★ 포맷 강제: 자산군 플레이북 + 이슈 설명서(학습형) + 관찰 포인트
    return f"""
너는 '일반 정보 제공용' 매크로 전략 요약가다. 아래 한국어 기사 요약 10개만 근거로 삼아,
과도한 확신/수익보장/개별종목 추천을 금지하고, 교육적 설명을 곁들인 '투자 브리핑'을 JSON으로 생성하라.
반드시 'JSON만' 출력하고, 키 이외의 문구/서론은 금지한다.

기사요약목록(JSON):
{seed}

출력 스키마(JSON):
{{
  "themes": ["오늘의 핵심 테마(짧게)"],
  "issues": [
    {{
      "title": "이슈 이름(짧게)",
      "what_happened": "무슨 일이 있었는가(팩트 요약)",
      "why_it_matters": "왜 중요한가(경제적 메커니즘 설명)",
      "transmission": ["전파 경로(금리→환율→수출 등)"],
      "monitor": ["수치/지표/발표/레벨 등 관찰 포인트"],
      "scenarios": {{
        "base": {{ "prob": 0.5, "text": "기본 시나리오 설명" }},
        "bull": {{ "prob": 0.25, "text": "호조 시나리오" }},
        "bear": {{ "prob": 0.25, "text": "부진 시나리오" }}
      }},
      "actions": ["행동 가이드(일반론: if/then 규칙; 과도한 확신 금지, 손절/헷지 원칙 포함)"]
    }}
  ],
  "playbook": {{
    "equities": {{
      "bias": "상승/중립/하락 중 하나(근거 한줄)",
      "tactical_1_2w": ["1~2주 전술(섹터/요인 위주, 종목 언급 금지)"],
      "strategic_1_3m": ["1~3개월 전략"],
      "triggers": ["매수/감축/헤지 트리거(수치/레벨)"],
      "hedges": ["헤지·분산 아이디어(일반론)"],
      "invalidations": ["시나리오 무효화 조건"]
    }},
    "bonds": {{
      "bias": "", "tactical_1_2w": [], "strategic_1_3m": [],
      "triggers": [], "hedges": [], "invalidations": []
    }},
    "fx_dollar": {{
      "bias": "", "tactical_1_2w": [], "strategic_1_3m": [],
      "triggers": [], "hedges": [], "invalidations": []
    }},
    "commodities": {{
      "oil": {{"bias":"", "notes":["…"]}},
      "gold": {{"bias":"", "notes":["…"]}},
      "industrial_metals": {{"bias":"", "notes":["…"]}}
    }}
  }},
  "calendar": ["다가오는 일정/지표(예: FOMC, CPI, 고용 등)"],
  "disclaimer": "본 내용은 정보 제공 목적이며 투자 판단의 최종 책임은 본인에게 있습니다."
}}
주의:
- 기사요약목록 내 정보로만 추론하라(할루시네이션 금지).
- 개별 종목/티커, 수익 보장 문장 금지. 섹터·자산군 단위로만.
- 수치·레벨·조건은 '가정적 예시'가 아닌 기사요약에서 합리적으로 추정되는 범위로.
- 한국어로 작성.
"""

# === GPT 호출: JSON 파싱하여 dict 반환 ===
def gpt_invest_brief_structured(articles):
    prompt = build_invest_prompt_v2(articles)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"사실 기반·보수적 톤의 매크로 전략 브리핑을 만드는 조력자."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except Exception:
            # JSON 파싱 실패 시 원문을 담아 반환
            return {"raw_text": raw, "disclaimer": "본 내용은 정보 제공 목적이며 투자 판단의 최종 책임은 본인에게 있습니다."}
    except Exception as e:
        print(f"[WARN] 투자 브리핑(JSON) 생성 실패: {e}")
        return {"raw_text": "투자 브리핑 생성 실패", "disclaimer": "본 내용은 정보 제공 목적이며 투자 판단의 최종 책임은 본인에게 있습니다."}
    
# ── 메인
if __name__ == "__main__":
    TARGET = 10

    # 1) 키워드×페이지네이션 수집
    raw = fetch_all_keywords(KEYWORDS, per_page=50, max_pages=2)

    # 2) 오늘 기사 + 제외어 필터 + URL 정규화/중복 제거
    base = filter_and_normalize(raw)

    # 3) 제목 유사 사전 dedup
    base = dedup_by_title(base, threshold=93)

    # 4) 군집화로 '인기' 추정
    groups = cluster_by_title(base, eps=0.65, min_samples=2)

    # 5) Top10 보장(+유사제목 차단)
    top = top_n_with_diversity(groups, base, n=TARGET, sim_threshold=93)

    # 6) 각 기사 본문 → GPT 요약
    results = []
    for i, g in enumerate(top, start=1):
        url = g["rep_url"]
        title = g["rep_title"]
        print(f"[{i}/{TARGET}] 요약 중: {title}")
        text = extract_article_text(url)
        summary = gpt_summarize(text, url)
        results.append({
            "rank": i,
            "size": g["size"],
            "title": title,
            "url": url,
            "source": g["rep_source"],
            "latest_pubDate": g["latest_pubDate"],
            "short_title": summary["short_title"],
            "summary": summary["summary"]
        })
        # 친절한 레이트 컨트롤(선택)
        time.sleep(0.2)

        # 7) (변경) 투자 브리핑: 구조화 JSON
    invest_struct = gpt_invest_brief_structured(results)

    # 8) 저장
    out = {
        "as_of": datetime.now(KST).isoformat(),
        "total_raw": len(raw),
        "total_after_filter": len(base),
        "cluster_count": len(groups),
        "returned": len(results),
        "invest_structured": invest_struct,   # ★ 추가(구조화)
        "top": results
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    with open("econ_popular_multi.json","w",encoding="utf-8") as f:
        f.write(text)
    print("저장 완료 → econ_popular_multi.json")