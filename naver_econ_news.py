import os, re, json, time, requests
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs, urlunparse
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from rapidfuzz import fuzz

# ── 환경
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
assert NAVER_CLIENT_ID and NAVER_CLIENT_SECRET, "환경변수 NAVER_CLIENT_ID/SECRET 필요(.env)"

KST = timezone(timedelta(hours=9))

# ── 키워드 세트: '경제' + 세부 토픽들 (원하면 더 넣고 빼세요)
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
def fetch_news_for_keyword(keyword, per_page=100, max_pages=2, sleep_sec=0.2):
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
        time.sleep(sleep_sec)     # 쿼터/레이트 한도 보호
    return collected

def fetch_all_keywords(keywords, per_page=100, max_pages=2):
    all_items = []
    for kw in keywords:
        try:
            all_items.extend(fetch_news_for_keyword(kw, per_page=per_page, max_pages=max_pages))
        except Exception as e:
            print(f"[WARN] '{kw}' 요청 실패: {e}")
    return all_items

# ── 필터링(‘오늘’ + 제외어), 그리고 정규화/기본 중복 제거
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

# ── 메인
if __name__ == "__main__":
    TARGET = 10

    # 1) 키워드 다발로 수집 (키워드×페이지네이션)
    raw = fetch_all_keywords(KEYWORDS, per_page=50, max_pages=2)  # 과금/쿼터 고려해 적당히
    # 2) 오늘 기사 + 제외어 필터 + URL 정규화/중복 제거
    base = filter_and_normalize(raw)
    # 3) 제목 유사 중복 제거(사전)
    base = dedup_by_title(base, threshold=93)
    # 4) 군집화로 인기 추정
    groups = cluster_by_title(base, eps=0.65, min_samples=2)
    # 5) Top10 보장(+유사제목 차단)
    top = top_n_with_diversity(groups, base, n=TARGET, sim_threshold=93)

    out = {
        "as_of": datetime.now(KST).isoformat(),
        "total_raw": len(raw),
        "total_after_filter": len(base),
        "cluster_count": len(groups),
        "returned": len(top),
        "top": [
            {
                "rank": i+1,
                "size": g["size"],
                "title": g["rep_title"],
                "url": g["rep_url"],
                "source": g["rep_source"],
                "latest_pubDate": g["latest_pubDate"]
            } for i,g in enumerate(top)
        ]
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    with open("econ_popular_multi.json","w",encoding="utf-8") as f:
        f.write(text)
    print("저장 완료 → econ_popular_multi.json")