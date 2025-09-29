import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import requests
# Python 3.10+에서 bs4 4.6.x 호환을 위한 collections.* 심볼 보정
try:
    import collections  # noqa: F401
    import collections.abc as _collections_abc  # noqa: F401
    for _name in ("Callable", "Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, _name):  # type: ignore[attr-defined]
            # type: ignore[attr-defined]
            setattr(collections, _name, getattr(_collections_abc, _name))
except Exception:
    pass
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from urllib3.util.retry import Retry


BASE_LIST_URL = "https://kabc.dongguk.edu/content/list?itemId=ABC_BJ&lang=ko"
BASE_HOST = "https://kabc.dongguk.edu"
TREE_ALL_URL = "https://kabc.dongguk.edu/content/list?itemId=ABC_BJ&lang=ko"


def create_session(total_retries: int = 3, backoff_factor: float = 0.3, timeout_sec: float = 20.0, verify: Union[bool, str] = True) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    session.request = _with_timeout(
        session.request, timeout_sec)  # type: ignore[assignment]
    # SSL 인증서 검증 설정 (True/False 또는 CA bundle 경로)
    session.verify = verify
    return session


def _with_timeout(request_fn, timeout_sec: float):
    def wrapped(method, url, **kwargs):  # noqa: ANN001
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout_sec
        return request_fn(method, url, **kwargs)

    return wrapped


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def canonicalize_view_url(url: str) -> str:
    """Return canonical content/view URL keeping only dataId query param.
    Examples:
      /content/view?dataId=ABC_IT_K0223_T_001&rt=R  -> /content/view?dataId=ABC_IT_K0223_T_001
      https://kabc.dongguk.edu/content/view?dataId=... -> same host/path with only dataId
    """
    p = urlparse(urljoin(BASE_HOST, url))
    if not p.path.endswith("/content/view"):
        return urljoin(BASE_HOST, url)
    q = parse_qs(p.query)
    new_q = {}
    # Prefer canonical dataId; fall back to ZsdataId if present
    if "dataId" in q and q["dataId"]:
        new_q["dataId"] = q["dataId"][0]
    elif "ZsdataId" in q and q["ZsdataId"]:
        new_q["dataId"] = q["ZsdataId"][0]
    new_query = urlencode(new_q)
    return urlunparse((p.scheme or "https", p.netloc or urlparse(BASE_HOST).netloc, p.path, "", new_query, ""))


def extract_view_links(soup: BeautifulSoup, base: str) -> List[str]:
    links: Set[str] = set()
    # Case 1: direct anchors to content/view
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "/content/view" in href:
            abs_url = urljoin(base, href)
            links.add(canonicalize_view_url(abs_url))
    # Case 2: elements carry data-url attributes (as in the screenshot)
    for el in soup.select('[data-url]'):
        durl = (el.get('data-url') or '').strip()
        if not durl:
            continue
        # Normalize to absolute content/view URL
        if "/content/view" in durl or durl.startswith("http"):
            abs_url = urljoin(base, durl)
        else:
            # Some pages provide only query like "?itemId=...&ZsdataId=..."
            query = durl if durl.startswith("?") else ("?" + durl)
            abs_url = urljoin(base, "/content/view" + query)
        links.add(canonicalize_view_url(abs_url))
    return sorted(links)


def _parse_query_from_data_url(data_url_value: str) -> Dict[str, str]:
    """data-url 값에서 쿼리 파라미터만 단순 파싱한다."""
    if not data_url_value:
        return {}
    query = data_url_value[data_url_value.find(
        '?') + 1:] if '?' in data_url_value else data_url_value
    params: Dict[str, str] = {}
    for token in [p for p in query.split('&') if '=' in p]:
        k, v = token.split('=', 1)
        params[k] = v
    return params


def _is_volume_data_id(data_id: str) -> bool:
    """'_T_###' 꼬리표가 붙은 권 ID 여부를 단순 문자열 검사로 판별한다."""
    if not data_id:
        return False
    if '_T_' not in data_id:
        return False
    prefix, suffix = data_id.rsplit('_T_', 1)
    if not prefix or not suffix:
        return False
    return len(suffix) == 3 and suffix.isdigit()


def _find_display_block_ul(root: Tag) -> Optional[Tag]:
    """style에 'display:block'이 들어간 첫 번째 UL을 찾는다."""
    for ul in root.find_all('ul'):
        style = (ul.get('style') or '').lower().replace(' ', '')
        if 'display:block' in style:
            return ul
    return None


def _parse_query_from_data_url(data_url_value: str) -> Dict[str, str]:
    """data-url 값에서 쿼리 파라미터만 단순 파싱한다(정규식 미사용)."""
    if not data_url_value:
        return {}
    query = data_url_value[data_url_value.find(
        '?') + 1:] if '?' in data_url_value else data_url_value
    params: Dict[str, str] = {}
    for token in [p for p in query.split('&') if '=' in p]:
        k, v = token.split('=', 1)
        params[k] = v
    return params


def _is_volume_data_id(data_id: str) -> bool:
    """'_T_###' 꼬리표가 붙은 권 ID 여부를 단순 문자열 검사로 판별한다."""
    if not data_id:
        return False
    if '_T_' not in data_id:
        return False
    prefix, suffix = data_id.rsplit('_T_', 1)
    if not prefix or not suffix:
        return False
    return len(suffix) == 3 and suffix.isdigit()


def _find_display_block_ul(root: Tag) -> Optional[Tag]:
    """style에 'display: block'이 들어간 첫 번째 UL을 찾는다."""
    for ul in root.find_all('ul'):
        style = (ul.get('style') or '').lower().replace(' ', '')
        if 'display:block' in style:
            return ul
    return None


def _extract_ajax_url_from_ul(ul_node: Tag) -> str:
    """Extract relative AJAX URL from a UL that contains a placeholder like '{url:./treeAjax?...}'."""
    if not ul_node:
        return ""
    # The placeholder is often inside a single LI as literal text
    text = ul_node.get_text(" ", strip=True)
    if not text:
        return ""
    # Expected format: {url:./treeAjax?itemId=ABC_IT&cate=bookName&depth=1&upPath=Z&dataId=}
    start = text.find('{url:')
    end = text.find('}', start + 5) if start >= 0 else -1
    if start >= 0 and end > start:
        rel = text[start + 5:end].strip()
        return rel
    return ""


def extract_all_volume_links_from_tree(soup: BeautifulSoup) -> List[str]:
    """Extract all volume view links from the left tree ('전체') structure.

    사용자 지시에 따른 DOM 전용 전략:
      1) data-url가 '?itemId=ABC_IT&cate=bookName&depth=1&upPath=Z&dataId=' 로 시작하는 LI를 찾는다
      2) 그 내부에서 style이 'display: block;' 인 UL을 찾고, 하위 LI(책)를 순회한다
      3) 각 책 LI 내부에서 다시 'display: block;' UL을 찾아, 하위 LI(권)을 순회한다
      4) 각 권 LI의 data-url에서 dataId를 읽어 '/content/view?dataId=...'로 만든다
    """
    links: Set[str] = set()
    # 1) depth=1 루트 LI 찾기 (속성이 자식에 있을 수 있으므로 캐리어 기준으로 찾고 상위 LI로 승격)
    carriers = soup.select(
        '[data-url^="?itemId=ABC_BJ&cate=bookName&depth=1&upPath=Z&dataId="]')
    if not carriers:
        return []
    root_li = carriers[0] if isinstance(
        carriers[0], Tag) and carriers[0].name == 'li' else carriers[0].find_parent('li')
    if not root_li:
        return []
    # 2) 보이는 UL(books)
    books_ul = _find_display_block_ul(root_li)
    if not books_ul:
        return []
    # 3) 각 책 LI 순회 (직계만)
    for book_li in books_ul.find_all('li', recursive=False):
        # 3-1) 보이는 UL(권 목록)
        vols_ul = _find_display_block_ul(book_li)
        if not vols_ul:
            continue
        # 3-2) 각 권 LI 순회 (직계만)
        for vol_li in vols_ul.find_all('li', recursive=False):
            vol_durl = (vol_li.get('data-url') or '').strip()
            if not vol_durl:
                carrier = vol_li.find(attrs={'data-url': True})
                if carrier:
                    vol_durl = (carrier.get('data-url') or '').strip()
            if not vol_durl:
                continue
            q = _parse_query_from_data_url(vol_durl)
            data_id = q.get('dataId') or q.get('ZsdataId') or ''
            if not _is_volume_data_id(data_id):
                continue
            abs_url = urljoin(BASE_HOST, f"/content/view?dataId={data_id}")
            links.add(canonicalize_view_url(abs_url))
    return sorted(links)


def collect_volume_links_via_ajax(session: requests.Session) -> List[str]:
    """Traverse the tree via AJAX endpoints to collect all volume links.

    Steps:
      1) Load the list page and locate the depth=1 root LI
      2) From its UL.ajax, extract the AJAX URL and fetch book-level LIs
      3) For each book LI, from its UL.ajax extract the AJAX URL and fetch volume-level LIs
      4) From volume LIs, extract dataId and build canonical content/view URLs
    """
    links: Set[str] = set()

    # 1) Load the list and find the root '전체' LI
    list_soup = get_soup(session, BASE_LIST_URL)
    carriers = list_soup.select(
        '[data-url^="?itemId=ABC_BJ&cate=bookName&depth=1&upPath=Z&dataId="]')
    if not carriers:
        # Try root-open URL
        root_open_url = build_tree_root_open_url()
        list_soup = get_soup(session, root_open_url)
        carriers = list_soup.select(
            '[data-url^="?itemId=ABC_BJ&cate=bookName&depth=1&upPath=Z&dataId="]')
        if not carriers:
            return []
    root_li = carriers[0] if carriers[0].name == 'li' else carriers[0].find_parent(
        'li')
    if not root_li:
        return []
    # 2) Extract AJAX URL for books and fetch
    books_ul = root_li.find('ul', class_='ajax') or root_li.find('ul')
    ajax_books_rel = _extract_ajax_url_from_ul(books_ul) if books_ul else ""
    if not ajax_books_rel:
        return []
    # treeAjax is relative to /content/ (list page path). Join against that base.
    ajax_books_url = urljoin(BASE_HOST + '/content/',
                             ajax_books_rel.lstrip('./'))
    books_soup = get_soup(session, ajax_books_url)

    # 3) Iterate books; for each, fetch its volumes via its own AJAX UL
    for book_li in books_soup.find_all('li', recursive=False):
        # Ignore volume nodes mistakenly at this level
        data_code = (book_li.get('data-code') or '').strip()
        if data_code and _is_volume_data_id(data_code):
            continue
        # Find the book's AJAX UL
        vol_ul = book_li.find('ul', class_='ajax') or book_li.find('ul')
        ajax_vol_rel = _extract_ajax_url_from_ul(vol_ul) if vol_ul else ""
        if not ajax_vol_rel:
            # Fallback: if volumes already present as direct child LIs
            for vol_li in book_li.find_all('li', recursive=False):
                durl = (vol_li.get('data-url') or '').strip()
                if not durl:
                    carrier = vol_li.find(attrs={'data-url': True})
                    durl = (carrier.get('data-url')
                            or '').strip() if carrier else ''
                if not durl:
                    continue
                q = _parse_query_from_data_url(durl)
                data_id = q.get('dataId') or q.get('ZsdataId') or ''
                if not _is_volume_data_id(data_id):
                    continue
                abs_url = urljoin(BASE_HOST, f"/content/view?dataId={data_id}")
                links.add(canonicalize_view_url(abs_url))
            continue
        ajax_vol_url = urljoin(BASE_HOST + '/content/',
                               ajax_vol_rel.lstrip('./'))
        vol_soup = get_soup(session, ajax_vol_url)
        for vol_li in vol_soup.find_all('li', recursive=False):
            durl = (vol_li.get('data-url') or '').strip()
            if not durl:
                carrier = vol_li.find(attrs={'data-url': True})
                durl = (carrier.get('data-url')
                        or '').strip() if carrier else ''
            if not durl:
                continue
            q = _parse_query_from_data_url(durl)
            data_id = q.get('dataId') or q.get('ZsdataId') or ''
            if not _is_volume_data_id(data_id):
                continue
            abs_url = urljoin(BASE_HOST, f"/content/view?dataId={data_id}")
            links.add(canonicalize_view_url(abs_url))

    return sorted(links)


def extract_volume_links_from_expanded_page(soup: BeautifulSoup, parent_id: str) -> List[str]:
    """부모 확장(depth=2) 페이지에서 해당 부모 LI 아래의 UL(접힘/펼침 무관) 내부 권들을 수집한다.

    DOM만 사용:
      - data-url 또는 data-code로 부모 LI를 식별
      - 부모 LI 내부에서 UL을 찾아(우선 display:block, 없으면 첫 번째 UL) 직계 LI를 권으로 간주
      - 각 권 LI의 data-url에서 dataId를 읽어 canonical 링크 구성
    """
    links: Set[str] = set()
    # 1) 부모 LI 찾기 (data-url 기준)
    carriers = soup.select(
        f'[data-url*="?itemId=ABC_BJ&cate=bookName&depth=2&upPath=Z&dataId={parent_id}"]'
    )
    target_li: Optional[Tag] = None
    if carriers:
        target_li = carriers[0] if carriers[0].name == 'li' else carriers[0].find_parent(
            'li')
    # 2) 보조: data-code로 직접 매칭
    if not target_li:
        for li in soup.find_all('li'):
            if (li.get('data-code') or '').strip() == parent_id:
                target_li = li
                break
    if not target_li:
        return []
    # 3) 부모 LI 하위에서 보이는 권 UL 찾기
    # 3) 부모 LI 하위의 모든 UL을 순회(접힘/펼침 무관)하며 직계 LI를 권으로 간주
    uls = target_li.find_all('ul')
    if not uls:
        return []
    for ul in uls:
        for vol_li in ul.find_all('li', recursive=False):
            vol_durl = (vol_li.get('data-url') or '').strip()
            if not vol_durl:
                carrier = vol_li.find(attrs={'data-url': True})
                if carrier:
                    vol_durl = (carrier.get('data-url') or '').strip()
            if not vol_durl:
                continue
            q = _parse_query_from_data_url(vol_durl)
            data_id = q.get('dataId') or q.get('ZsdataId') or ''
            if not _is_volume_data_id(data_id):
                continue
            abs_url = urljoin(BASE_HOST, f"/content/view?dataId={data_id}")
            links.add(canonicalize_view_url(abs_url))
    return sorted(links)


def extract_parent_ids_from_tree(soup: BeautifulSoup) -> List[str]:
    """depth=1 루트 LI의 첫 번째 UL 하위 직계 LI에서 부모 dataId를 수집한다."""
    parent_ids: Set[str] = set()
    carriers = soup.select(
        '[data-url^="?itemId=ABC_BJ&cate=bookName&depth=1&upPath=Z&dataId="]')
    if not carriers:
        return []
    root_li = carriers[0] if isinstance(
        carriers[0], Tag) and carriers[0].name == 'li' else carriers[0].find_parent('li')
    if not root_li:
        return []
    books_ul = root_li.find('ul')
    if not books_ul:
        return []
    for book_li in books_ul.find_all('li', recursive=False):
        durl = (book_li.get('data-url') or '').strip()
        if not durl:
            carrier = book_li.find(attrs={'data-url': True})
            durl = (carrier.get('data-url') or '').strip() if carrier else ''
        if durl:
            q = _parse_query_from_data_url(durl)
            did = q.get('dataId') or q.get('ZsdataId') or ''
            if did and did.startswith('ABC_') and not _is_volume_data_id(did):
                parent_ids.add(did)
                continue
        dc = (book_li.get('data-code') or '').strip()
        if dc and dc.startswith('ABC_') and not _is_volume_data_id(dc):
            parent_ids.add(dc)
    return sorted(parent_ids)


def build_tree_expand_url(parent_data_id: str) -> str:
    """좌측 트리에서 parent 노드를 펼친 상태의 목록 URL 구성."""
    return urljoin(
        BASE_HOST,
        f"/content/list?itemId=ABC_BJ&lang=ko&cate=bookName&depth=2&upPath=Z&dataId={parent_data_id}",
    )


def build_tree_root_open_url() -> str:
    """depth=1 루트를 펼친 상태의 목록 URL 구성 (dataId 비지정)."""
    return urljoin(
        BASE_HOST,
        "/content/list?itemId=ABC_BJ&lang=ko&cate=bookName&depth=1&upPath=Z",
    )


def crawl_from_tree(session: requests.Session, output_path: str, delay_sec: float) -> None:
    """트리(좌측 전체 탭)에서 부모를 순회하고 각 부모 확장 페이지(depth=2)에서 권을 수집한다."""
    collected: Set[str] = set()
    # AJAX 기반으로 전체 트리를 순회하여 권 링크를 수집
    for u in collect_volume_links_via_ajax(session):
        collected.add(canonicalize_view_url(u))

    vol_links: List[str] = sorted(collected)

    seen: Set[str] = set()
    total = 0
    for url in vol_links:
        canon = canonicalize_view_url(url)
        if canon in seen:
            continue
        seen.add(canon)
        try:
            record, _ = process_detail_page(session, canon, delay_sec)
            write_jsonl(output_path, record)
            total += 1
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 실패(트리): {canon} -> {e}", file=sys.stderr)
    print(f"완료(트리): {total}건 저장 -> {output_path}")


def _candidate_content_containers(soup: BeautifulSoup) -> List[Tag]:
    selectors = [
        "div#content",
        "div.content",
        "section.content",
        "div#main",
        "main",
        "div.view",
        "article",
        "div.container",
        "div.wrapper",
    ]
    candidates: List[Tag] = []
    for sel in selectors:
        candidates.extend(soup.select(sel))
    if soup.body:
        candidates.append(soup.body)
    seen: Set[int] = set()
    unique: List[Tag] = []
    for node in candidates:
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(node)
    return unique


def _text_len_score(text: str) -> int:
    return len(text or "")


HAN_BLOCKS = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
]


def _is_han_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in HAN_BLOCKS:
        if start <= code <= end:
            return True
    return False


def _contains_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF for c in text)


def _han_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return 0.0
    han = sum(1 for c in text if _is_han_char(c))
    return han / max(total, 1)


NOISE_KEYWORDS: Set[str] = {
    "불교학술원", "불교기록문화유산", "한국불교문화포털", "공지사항", "오류신고", "사업소개",
    "한국어", "English", "통합대장경", "한국불교전서", "신집성문헌", "고려교장",
    "변상도", "근대불교잡지", "근대불교문헌", "근대불교사진", "조선사찰본서목",
    "검색", "상세검색", "서지", "해제", "원문", "번역문", "주석", "원문이미지", "개인화",
    "보기", "라인정보", "표점", "각주", "기능버튼", "인용표기", "경명순", "전체",
    "경번호순", "분류체계별", "주제별", "URL복사", "통합뷰어", "단락보기", "경판보기",
    "경명+상세정보", "경명+간략정보", "經名+상세정보", "經名+간략정보", "기타",
}

_PUNCT_CHARS = r".,!?;:\u00b7\-—–…‘’“”'\"()\[\]{}<>〈〉《》【】〔〕、，。！？：；|+"
_RE_ONLY_PUNCT = re.compile(rf"^[{_PUNCT_CHARS}\s]+$")
_RE_URL_TOKEN = re.compile(r"^\{\s*url:.*\}$")
_RE_LINE_ID = re.compile(r"^[0-9]{3}_[0-9]{4}_[a-z]_[0-9]{2}L$", re.IGNORECASE)
_RE_NUMBER_PAREN = re.compile(r"^\d+\)\s*$")
_RE_ABC_CONTENT_ID = re.compile(r"^ABC_[A-Z0-9_]+$")
_RE_BOOK_TITLE_LINE = re.compile(r"^『.*』.*$")
_RE_COPYRIGHT = re.compile(r"ⓒ|동국대학교\s*불교학술원")
# Sutra title line like: 가야산정경(伽耶山頂經)
_RE_SUTRA_TITLE_SENTINEL = re.compile(r"^.+\([\u4e00-\u9fff]+\)\s*$")
# Trailing footnote lines
_RE_FOOTNOTE_HINT = re.compile(r"고려대장경|\(역\)\s*$")


def _is_punct_only(text: str) -> bool:
    return bool(_RE_ONLY_PUNCT.match(text))


def _is_noise_line(text: str, page_title: str) -> bool:
    if not text:
        return True
    if page_title and text.strip() == page_title.strip():
        return True
    if _is_punct_only(text):
        return True
    if len(text) == 1 and _contains_hangul(text):
        return True
    if _RE_URL_TOKEN.match(text):
        return True
    if _RE_LINE_ID.match(text):
        return True
    if _RE_NUMBER_PAREN.match(text):
        return True
    if _RE_ABC_CONTENT_ID.match(text):
        return True
    if _RE_BOOK_TITLE_LINE.match(text):
        return True
    if _RE_COPYRIGHT.search(text):
        return True
    if text in NOISE_KEYWORDS:
        return True
    if len(text) <= 20 and any(k in text for k in NOISE_KEYWORDS):
        return True
    return False


def _trim_preamble_to_sutra(lines: List[str]) -> List[str]:
    idx = -1
    for i, ln in enumerate(lines):
        if _RE_SUTRA_TITLE_SENTINEL.match(ln):
            idx = i
            break
    if idx >= 0:
        return lines[idx:]
    return lines


def clean_and_filter_lines(raw_text: str, page_title: str, han_ratio_threshold: float = 0.35) -> List[str]:
    lines = [re.sub(r"\s+", " ", line).strip()
             for line in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]
    # Remove preamble up to sutra title sentinel
    lines = _trim_preamble_to_sutra(lines)
    filtered: List[str] = []
    for ln in lines:
        if _is_noise_line(ln, page_title):
            continue
        ratio = _han_ratio(ln)
        if ratio <= han_ratio_threshold or _contains_hangul(ln):
            filtered.append(ln)
    return filtered


def pick_largest_text_container(soup: BeautifulSoup) -> str:
    best_text = ""
    best_score = -1
    for node in _candidate_content_containers(soup):
        text = node.get_text("\n", strip=True)
        score = _text_len_score(text)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def _get_text_or_empty(node: Optional[Tag]) -> str:
    if not node:
        return ""
    return node.get_text("\n", strip=True)


def extract_title(soup: BeautifulSoup) -> str:
    h = soup.find(["h1", "h2"]) or soup.title
    if h:
        return h.get_text(strip=True)
    return ""


_RE_NOTE_INLINE = re.compile(r"^(?:역주|주\s*[:：]|\[?주\s*\d+\]?|\d+\)\s*).*")


def split_translation_and_notes(lines: List[str]) -> Tuple[str, List[str]]:
    if not lines:
        return "", []
    inline_notes: List[str] = []
    body_lines: List[str] = []
    for ln in lines:
        if _RE_NOTE_INLINE.match(ln):
            inline_notes.append(ln)
        else:
            body_lines.append(ln)
    note_index = -1
    for i, ln in enumerate(body_lines):
        if re.search(r"주석", ln):
            note_index = i
            break
    if note_index >= 0:
        translation_lines = body_lines[:note_index]
        notes_lines = body_lines[note_index:]
        if notes_lines and re.fullmatch(r"\s*주석\s*", notes_lines[0]):
            notes_lines = notes_lines[1:]
    else:
        translation_lines = list(body_lines)
        notes_lines = []
    tail_notes: List[str] = []
    while translation_lines:
        last = translation_lines[-1]
        if _RE_NUMBER_PAREN.match(last) or _RE_FOOTNOTE_HINT.search(last) or _RE_COPYRIGHT.search(last):
            tail_notes.append(last)
            translation_lines.pop()
            continue
        if len(last) <= 40 and ("고려대장경" in last or "역)" in last):
            tail_notes.append(last)
            translation_lines.pop()
            continue
        break
    tail_notes.reverse()
    all_notes = [n for n in (
        notes_lines + tail_notes + inline_notes) if n.strip()]
    return "\n".join(translation_lines).strip(), all_notes


def load_resume(resume_file: str) -> Set[str]:
    if not os.path.exists(resume_file):
        return set()
    try:
        with open(resume_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return set(k for k, v in data.items() if v)
            if isinstance(data, list):
                return set(str(u) for u in data)
    except Exception:
        return set()
    return set()


def save_resume(resume_file: str, processed: Set[str]) -> None:
    os.makedirs(os.path.dirname(resume_file), exist_ok=True)
    with open(resume_file, "w", encoding="utf-8") as f:
        json.dump({u: True for u in sorted(processed)},
                  f, ensure_ascii=False, indent=2)


def write_jsonl(output_path: str, record: Dict) -> None:  # noqa: ANN401
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_pagination_param_and_max(soup: BeautifulSoup, list_url: str) -> Tuple[Optional[str], Optional[int]]:
    parsed = urlparse(list_url)
    base_query = parse_qs(parsed.query)
    page_param_name: Optional[str] = None
    max_page: Optional[int] = None
    candidate_names = {"page", "pageIndex", "pageNo", "curPage", "pageNum"}

    page_numbers: Set[int] = set()
    for a in soup.find_all("a", href=True):
        text = (a.get_text(strip=True) or "").replace(",", "")
        if text.isdigit():
            try:
                page_numbers.add(int(text))
            except ValueError:
                pass
        href = a["href"].strip()
        if not href or "content/list" not in href:
            continue
        p = urlparse(urljoin(BASE_HOST, href))
        q = parse_qs(p.query)
        for name in q:
            if name in base_query:
                continue
            if name.lower() in {n.lower() for n in candidate_names}:
                page_param_name = name
        for name, vals in q.items():
            if vals and vals[0].isdigit() and name.lower() in {n.lower() for n in candidate_names}:
                page_param_name = name

    if page_numbers:
        try:
            max_page = max(page_numbers)
        except ValueError:
            max_page = None

    return page_param_name, max_page


def build_url_with_param(url: str, param: str, value: int) -> str:
    p = urlparse(url)
    q = parse_qs(p.query)
    q[param] = [str(value)]
    new_query = urlencode({k: v[0] if isinstance(
        v, list) else v for k, v in q.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def build_url_with_params(url: str, params: Dict[str, Union[str, int]]) -> str:  # noqa: ANN401
    p = urlparse(url)
    q = parse_qs(p.query)
    for k, v in params.items():
        q[k] = [str(v)]
    new_query = urlencode({k: v[0] if isinstance(
        v, list) else v for k, v in q.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def iter_list_pages(session: requests.Session, list_url: str, delay_sec: float, max_pages: Optional[int]) -> Iterable[Tuple[int, BeautifulSoup]]:
    """Iterate pages from the list, auto-detecting pagination param and max page.

    Behavior:
      - Fetch the first list page to detect pagination parameter name and max page
      - If detection fails, fall back to incremental paging using 'pageIndex'
      - If max_pages is provided, it caps traversal
    """
    page_param_name: str = "pageIndex"
    # Try to fetch the first page without pagination to detect param and max
    try:
        first_soup = get_soup(session, list_url)
        detected_param, detected_max = find_pagination_param_and_max(
            first_soup, list_url)
        if detected_param:
            page_param_name = detected_param
        if max_pages is None and detected_max:
            max_pages = detected_max
        # Yield first page as-is (already fetched)
        yield 1, first_soup
        page = 2
    except Exception:
        # Detection failed; start from page 1 with default param
        page = 1

    while max_pages is None or page <= max_pages:
        page_url = build_url_with_params(
            list_url, {page_param_name: page, "pageUnit": 20})
        soup = get_soup(session, page_url)
        links = extract_view_links(soup, BASE_HOST)
        if not links:
            # No more pages
            break
        yield page, soup
        page += 1
        time.sleep(delay_sec)


def extract_sections_from_dom(soup: BeautifulSoup) -> Tuple[str, str]:
    # Try common containers first
    translation_selectors = [
        "div.translation", "#translation", "section.translation",
        "div.trans", "#trans", "div#kabcTranslation",
    ]
    notes_selectors = [
        "div.annotation", "#annotation", "section.annotation",
        "div.footnote", "#footnote", "#footnotes", "ol.footnote", "ul.notes",
        "div.notes", "#notes", ".jusok__list",
    ]
    translation = ""
    notes = ""
    for sel in translation_selectors:
        node = soup.select_one(sel)
        if node:
            translation = _get_text_or_empty(node)
            break
    for sel in notes_selectors:
        node = soup.select_one(sel)
        if node:
            notes = _get_text_or_empty(node)
            break
    return translation, notes


def remove_note_containers_from_dom(soup: BeautifulSoup) -> None:
    try:
        selectors = [
            "div.annotation", "#annotation", "section.annotation",
            "div.footnote", "#footnote", "#footnotes", "ol.footnote", "ul.notes",
            "div.notes", "#notes", ".jusok__list",
        ]
        for sel in selectors:
            for n in soup.select(sel):
                try:
                    n.decompose()
                except Exception:
                    pass
    except Exception:
        pass


def extract_jusok_notes_and_remove(soup: BeautifulSoup) -> List[Dict[str, str]]:
    nodes = soup.select('.jusok__list [data-jusok-desc-num]')
    items: List[Dict[str, str]] = []
    if not nodes:
        # if children carry items but list wrapper exists
        wrapper_nodes = soup.select('.jusok__list')
        for w in wrapper_nodes:
            try:
                w.decompose()
            except Exception:
                pass
        return items
    for n in nodes:
        num = n.get('data-jusok-desc-num', '').strip()
        text = _get_text_or_empty(n)
        if text:
            items.append({'num': num, 'text': text})
        # remove the item node from DOM to avoid leakage
        try:
            n.decompose()
        except Exception:
            pass
    return items


def extract_volume_view_links_from_detail(soup: BeautifulSoup) -> List[str]:
    """Extract volume navigation links (모든 권) from the detail page DOM.

    Strategies:
      1) Prefer structured `data-url` elements that include `ZsdataId=...` or `dataId=...`.
      2) Fallback to anchors to `/content/view` whose text looks like volume labels.
    """
    links: Set[str] = set()

    # 1) Structured data-url with ZsdataId or dataId (권 패턴 우선)
    for el in soup.select('[data-url]'):
        durl = (el.get('data-url') or '').strip()
        if not durl:
            continue
        # Normalize to query part
        query = durl[durl.find('?') + 1:] if '?' in durl else durl
        params = dict((kv.split('=')[0], kv.split('=')[1])
                      for kv in [p for p in query.split('&') if '=' in p])
        data_id = params.get('dataId') or params.get('ZsdataId') or ''
        if not data_id:
            continue
        # Only include true volume ids like ..._T_001
        if not re.search(r'_T_\d+$', data_id):
            continue
        abs_url = urljoin(BASE_HOST, f"/content/view?dataId={data_id}")
        links.add(canonicalize_view_url(abs_url))

    # 2) Fallback: anchors labeled like volumes
    for a in soup.find_all("a", href=True):
        txt = (a.get_text(strip=True) or "")
        if not txt:
            continue
        if not ("권" in txt or re.fullmatch(r"\d+\s*권", txt)):
            continue
        href = a["href"].strip()
        if not href:
            continue
        if "/content/view" not in href and "dataId=" not in href and "ZsdataId=" not in href:
            continue
        if "/content/view" in href or href.startswith("http"):
            abs_url = urljoin(BASE_HOST, href)
        else:
            query = href if href.startswith("?") else ("?" + href)
            abs_url = urljoin(BASE_HOST, "/content/view" + query)
        links.add(canonicalize_view_url(abs_url))

    return sorted(links)


def _extract_data_id_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        q = parse_qs(p.query)
        if "dataId" in q and q["dataId"]:
            return q["dataId"][0]
        if "ZsdataId" in q and q["ZsdataId"]:
            return q["ZsdataId"][0]
    except Exception:
        pass
    return ""


def generate_volume_candidate_ids(current_data_id: str, max_probe: int = 200) -> List[str]:
    """Generate plausible volume dataIds like PREFIX_T_001..N from a given dataId.

    If the current id already has a _T_### suffix, use its prefix; otherwise append _T_### to the id.
    This only generates candidates; validity is checked at fetch time.
    """
    m = re.match(r"^(.*?_T_)\d+$", current_data_id)
    if m:
        prefix = m.group(1)
    else:
        prefix = current_data_id.rstrip("_") + "_T_"
    return [f"{prefix}{i:03d}" for i in range(1, max_probe + 1)]


def process_detail_page(session: requests.Session, url: str, delay_sec: float) -> Tuple[Dict, List[str]]:  # noqa: ANN401
    time.sleep(delay_sec)
    soup = get_soup(session, url)
    title = extract_title(soup)

    # Extract jusok notes first and remove from DOM
    jusok_items = extract_jusok_notes_and_remove(soup)

    # DOM-first extraction
    dom_trans, dom_notes = extract_sections_from_dom(soup)

    # Prepare notes items (from DOM containers and jusok)
    notes_items: List[Dict[str, str]] = []
    if jusok_items:
        notes_items.extend(jusok_items)
    else:
        if dom_notes:
            for ln in clean_and_filter_lines(dom_notes, title):
                if ln.strip():
                    notes_items.append({'num': '', 'text': ln})

    # Build translation text
    if dom_trans:
        trans_lines = clean_and_filter_lines(dom_trans, title)
    else:
        # Remove remaining note containers and derive translation from largest text
        remove_note_containers_from_dom(soup)
        raw_text = pick_largest_text_container(soup)
        trans_lines = clean_and_filter_lines(raw_text, title)
        # Fallback split if heuristics detect embedded notes
        if not notes_items:
            t_text, n_lines = split_translation_and_notes(trans_lines)
            trans_lines = [ln for ln in t_text.split("\n") if ln.strip()]
            for ln in n_lines:
                if ln.strip():
                    notes_items.append({'num': '', 'text': ln})

    translation = "\n".join([ln for ln in trans_lines if ln.strip()]).strip()

    record = {
        "url": canonicalize_view_url(url),
        "title": title,
        "translation": translation,
        "notes": notes_items,
    }
    # Discover additional volume links within the detail page (e.g., 001권, 002권 ...)
    related_volume_urls = extract_volume_view_links_from_detail(soup)
    return record, related_volume_urls


def crawl(
    list_url: str,
    output_path: str,
    delay_sec: float,
    max_pages: Optional[int],
    resume_file: Optional[str],
    verify: Union[bool, str] = True,
) -> None:
    session = create_session(verify=verify)

    processed: Set[str] = set()
    if resume_file:
        processed = load_resume(resume_file)

    seen_view_urls: Set[str] = set()
    total_written = 0

    for page_num, soup in iter_list_pages(session, list_url, delay_sec, max_pages):
        view_links = extract_view_links(soup, BASE_HOST)
        new_links = [u for u in view_links if canonicalize_view_url(
            u) not in seen_view_urls]
        if not new_links and page_num > 1:
            break
        for view_url in new_links:
            canon_url = canonicalize_view_url(view_url)
            seen_view_urls.add(canon_url)
            if canon_url in processed:
                continue
            try:
                record, more_volume_urls = process_detail_page(
                    session, canon_url, delay_sec)
                # Write combined record
                write_jsonl(output_path, record)
                # Brute-probe sibling volumes based on current dataId pattern
                cur_id = _extract_data_id_from_url(canon_url)
                if cur_id:
                    cand_ids = generate_volume_candidate_ids(
                        cur_id, max_probe=300)
                    consecutive_failures = 0
                    for cand_id in cand_ids:
                        if cand_id == cur_id:
                            continue
                        vcanon = canonicalize_view_url(
                            f"/content/view?dataId={cand_id}")
                        if vcanon in seen_view_urls:
                            continue
                        seen_view_urls.add(vcanon)
                        try:
                            vrec, _ = process_detail_page(
                                session, vcanon, delay_sec)
                            if not vrec.get("translation") and not vrec.get("notes"):
                                consecutive_failures += 1
                                if consecutive_failures >= 3:
                                    break
                                continue
                            consecutive_failures = 0
                            write_jsonl(output_path, vrec)
                            total_written += 1
                            if resume_file and total_written % 20 == 0:
                                save_resume(resume_file, processed)
                        except Exception:
                            consecutive_failures += 1
                            if consecutive_failures >= 3:
                                break
                # Queue additional volume URLs discovered on the detail page
                for vurl in more_volume_urls:
                    vcanon = canonicalize_view_url(vurl)
                    if vcanon not in seen_view_urls:
                        seen_view_urls.add(vcanon)
                        # Immediate process to ensure we traverse 001권..N권
                        try:
                            vrec, _ = process_detail_page(
                                session, vcanon, delay_sec)
                            write_jsonl(output_path, vrec)
                            total_written += 1
                            if resume_file and total_written % 20 == 0:
                                save_resume(resume_file, processed)
                        except Exception as ve:  # noqa: BLE001
                            print(
                                f"[warn] 실패(권 링크): {vcanon} -> {ve}", file=sys.stderr)
                # Additionally write separate translation and notes records for post-train
                if os.environ.get("KABC_SPLIT_OUTPUT", "0") == "1":
                    if record.get("translation"):
                        write_jsonl(
                            output_path.replace(
                                ".jsonl", ".translation.jsonl"),
                            {
                                "url": record.get("url"),
                                "title": record.get("title"),
                                "text": record.get("translation"),
                                "type": "translation",
                            },
                        )
                    if record.get("notes"):
                        write_jsonl(
                            output_path.replace(".jsonl", ".notes.jsonl"),
                            {
                                "url": record.get("url"),
                                "title": record.get("title"),
                                "items": record.get("notes"),
                                "type": "notes",
                            },
                        )
                total_written += 1
                processed.add(canon_url)
                if resume_file and total_written % 20 == 0:
                    save_resume(resume_file, processed)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 실패: {canon_url} -> {e}", file=sys.stderr)

    if resume_file:
        save_resume(resume_file, processed)
    print(f"완료: {total_written}건 저장 -> {output_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KABC ABC_BJ 번역/주석 크롤러")
    parser.add_argument(
        "--list-url",
        default=BASE_LIST_URL,
        help="목록 URL (기본: https://kabc.dongguk.edu/content/list?itemId=ABC_BJ&lang=ko)",
    )
    default_output = "/home/work/paper/buddha/crawling/output/kabc_ABC_BJ.jsonl"
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"JSONL 저장 경로 (기본: {default_output})",
    )
    parser.add_argument("--delay", type=float, default=0.4, help="요청 간 대기(초)")
    parser.add_argument("--max-pages", type=int,
                        default=None, help="최대 페이지 수 (미설정 시 자동)")
    default_resume = "/home/work/paper/buddha/crawling/output/kabc_ABC_BJ.resume.json"
    parser.add_argument("--resume", action="store_true", help="재개 기능 사용")
    parser.add_argument(
        "--resume-file",
        default=default_resume,
        help=f"재개 상태 저장 경로 (기본: {default_resume})",
    )
    parser.add_argument("--insecure", action="store_true",
                        help="SSL 인증서 검증 비활성화")
    parser.add_argument("--ca-bundle", type=str,
                        default=None, help="사용할 CA bundle 경로 지정")
    parser.add_argument("--split-output", action="store_true",
                        help="번역/주석 별도 JSONL 생성")
    parser.add_argument("--mode", choices=["list", "tree"], default="list",
                        help="크롤링 모드: list(기본, 페이지네이션) | tree(전체 트리에서 권 링크 수집)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    resume_file = args.resume_file if args.resume else None
    verify: Union[bool, str] = True
    if args.insecure:
        verify = False
    elif args.ca_bundle:
        verify = args.ca_bundle
    if args.mode == "tree":
        session = create_session(verify=verify)
        crawl_from_tree(session, args.output, args.delay)
    else:
        crawl(
            list_url=args.list_url,
            output_path=args.output,
            delay_sec=args.delay,
            max_pages=args.max_pages,
            resume_file=resume_file,
            verify=verify,
        )


if __name__ == "__main__":
    main()
