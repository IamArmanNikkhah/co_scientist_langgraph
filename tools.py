from __future__ import annotations

import re
from typing import Dict, Optional

from langchain_core.tools import tool
import os
import json
import time
import urllib.parse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdf_extract_text = None  # type: ignore


def _extract_first_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text.
    - Skips content before the first '{'
    - Tracks nested braces
    - Ignores braces inside quoted strings (handles escapes)
    - Works even if surrounded by code fences or extra prose
    """
    i = 0
    n = len(text)
    in_string = False
    escape = False
    depth = 0
    start = -1

    while i < n:
        ch = text[i]
        if depth == 0:
            if ch == '{':
                depth = 1
                start = i
                i += 1
                continue
            i += 1
            continue

        # depth > 0: inside JSON
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
        i += 1
    return None


@tool("extract_json_from_text")
def extract_json_from_text(text: str) -> Optional[str]:
    """Extract a single JSON object from arbitrary text.
    Supports fenced blocks (```...```) and inline JSON, returning the first balanced object.
    """
    if not text:
        return None

    # If there is a fenced block, focus on it first
    fence = re.search(r"```[a-zA-Z]*\n([\s\S]*?)```", text)
    if fence:
        candidate = _extract_first_balanced_json(fence.group(1))
        if candidate:
            return candidate

    # Fallback to full text scan
    return _extract_first_balanced_json(text)


@tool("calculate_elo")
def calculate_elo(r1: float, r2: float, winner: int, k_factor: float = 32.0) -> Dict[str, int]:
    """Compute Elo updates for two ratings given a winner. Returns new_r1 and new_r2 (rounded ints)."""
    e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
    e2 = 1.0 / (1.0 + 10 ** ((r1 - r2) / 400.0))
    s1 = 1 if winner == 1 else 0
    s2 = 1 if winner == 2 else 0
    new_r1 = int(round(r1 + k_factor * (s1 - e1)))
    new_r2 = int(round(r2 + k_factor * (s2 - e2)))
    return {"new_r1": new_r1, "new_r2": new_r2}


@tool("infer_domain_from_goal")
def infer_domain_from_goal(research_goal: str) -> str:
    """Heuristic domain inference to guide observation extraction when LLM-based inference is unavailable."""
    text = research_goal.lower()
    if any(k in text for k in ["gene", "protein", "cell", "biolog", "enzyme"]):
        return "biology"
    if any(k in text for k in ["clinic", "patient", "trial", "therapy", "disease"]):
        return "medicine"
    if any(k in text for k in ["synthesis", "reaction", "molecule", "compound"]):
        return "chemistry"
    if any(k in text for k in ["quantum", "optics", "relativity", "particle", "thermo"]):
        return "physics"
    if any(k in text for k in ["alloy", "polymer", "crystal", "material", "microstructure"]):
        return "materials_science"
    if any(k in text for k in ["algorithm", "neural", "network", "model", "data", "compute"]):
        return "computer_science"
    return "general"


# --------------------------
# Literature Search & Fetch
# --------------------------

_USER_AGENT = "co-scientist/0.1 (+https://example.org; contact: system)"


def _http_get(url: str, params: Optional[Dict[str, str]] = None, timeout: int = 12):
    if requests is None:
        raise RuntimeError("requests package not available; please install 'requests'.")
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _split_authors(obj) -> list:
    authors = []
    if isinstance(obj, list):
        for a in obj:
            if isinstance(a, dict):
                name = _norm(a.get("name") or ((a.get("given") or "") + " " + (a.get("family") or "")))
                if name:
                    authors.append(name)
            elif isinstance(a, str):
                nm = _norm(a)
                if nm:
                    authors.append(nm)
    return authors


@tool("search_pubmed")
def search_pubmed(query: str, max_results: int = 10) -> Dict:
    """Search PubMed via NCBI E-utilities. Returns dict with items: id, title, authors, year, journal, doi, url."""
    try:
        q = urllib.parse.quote(query)
        esearch = _http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": q, "retmode": "json", "retmax": str(max_results)},
        ).json()
        idlist = esearch.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            return {"items": []}
        esummary = _http_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(idlist), "retmode": "json"},
        ).json()
        result = []
        docs = esummary.get("result", {})
        for pid in idlist:
            rec = docs.get(pid) or {}
            title = _norm(rec.get("title"))
            journal = _norm(rec.get("fulljournalname") or rec.get("source"))
            year = None
            try:
                year = int(str(rec.get("pubdate", "")).split(" ")[0]) if rec.get("pubdate") else None
            except Exception:
                year = None
            authors = [
                _norm(" ".join([a.get("name") or _norm((a.get("authtype") or ""))]).strip())
                for a in rec.get("authors", [])
                if _norm(a.get("name"))
            ]
            doi = None
            articleids = rec.get("articleids", [])
            for aid in articleids:
                if aid.get("idtype") == "doi":
                    doi = _norm(aid.get("value"))
                    break
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            result.append({
                "source": "pubmed",
                "id": pid,
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "url": url,
            })
        return {"items": result}
    except Exception as e:
        return {"error": f"pubmed_search_failed: {e}", "items": []}


@tool("search_crossref")
def search_crossref(query: str, max_results: int = 10) -> Dict:
    """Search Crossref works. Returns dict with items containing doi, title, authors, year, journal, url."""
    try:
        resp = _http_get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": str(max_results)},
        ).json()
        items = []
        for w in resp.get("message", {}).get("items", []):
            doi = _norm(w.get("DOI"))
            title = _norm(" ".join(w.get("title", []) or []))
            authors = _split_authors(w.get("author", []))
            year = None
            if w.get("issued", {}).get("date-parts"):
                try:
                    year = int(w["issued"]["date-parts"][0][0])
                except Exception:
                    year = None
            journal = _norm((w.get("container-title") or [None])[0])
            url = w.get("URL") or (f"https://doi.org/{doi}" if doi else None)
            items.append({
                "source": "crossref",
                "id": doi or url or title,
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "url": url,
            })
        return {"items": items}
    except Exception as e:
        return {"error": f"crossref_search_failed: {e}", "items": []}


@tool("search_arxiv")
def search_arxiv(query: str, max_results: int = 10) -> Dict:
    """Search arXiv via API. Returns dict with items: id, title, authors, year, url, arxiv_id."""
    try:
        # Simple XML parse without external deps
        url = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": "0", "max_results": str(max_results)}
        resp = _http_get(url, params=params)
        text = resp.text
        # Minimal parsing
        entries = text.split("<entry>")
        items = []
        for ent in entries[1:]:
            title = _norm(re.search(r"<title>([\s\S]*?)</title>", ent).group(1)) if re.search(r"<title>([\s\S]*?)</title>", ent) else ""
            # authors
            authors = [
                _norm(m)
                for m in re.findall(r"<name>([\s\S]*?)</name>", ent)
                if _norm(m)
            ]
            # id/url
            idm = re.search(r"<id>([\s\S]*?)</id>", ent)
            url_id = _norm(idm.group(1)) if idm else None
            # year
            y = None
            pm = re.search(r"<published>(\d{4})-\d{2}-\d{2}", ent)
            if pm:
                try:
                    y = int(pm.group(1))
                except Exception:
                    y = None
            # arxiv id
            aid = None
            if url_id and "/abs/" in url_id:
                aid = url_id.split("/abs/")[-1]
            items.append({
                "source": "arxiv",
                "id": aid or url_id or title,
                "title": title,
                "authors": authors,
                "year": y,
                "journal": "arXiv",
                "doi": None,
                "url": url_id,
                "arxiv_id": aid,
            })
        return {"items": items}
    except Exception as e:
        return {"error": f"arxiv_search_failed: {e}", "items": []}


@tool("search_scholar")
def search_scholar(query: str, max_results: int = 10) -> Dict:
    """Search Google Scholar via SerpAPI (requires SERPAPI_API_KEY). Returns items with title, url, snippet, year, authors if available.

    Note: We do not scrape Scholar directly to respect ToS. If no key is set, returns an informative error.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return {"error": "SERPAPI_API_KEY not set; cannot query Scholar.", "items": []}
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google_scholar", "q": query, "num": max_results, "api_key": api_key}
        data = _http_get(url, params=params).json()
        results = []
        for it in data.get("organic_results", [])[: max_results]:
            info = it or {}
            year = None
            try:
                ym = re.search(r"(19|20)\d{2}", info.get("publication_info", {}).get("summary", ""))
                if ym:
                    year = int(ym.group(0))
            except Exception:
                year = None
            results.append({
                "source": "scholar",
                "id": info.get("result_id") or info.get("link") or info.get("title"),
                "title": _norm(info.get("title")),
                "url": info.get("link"),
                "snippet": _norm(info.get("snippet")),
                "year": year,
                "authors": [],
                "journal": None,
                "doi": None,
            })
        return {"items": results}
    except Exception as e:
        return {"error": f"scholar_search_failed: {e}", "items": []}


@tool("fetch_url")
def fetch_url(url: str) -> Dict:
    """Fetch a URL and return content bytes, content_type, and headers. Bytes are base64-encoded."""
    import base64
    try:
        resp = _http_get(url)
        ct = resp.headers.get("Content-Type", "application/octet-stream")
        b64 = base64.b64encode(resp.content).decode("ascii")
        return {"url": url, "content_type": ct, "content_b64": b64, "headers": dict(resp.headers)}
    except Exception as e:
        return {"error": f"fetch_failed: {e}", "url": url}


@tool("extract_text_from_html")
def extract_text_from_html(html: str) -> Dict:
    """Extract readable text from HTML using BeautifulSoup (strips scripts/styles)."""
    if BeautifulSoup is None:
        return {"error": "beautifulsoup4 not installed"}
    try:
        soup = BeautifulSoup(html, "lxml") if BeautifulSoup else None
        if soup is None:
            return {"text": ""}
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join([t.strip() for t in soup.get_text("\n").splitlines() if t.strip()])
        return {"text": text}
    except Exception as e:
        return {"error": f"html_extract_failed: {e}"}


@tool("extract_text_from_pdf")
def extract_text_from_pdf(content_b64: str) -> Dict:
    """Extract text from PDF content provided as base64. Requires pdfminer.six."""
    if pdf_extract_text is None:
        return {"error": "pdfminer.six not installed"}
    import base64
    import io
    try:
        pdf_bytes = base64.b64decode(content_b64)
        with io.BytesIO(pdf_bytes) as f:
            text = pdf_extract_text(f)
        return {"text": text}
    except Exception as e:
        return {"error": f"pdf_extract_failed: {e}"}


@tool("resolve_doi")
def resolve_doi(doi: str) -> Dict:
    """Resolve DOI metadata via Crossref. Returns title, authors, year, journal, url, reference count (if any)."""
    try:
        doi = doi.strip().lower().replace("doi:", "").strip()
        resp = _http_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}").json()
        w = resp.get("message", {})
        title = _norm(" ".join(w.get("title", []) or []))
        authors = _split_authors(w.get("author", []))
        year = None
        if w.get("issued", {}).get("date-parts"):
            try:
                year = int(w["issued"]["date-parts"][0][0])
            except Exception:
                year = None
        journal = _norm((w.get("container-title") or [None])[0])
        url = w.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        references = w.get("reference", []) or []
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "doi": w.get("DOI") or doi,
            "url": url,
            "references": references,
        }
    except Exception as e:
        return {"error": f"resolve_doi_failed: {e}"}


@tool("citation_chain")
def citation_chain(doi: str, direction: str = "references", depth: int = 1, max_per_node: int = 10) -> Dict:
    """Traverse citations via OpenAlex. direction="references" or "citations". Returns a shallow graph of works.

    Each node has: openalex_id, doi, title, year, authors (names), url.
    """
    if requests is None:
        return {"error": "requests not installed"}
    try:
        doi = doi.strip().lower().replace("doi:", "").strip()
        def fetch_work(doi_val: str) -> Optional[Dict]:
            u = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi_val)}"
            r = _http_get(u).json()
            if r.get("id") is None:
                return None
            title = _norm(r.get("title"))
            year = r.get("publication_year")
            authors = [
                _norm(a.get("author", {}).get("display_name"))
                for a in r.get("authorships", [])
                if _norm(a.get("author", {}).get("display_name"))
            ]
            return {
                "openalex_id": r.get("id"),
                "doi": r.get("doi") or doi_val,
                "title": title,
                "year": year,
                "authors": authors,
                "url": r.get("primary_location", {}).get("landing_page_url") or r.get("open_access", {}).get("oa_url"),
                "referenced_works": r.get("referenced_works", []),
                "cited_by_count": r.get("cited_by_count", 0),
                "cited_by_api_url": r.get("cited_by_api_url"),
            }

        root = fetch_work(doi)
        if not root:
            return {"error": "work_not_found", "nodes": [], "edges": []}
        nodes = {root["openalex_id"]: root}
        edges = []

        frontier = [(root, 0)]
        while frontier:
            node, d = frontier.pop(0)
            if d >= depth:
                continue
            if direction == "references":
                refs = node.get("referenced_works", [])[: max_per_node]
                for wid in refs:
                    try:
                        r = _http_get(wid).json()
                        doi_v = (r.get("doi") or "").replace("https://doi.org/", "")
                        child = {
                            "openalex_id": r.get("id"),
                            "doi": doi_v or None,
                            "title": _norm(r.get("title")),
                            "year": r.get("publication_year"),
                            "authors": [
                                _norm(a.get("author", {}).get("display_name"))
                                for a in r.get("authorships", [])
                                if _norm(a.get("author", {}).get("display_name"))
                            ],
                            "url": r.get("primary_location", {}).get("landing_page_url") or r.get("open_access", {}).get("oa_url"),
                        }
                        if child["openalex_id"] not in nodes:
                            nodes[child["openalex_id"]] = child
                            frontier.append((child, d + 1))
                        edges.append({"from": node["openalex_id"], "to": child["openalex_id"], "type": "references"})
                    except Exception:
                        continue
            else:
                api = node.get("cited_by_api_url")
                if not api:
                    continue
                try:
                    cited = _http_get(api, params={"per_page": str(max_per_node)}).json()
                    for r in cited.get("results", [])[: max_per_node]:
                        doi_v = (r.get("doi") or "").replace("https://doi.org/", "")
                        child = {
                            "openalex_id": r.get("id"),
                            "doi": doi_v or None,
                            "title": _norm(r.get("title")),
                            "year": r.get("publication_year"),
                            "authors": [
                                _norm(a.get("author", {}).get("display_name"))
                                for a in r.get("authorships", [])
                                if _norm(a.get("author", {}).get("display_name"))
                            ],
                            "url": r.get("primary_location", {}).get("landing_page_url") or r.get("open_access", {}).get("oa_url"),
                        }
                        if child["openalex_id"] not in nodes:
                            nodes[child["openalex_id"]] = child
                            frontier.append((child, d + 1))
                        edges.append({"from": child["openalex_id"], "to": node["openalex_id"], "type": "cites"})
                except Exception:
                    continue

        return {"nodes": list(nodes.values()), "edges": edges}
    except Exception as e:
        return {"error": f"citation_chain_failed: {e}", "nodes": [], "edges": []}


@tool("dedupe_records")
def dedupe_records(records: list) -> Dict:
    """Dedupe literature records by DOI/arXiv/PMID/title. Input: list of dicts.

    Returns a dict with 'items' unique on a composite key. Prefers entries with DOI and longer abstracts.
    """
    def norm_title(t: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]

    seen = {}
    for r in records or []:
        doi = _norm(r.get("doi"))
        aid = _norm(r.get("arxiv_id"))
        pmid = _norm(r.get("pmid") or r.get("id")) if str(r.get("source", "")).lower() == "pubmed" else None
        title_key = norm_title(r.get("title"))
        key = doi or aid or pmid or title_key
        if not key:
            continue
        prev = seen.get(key)
        if not prev:
            seen[key] = r
            continue
        # Prefer with DOI, then with url, then with longer text/abstract
        def quality(x: dict) -> int:
            q = 0
            if _norm(x.get("doi")):
                q += 3
            if _norm(x.get("url")):
                q += 1
            if x.get("abstract"):
                q += min(3, len(str(x.get("abstract"))) // 200)
            return q
        if quality(r) > quality(prev):
            seen[key] = r

    return {"items": list(seen.values())}

