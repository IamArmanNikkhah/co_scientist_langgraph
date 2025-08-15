from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from ..state import GraphState, safe_append_error
from ..tools import (
    search_pubmed,
    search_crossref,
    search_arxiv,
    search_scholar,
    fetch_url,
    extract_text_from_html,
    extract_text_from_pdf,
    resolve_doi,
    dedupe_records,
    extract_json_from_text,
)


def _limit_text(text: str, max_chars: int = 5000) -> str:
    t = (text or "").strip()
    if len(t) > max_chars:
        return t[: max_chars - 3] + "..."
    return t


def make_literature_node(llm: ChatOpenAI):
    async def literature_node(state: GraphState) -> GraphState:
        s = state
        goal = (s.get("research_goal") or "").strip()
        decision = s.get("decision", {}) or {}
        params = decision.get("parameters", {}) or s.get("parameters", {}) or {}

        query = (params.get("search_query") or goal).strip()
        if not query:
            safe_append_error(s, "Literature: empty search query (research_goal missing).")
            return {}
        sources: List[str] = params.get("sources", ["pubmed", "crossref", "arxiv"])  # type: ignore
        try:
            max_results = int(params.get("max_results", 12))
        except Exception:
            max_results = 12

        # 1) Search across selected sources
        records: List[Dict[str, Any]] = []
        try:
            if "pubmed" in sources:
                pm = search_pubmed.invoke({"query": query, "max_results": max_results}) or {}
                records.extend(pm.get("items", []))
        except Exception as e:
            safe_append_error(s, f"Literature: pubmed search failed: {e}")

        try:
            if "crossref" in sources:
                cr = search_crossref.invoke({"query": query, "max_results": max_results}) or {}
                records.extend(cr.get("items", []))
        except Exception as e:
            safe_append_error(s, f"Literature: crossref search failed: {e}")

        try:
            if "arxiv" in sources:
                ax = search_arxiv.invoke({"query": query, "max_results": max_results}) or {}
                records.extend(ax.get("items", []))
        except Exception as e:
            safe_append_error(s, f"Literature: arxiv search failed: {e}")

        try:
            if "scholar" in sources:
                sc = search_scholar.invoke({"query": query, "max_results": max_results}) or {}
                records.extend(sc.get("items", []))
        except Exception as e:
            safe_append_error(s, f"Literature: scholar search failed: {e}")

        # 2) Dedupe
        try:
            deduped = dedupe_records.invoke({"records": records}) or {"items": []}
            items: List[Dict[str, Any]] = deduped.get("items", [])
        except Exception as e:
            safe_append_error(s, f"Literature: dedupe failed: {e}")
            items = records

        # 3) Fetch and extract a limited set
        extracted: List[Dict[str, Any]] = []
        to_fetch = items[: min(6, len(items))]
        for rec in to_fetch:
            url = rec.get("url")
            doi = rec.get("doi")
            text = ""
            try:
                if not url and doi:
                    meta = resolve_doi.invoke({"doi": doi}) or {}
                    url = meta.get("url") or url
                if url:
                    fetched = fetch_url.invoke({"url": url}) or {}
                    if fetched.get("content_b64"):
                        ct = (fetched.get("content_type") or "").lower()
                        if "pdf" in ct:
                            tex = extract_text_from_pdf.invoke({"content_b64": fetched.get("content_b64")}) or {}
                            text = tex.get("text", "")
                        else:
                            import base64

                            raw_bytes = base64.b64decode(fetched.get("content_b64"))
                            html = raw_bytes.decode("utf-8", errors="ignore")
                            tex = extract_text_from_html.invoke({"html": html}) or {}
                            text = tex.get("text", "")
            except Exception as e:
                safe_append_error(s, f"Literature: fetch/extract failed for {url or doi}: {e}")

            rec_out = {
                **rec,
                "extracted_text": _limit_text(text, 12000),
            }
            extracted.append(rec_out)

        # 4) LLM reasoning per article
        articles_with_reasoning: List[Dict[str, Any]] = []
        for rec in extracted:
            try:
                prompt = (
                    "You are compiling literature for a research system. Given a research goal and an article's extracted text, "
                    "produce a single JSON with keys: title, year, citation, key_findings (list), relevance_to_goal, methodology_notes, limitations.\n\n"
                    f"Research Goal: {goal}\n\n"
                    f"Article Title: {rec.get('title','')}\n"
                    f"Year: {rec.get('year','')}\n"
                    f"Citation URL/DOI: {rec.get('url') or rec.get('doi') or ''}\n\n"
                    f"Extracted Text (may be partial):\n{_limit_text(rec.get('extracted_text',''), 3500)}\n\n"
                    "Return ONLY the JSON."
                )
                resp = await llm.ainvoke(prompt)
                raw = getattr(resp, "content", str(resp))
                js = extract_json_from_text.invoke({"text": raw}) or raw
                data = json.loads(js)
                data["source_record"] = {k: rec.get(k) for k in ["source", "id", "doi", "url", "authors", "journal", "arxiv_id"]}
                articles_with_reasoning.append(data)
            except Exception as e:
                safe_append_error(s, f"Literature: reasoning failed for {rec.get('title','?')}: {e}")
                # Fallback minimal
                articles_with_reasoning.append({
                    "title": rec.get("title", "Untitled"),
                    "year": rec.get("year"),
                    "citation": rec.get("url") or rec.get("doi"),
                    "key_findings": [],
                    "relevance_to_goal": "",
                    "methodology_notes": "",
                    "limitations": "",
                    "source_record": {k: rec.get(k) for k in ["source", "id", "doi", "url", "authors", "journal", "arxiv_id"]},
                })

        # 5) Build chronology text (append to existing if provided)
        def _year(x):
            try:
                return int(x.get("year") or 0)
            except Exception:
                return 0

        ordered = sorted(articles_with_reasoning, key=_year)
        # Start with existing chronology if any
        existing = (s.get("articles_with_reasoning_text") or "").strip()
        lines = [ln for ln in existing.splitlines() if ln.strip()] if existing else []
        for a in ordered:
            yr = a.get("year") or ""
            ttl = a.get("title") or "Untitled"
            rel = (a.get("relevance_to_goal") or "").strip()
            if len(rel) > 220:
                rel = rel[:217] + "..."
            lines.append(f"- [{yr}] {ttl} — {rel}")
        chronology_text = "\n".join(lines)

        return {
            "articles_with_reasoning": (s.get("articles_with_reasoning") or []) + articles_with_reasoning,
            "articles_with_reasoning_text": chronology_text,
        }

    return literature_node





