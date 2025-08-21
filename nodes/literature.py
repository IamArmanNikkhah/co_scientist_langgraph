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
    web_search_perplexity,
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


def _apply_perplexity_priority_selection(items: List[Dict[str, Any]], sources: List[str], max_items: int = 6) -> List[Dict[str, Any]]:
    """Apply Perplexity-specific priority selection to ensure web search results are included when requested.
    
    If 'perplexity' is in the requested sources and we have Perplexity results, guarantee at least one
    Perplexity result is included in the selection, then fill remaining slots with other sources.
    
    Args:
        items: List of deduplicated literature items
        sources: List of requested source types
        max_items: Maximum number of items to select for processing
        
    Returns:
        List of selected items with Perplexity prioritized if requested
    """
    # Separate Perplexity from other sources
    perplexity_items = [item for item in items if item.get('source') == 'perplexity']
    other_items = [item for item in items if item.get('source') != 'perplexity']
    
    selected = []
    
    # If Perplexity was requested and we have Perplexity results, guarantee inclusion
    if "perplexity" in sources and perplexity_items:
        selected.append(perplexity_items[0])  # Take the first (and likely only) Perplexity result
        remaining_slots = max_items - 1
    else:
        remaining_slots = max_items
    
    # Fill remaining slots with best quality items from other sources
    selected.extend(other_items[:remaining_slots])
    
    return selected


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

        # Optional: Perplexity web search (Sonar-Pro) to gather up-to-date web context
        try:
            if "perplexity" in sources:
                px = web_search_perplexity.invoke({"query": query, "focus": goal, "max_tokens": 800}) or {}
                if px.get("content"):
                    records.append({
                        "source": "perplexity",
                        "id": "perplexity-web-answer",
                        "title": f"Perplexity Answer: {query[:60]}",
                        "authors": [],
                        "year": None,
                        "journal": "web",
                        "doi": None,
                        "url": None,
                        "abstract": _limit_text(px.get("content", ""), 4000),
                    })
        except Exception as e:
            safe_append_error(s, f"Literature: perplexity search failed: {e}")

        # 2) Dedupe
        try:
            deduped = dedupe_records.invoke({"records": records}) or {"items": []}
            items: List[Dict[str, Any]] = deduped.get("items", [])
        except Exception as e:
            safe_append_error(s, f"Literature: dedupe failed: {e}")
            items = records

        # 3) Fetch and extract a limited set with Perplexity priority
        extracted: List[Dict[str, Any]] = []
        to_fetch = _apply_perplexity_priority_selection(items, sources, 6)
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
                    "you must return a JSON OBJECT (not an array or other structure) with the exact schema below.\n\n"
                    "REQUIRED JSON OBJECT SCHEMA:\n"
                    "{\n"
                    '  "title": "string - the article title",\n'
                    '  "year": "string or number - publication year",\n'
                    '  "citation": "string - URL, DOI, or citation info",\n'
                    '  "key_findings": ["array", "of", "string", "findings"],\n'
                    '  "relevance_to_goal": "string - how this relates to the research goal",\n'
                    '  "methodology_notes": "string - notes about methodology used",\n'
                    '  "limitations": "string - study limitations or constraints"\n'
                    "}\n\n"
                    f"Research Goal: {goal}\n\n"
                    f"Article Title: {rec.get('title','')}\n"
                    f"Year: {rec.get('year','')}\n"
                    f"Citation URL/DOI: {rec.get('url') or rec.get('doi') or ''}\n\n"
                    f"Extracted Text (may be partial):\n{_limit_text(rec.get('extracted_text',''), 3500)}\n\n"
                    "CRITICAL: Return ONLY a valid JSON object matching the exact schema above. Do not return arrays, strings, or other formats."
                )
                resp = await llm.ainvoke(prompt)
                raw = getattr(resp, "content", str(resp))
                js = extract_json_from_text.invoke({"text": raw}) or raw
                data = json.loads(js)
                # Ensure data is a dictionary, not a list or other type
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dictionary but got {type(data).__name__}")
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





