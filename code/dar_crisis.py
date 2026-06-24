#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crisis-tier & augmentation utilities

- Public-domain fetchers (NIMH) → lightweight external corpus for crisis fallback
- CrisisResources: region-aware hotlines/web resources from a local JSON file
- CrisisResponder: renders structured crisis responses (hotlines + optional snippets)
- Coverage heuristics to decide when to promote to crisis tier

Notes
-----
* The fetched NIMH content is used ONLY for fallback snippets and is kept separate
  from the evaluation corpus to avoid contaminating IR metrics.
* If 'requests' or 'bs4' are unavailable, functions degrade gracefully.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

# Optional deps
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

# Local
from dar_utils import Config
from dar_core import BM25Index


# ======================================================
# HTML utils & HTTP
# ======================================================

def _html_to_text(html: str) -> str:
    """Robustly extract visible text from HTML (drops nav/footer/scripts)."""
    if BeautifulSoup is None:
        # Simple fallback: strip tags
        return re.sub(r"<[^>]+>", " ", html or "")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text

def _fetch(url: str, timeout: int = 25) -> Optional[str]:
    """Fetch text/html; returns None on error or missing deps."""
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "dar-router/1.0"})
        if r.ok and "text" in r.headers.get("content-type", ""):
            return r.text
    except Exception:
        return None
    return None


# ======================================================
# Public-domain dataset fetcher (NIMH)
# ======================================================

def fetch_nimh_pages(max_pages: int = 60) -> List[Tuple[str, str, str]]:
    """
    Crawl a small subset of NIMH Health Topics / publications (public domain).
    Returns: list of (source_url, title, text)
    """
    base = "https://www.nimh.nih.gov"
    index_urls = [
        "https://www.nimh.nih.gov/health/topics",
        "https://www.nimh.nih.gov/health/publications",
    ]
    if BeautifulSoup is None or requests is None:
        return []

    seen, rows = set(), []
    for idx in index_urls:
        html = _fetch(idx)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin(base, href)
            if urlparse(href).netloc != urlparse(base).netloc:
                continue
            if "/health/" not in href:
                continue
            if href in seen:
                continue
            seen.add(href)
            if len(rows) >= max_pages:
                break
            page = _fetch(href)
            if not page:
                continue
            title = a.get_text(strip=True) or "NIMH Topic"
            text = _html_to_text(page)
            if len(text) < 400:
                continue
            rows.append((href, title, text))
        if len(rows) >= max_pages:
            break
    return rows

def build_crisis_docs_from_pages(pages: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Convert (url, title, text) → docs_df-like structure:
    columns: doc_id, qid, text
    """
    rows, did = [], 0
    for url, title, text in pages:
        qid = f"AUG::{url}"
        rows.append((did, qid, f"{title}. {text}"))
        did += 1
    return pd.DataFrame(rows, columns=["doc_id", "qid", "text"])


# ======================================================
# Low-coverage heuristic
# ======================================================

def estimate_low_coverage(bm25_hits: List[Tuple[int, float]]) -> bool:
    """
    Very simple BM25 score heuristic:
      - low mean score and low dispersion → likely poor coverage.
    """
    if not bm25_hits:
        return True
    scores = np.array([s for _, s in bm25_hits[:20]], dtype=float)
    if scores.size == 0:
        return True
    mean = float(scores.mean())
    nqc = float(scores.std() / (abs(mean) + 1e-9))
    return (mean < 0.05) and (nqc < 0.5)


# ======================================================
# Crisis resources
# ======================================================

class CrisisResources:
    """
    Loads a simple mapping of region → {hotlines, web, note}.
    JSON schema:
        {
          "US": {
            "note": "...",
            "hotlines": [{"name":"988...", "value":"Call/text 988"}],
            "web": [{"name":"988", "url":"https://..."}]
          },
          "GLOBAL": {...}
        }
    """

    def __init__(self, mapping: Dict[str, Any]):
        self.map = mapping or {}

    @classmethod
    def from_json(cls, path: Path) -> "CrisisResources":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        return cls(data)

    def for_country(self, cc: Optional[str]) -> Dict[str, Any]:
        cc = (cc or "").upper()
        return self.map.get(cc, self.map.get("GLOBAL", {}))


# ======================================================
# Crisis responder
# ======================================================

class CrisisResponder:
    """
    Produces a structured crisis response object including region-aware resources
    and optional evidence-based snippets from a crisis corpus.
    """

    def __init__(self,
                 resources: CrisisResources,
                 crisis_docs_df: Optional[pd.DataFrame] = None,
                 crisis_bm25: Optional[BM25Index] = None,
                 cfg: Optional[Config] = None):
        self.resources = resources
        self.docs_df = crisis_docs_df if crisis_docs_df is not None else pd.DataFrame(columns=["doc_id", "qid", "text"])
        self.idx = crisis_bm25
        self.cfg = cfg or Config(workdir=Path("."))

        # Map doc_id → text for robust lookup
        if not self.docs_df.empty and "doc_id" in self.docs_df.columns:
            self._id2text = self.docs_df.set_index("doc_id")["text"].to_dict()
        else:
            self._id2text = {}

    def _snippets_for(self, query: str, topn: int) -> List[Dict[str, Any]]:
        if self.idx is None or not self._id2text:
            return []
        hits = self.idx.search([query], topk=min(topn, len(self._id2text)))[0]
        out: List[Dict[str, Any]] = []
        for did, sc in hits[:min(3, topn)]:
            text = self._id2text.get(int(did))
            if not text:
                continue
            out.append({
                "score": float(sc),
                "excerpt": text[:700] + ("..." if len(text) > 700 else "")
            })
        return out

    def respond(self, query: str, qid: Optional[str] = None, country: Optional[str] = None) -> Dict[str, Any]:
        cc_info = self.resources.for_country(country)
        payload: Dict[str, Any] = {
            "qid": qid,
            "query": query,
            "region": (country or "").upper() or None,
            "hotlines": cc_info.get("hotlines", []),
            "web": cc_info.get("web", []),
            "note": cc_info.get("note"),
            "snippets": self._snippets_for(query, topn=self.cfg.crisis_topk)
        }
        return payload

    def batch_respond(self, queries: List[str], qids: Optional[List[str]] = None, country: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, q in enumerate(queries):
            qid = qids[i] if qids and i < len(qids) else None
            out.append(self.respond(q, qid=qid, country=country))
        return out


# ======================================================
# Convenience builder
# ======================================================

def build_crisis_components(augment_nimh: bool,
                            workdir: Path,
                            resources_path: Optional[str],
                            cfg: Config) -> Tuple[CrisisResources, pd.DataFrame, Optional[BM25Index]]:
    """
    Creates CrisisResources + (optional) crisis corpus & index.
    Returns: (resources, crisis_docs_df, crisis_bm25_index_or_none)
    """
    # Load resources
    res = CrisisResources.from_json(Path(resources_path)) if resources_path else CrisisResources({})

    # Optional NIMH augmentation
    crisis_docs_df = pd.DataFrame(columns=["doc_id", "qid", "text"])
    crisis_bm25: Optional[BM25Index] = None

    if augment_nimh:
        print("[augment] fetching NIMH pages (public domain)...")
        try:
            pages = fetch_nimh_pages(max_pages=60)
            crisis_docs_df = build_crisis_docs_from_pages(pages)
            crisis_docs_df.to_parquet(workdir / "crisis_docs.parquet", index=False)
            if len(crisis_docs_df):
                crisis_bm25 = BM25Index(
                    crisis_docs_df["text"].tolist(),
                    crisis_docs_df["doc_id"].tolist()
                )
            print(f"[augment] crisis corpus pages={len(crisis_docs_df)}")
        except Exception as e:
            print(f"[augment] NIMH fetch failed: {e} (continuing without crisis docs)")

    return res, crisis_docs_df, crisis_bm25
