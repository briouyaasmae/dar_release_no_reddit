#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities, configuration, constants, and lightweight helpers.

This module is intentionally dependency-light so that other modules
(dar_core.py, dar_crisis.py, dar_router_main.py) can import from here
without pulling in heavy ML frameworks prematurely.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Reproducibility seed
# ---------------------------
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# JSON helpers
# ---------------------------
class NumpyPathEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles numpy scalars/arrays & pathlib Paths."""
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

# ---------------------------
# Sentence splitting
# ---------------------------
# Prefer nltk punkt if available; otherwise fall back to a simple regex.
try:
    import nltk  # type: ignore
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize as SENT_SPLIT  # type: ignore
except Exception:
    def SENT_SPLIT(text: str) -> List[str]:
        # Split on punctuation followed by space and an alnum.
        return re.split(r"(?<=[.!?])\s+(?=[A-Za-z0-9])", text or "")

# ---------------------------
# Tokens & lightweight text utils
# ---------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+")

def simple_tokens(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

# ---------------------------
# Bootstrap utilities
# ---------------------------
def bootstrap_ci_mean(values: List[float],
                      n_resamples: int = 1000,
                      alpha: float = 0.05,
                      seed: int = SEED) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for the mean."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    N = arr.size
    means = []
    for _ in range(n_resamples):
        idx = rng.randint(0, N, N)
        means.append(float(np.mean(arr[idx])))
    low = float(np.percentile(means, 100 * (alpha / 2.0)))
    high = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return low, high

def bootstrap_ci_corr(x: np.ndarray,
                      y: np.ndarray,
                      kind: str = "pearson",
                      n_resamples: int = 1000,
                      alpha: float = 0.05,
                      seed: int = SEED) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for a correlation (pearson|spearman)."""
    rng = np.random.RandomState(seed)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    N = x.size
    vals: List[float] = []
    for _ in range(n_resamples):
        idx = rng.randint(0, N, N)
        xi, yi = x[idx], y[idx]
        if kind == "pearson":
            if xi.std() == 0 or yi.std() == 0:
                v = 0.0
            else:
                v = float(np.corrcoef(xi, yi)[0, 1])
        else:
            ra = xi.argsort().argsort()
            rb = yi.argsort().argsort()
            if ra.std() == 0 or rb.std() == 0:
                v = 0.0
            else:
                v = float(np.corrcoef(ra, rb)[0, 1])
        vals.append(v)
    low = float(np.percentile(vals, 100 * (alpha / 2.0)))
    high = float(np.percentile(vals, 100 * (1.0 - alpha / 2.0)))
    return low, high

# ---------------------------
# Version / environment stamp
# ---------------------------
def get_version_info(device_used: str) -> Dict[str, Any]:
    def _ver(modname: str) -> str:
        try:
            mod = __import__(modname)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "unknown"

    # torch / cuda
    try:
        import torch  # type: ignore
        torch_ver = torch.__version__
        cuda_avail = bool(torch.cuda.is_available())
        cuda_ver = torch.version.cuda
    except Exception:
        torch_ver, cuda_avail, cuda_ver = "unknown", False, None

    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": _ver("lightgbm"),
        "transformers": _ver("transformers"),
        "sentence_transformers": _ver("sentence_transformers"),
        "faiss": _ver("faiss"),
        "rank_bm25": _ver("rank_bm25"),
        "torch": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "device_used": device_used,
        "seed": SEED
    }
    return info

# ---------------------------
# Fixed question-type set
# ---------------------------
QTYPE_LIST: List[str] = [
    "how", "what", "why", "when", "where", "who", "which",
    "can", "should", "could", "would", "is", "are", "do", "does", "other"
]

# ---------------------------
# Config (centralized)
# ---------------------------
@dataclass
class Config:
    # IO
    workdir: Path

    # Runtime / device
    device: str = "auto"        # "cpu" | "cuda" | "auto"
    seed: int = SEED

    # Corpus / chunking
    chunk_mode: str = "sentence"      # "sentence" | "semantic" | "overlap"
    max_docs: int = 200_000
    # Bumped defaults slightly to improve coverage
    min_chunk_words: int = 150
    max_chunk_words: int = 350
    overlap_size: int = 100            # for overlap mode
    overlap_chunk_words: int = 220     # for overlap mode

    # Retrieval
    topk_bm25: int = 200
    topk_dense: int = 200
    topk_fusion: int = 200
    eval_k: int = 10
    rrf_k: int = 60

    # Verified rerank
    use_verified: bool = True
    verified_topn: int = 50
    verified_max_len: int = 256

    # Costs (arbitrary units)
    cost_sparse: float = 1.0
    cost_hybrid: float = 4.0
    cost_verified: float = 10.0

    # Routing thresholds (UPDATED DEFAULTS)
    thr_sparse: float = 0.50
    thr_hybrid: float = 0.55
    safety_override_thr: float = 0.55
    high_risk_prob_thr: float = 0.30
    dual_delta: float = 0.00

    # Training / evaluation
    kfolds: int = 5
    limit_queries: int = 0

    # Optional experiments
    sweep: bool = True
    run_baselines: bool = True
    run_ablations: bool = True
    run_sims: bool = True
    run_transfer: bool = True
    run_chunk_ablation: bool = False

    # QPP post-retrieval feature K (for train + inference)
    qpp_post_k: int = 50

    # Crisis tier (fallback) -----------------------------------------
    enable_crisis_tier: bool = True
    cost_crisis: float = 0.5
    crisis_topk: int = 8
    crisis_resources_path: Optional[str] = None  # path to crisis_resources.json
    user_country: Optional[str] = None          # e.g., "US","GB","CA","MA"
    hybrid_mode: str = "rrf"        # "rrf" | "dense_only" | "bm25_only"
    verified_source: str = "hybrid" # "hybrid" | "dense" | "sparse"
    rrf_w_bm25: float = 1.0         # weight for BM25 in weighted RRF
    rrf_w_dense: float = 1.0        # weight for Dense in weighted RRF
