#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core IR/QPP components:
- Data IO & cleaning (CounselChat)
- Chunking & corpus building
- Indices (BM25, Dense + FAISS), RRF fusion, Cross-Encoder rerank
- Feature engineering (pre-, embedding-, post-retrieval; emotion & risk)
- Predictors (LightGBM + isotonic calibration)
- Routing policies and simulation
- Evaluation metrics & baselines
- Analyses helpers (adversarial consistency, regrets, chunk ablation)
"""

from __future__ import annotations

import gc
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from tqdm import tqdm

# External IR libs
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local utils
from dar_utils import (
    SEED,
    NumpyPathEncoder,
    SENT_SPLIT,
    simple_tokens,
    QTYPE_LIST,
    Config,
    bootstrap_ci_mean,
    bootstrap_ci_corr,
)

# ======================================================
# Data: CounselChat
# ======================================================

def download_counselchat_csv(dst: Path) -> Path:
    """
    Downloads combined CounselChat CSV. If GitHub is blocked, you can put a CSV at dst manually.
    """
    url = "https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/combined-data.csv"
    import urllib.request
    try:
        print(f"[data] downloading CounselChat from {url}")
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        raise RuntimeError(
            "Download failed. Place the combined-data.csv manually at: "
            f"{dst}\nOriginal error: {e}"
        )
    return dst

def load_and_clean(data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    keep_cols = [c for c in ["questionID","questionTitle","questionText","answerText"] if c in df.columns]
    if len(keep_cols) < 3:
        raise ValueError(f"Unexpected columns. Found: {df.columns.tolist()}")
    df = df[keep_cols].rename(columns={"questionID":"qid","questionTitle":"qtitle","questionText":"qtext","answerText":"answer"})
    df["qid"] = df["qid"].astype(str)
    df["qtext"] = (df.get("qtitle","").fillna("").astype(str).str.strip() + " " + df.get("qtext","").fillna("").astype(str).str.strip()).str.strip()
    df["qtext"] = df["qtext"].str.replace(r"\s+", " ", regex=True)
    df["answer"] = df["answer"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["qtext"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]
    
    df = df.drop_duplicates(subset=["qid","answer"])
    return df

# ======================================================
# Chunking
# ======================================================

def chunk_sentence_aware(text: str, min_words: int, max_words: int) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if min_words > max_words:
        min_words, max_words = max_words, min_words

    sents = SENT_SPLIT(text)
    chunks, cur, cnt = [], [], 0

    def flush(force=False):
        nonlocal cur, cnt
        if not cur: return
        if force or cnt >= min_words or not chunks:
            chunks.append(" ".join(cur)); cur=[]; cnt=0
        else:
            chunks[-1] = chunks[-1] + " " + " ".join(cur); cur=[]; cnt=0

    for s in sents:
        ws = s.strip().split()
        wlen = len(ws)
        if wlen > max_words:
            if cnt >= min_words:
                flush(force=True)
            for i in range(0, wlen, max_words):
                seg = ws[i:i+max_words]
                if len(seg) >= min_words:
                    chunks.append(" ".join(seg))
                else:
                    cur.append(" ".join(seg)); cnt += len(seg)
            continue
        if cnt + wlen <= max_words:
            cur.append(s); cnt += wlen
        else:
            if cnt >= min_words:
                flush(force=True)
                cur.append(s); cnt = wlen
            else:
                cur.append(s); cnt += wlen
                if cnt >= min_words: flush(force=True)

    if cur:
        if chunks and cnt < min_words:
            chunks[-1] = chunks[-1] + " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks or [text]

def chunk_semantic(text: str, min_words: int = 150, max_words: int = 350) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    paras = re.split(r'\n\s*\n+', text)
    chunks, cur, cnt = [], [], 0
    for para in paras:
        sents = SENT_SPLIT(para)
        for s in sents:
            n = len(s.split())
            if n > max_words:
                if cnt >= min_words:
                    chunks.append(" ".join(cur)); cur=[]; cnt=0
                ws = s.split()
                for i in range(0, len(ws), max_words):
                    seg = ws[i:i+max_words]
                    if len(seg) >= min_words:
                        chunks.append(" ".join(seg))
                    else:
                        cur.append(" ".join(seg)); cnt += len(seg)
                continue
            if cnt + n > max_words:
                if cnt >= min_words:
                    chunks.append(" ".join(cur)); cur=[s]; cnt=n
                else:
                    cur.append(s); cnt += n
                    chunks.append(" ".join(cur)); cur=[]; cnt=0
            else:
                cur.append(s); cnt += n
        if cnt >= min_words:
            chunks.append(" ".join(cur)); cur=[]; cnt=0
    if cur and cnt >= min_words*0.7:
        chunks.append(" ".join(cur))
    elif cur and chunks:
        chunks[-1] += " " + " ".join(cur)
    return chunks or [text]

def chunk_overlap(text: str, chunk_size: int = 220, overlap: int = 100) -> List[str]:
    w = (text or "").split()
    if not w: return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(w):
        seg = w[i:i+chunk_size]
        if len(seg) >= chunk_size * 0.5:
            chunks.append(" ".join(seg))
        i += step
    return chunks or [" ".join(w)]

def build_corpus(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, qrels, did = [], [], 0
    cmode = cfg.chunk_mode

    for qid, g in tqdm(df.groupby("qid"), desc=f"[chunk] mode={cmode} per question"):
        for ans in g["answer"].tolist():
            if cmode == "sentence":
                chunks = chunk_sentence_aware(ans, cfg.min_chunk_words, cfg.max_chunk_words)
            elif cmode == "semantic":
                chunks = chunk_semantic(ans, min_words=cfg.min_chunk_words, max_words=cfg.max_chunk_words)
            elif cmode == "overlap":
                chunks = chunk_overlap(ans, chunk_size=cfg.overlap_chunk_words, overlap=cfg.overlap_size)
            else:
                chunks = chunk_sentence_aware(ans, cfg.min_chunk_words, cfg.max_chunk_words)

            for ch in chunks:
                rows.append((did, qid, ch))
                qrels.append((qid, did, 3))
                did += 1

    docs_df = pd.DataFrame(rows, columns=["doc_id", "qid", "text"])
    qrels_df = pd.DataFrame(qrels, columns=["qid", "doc_id", "rel"])

    # If we cap the corpus, sample DOCS and QRELS consistently
    if len(docs_df) > cfg.max_docs:
        keep_ids = (
            docs_df.sample(cfg.max_docs, random_state=SEED)["doc_id"]
            .astype(int)
            .tolist()
        )
        keep_ids = set(keep_ids)
        docs_df = docs_df[docs_df["doc_id"].isin(keep_ids)].copy()
        qrels_df = qrels_df[qrels_df["doc_id"].isin(keep_ids)].copy()

        # Optional: keep things sorted by doc_id for readability
        docs_df = docs_df.sort_values("doc_id").reset_index(drop=True)
        qrels_df = qrels_df.sort_values(["qid", "doc_id"]).reset_index(drop=True)

    return docs_df, qrels_df

    
def rrf_fusion_weighted(run_bm25: List[List[Tuple[int,float]]],
                        run_dense: List[List[Tuple[int,float]]],
                        k: int, topk: int,
                        w_bm25: float = 1.0, w_dense: float = 1.0) -> List[List[Tuple[int,float]]]:
    out = []
    BIG = 10_000_000
    for hb, hd in zip(run_bm25, run_dense):
        rank_b = {doc: r for r, (doc, _) in enumerate(hb)}
        rank_d = {doc: r for r, (doc, _) in enumerate(hd)}
        docs = set(rank_b) | set(rank_d)
        scores = {}
        for d in docs:
            rb = rank_b.get(d, BIG)
            rd = rank_d.get(d, BIG)
            scores[d] = (w_bm25 / (k + 1 + rb)) + (w_dense / (k + 1 + rd))
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        out.append([(int(d), float(s)) for d, s in ranked])
    return out

# ======================================================
# Dense index (SentenceTransformer + FAISS cosine)
# ======================================================

class DenseIndex:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.index = None
        self.doc_ids = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True)
        return emb.astype("float32")

    def build(self, docs: List[str], ids: List[int]):
        embs = self._encode(docs)
        index = faiss.IndexFlatIP(embs.shape[1])  # cosine (with normalized vectors)
        index.add(embs)
        self.index = index
        self.doc_ids = np.array(ids, dtype=np.int64)

    def search(self, queries: List[str], topk: int) -> List[List[Tuple[int,float]]]:
        qembs = self._encode(queries)
        sims, idxs = self.index.search(qembs, topk)
        out = []
        for i in range(len(queries)):
            hits = [(int(self.doc_ids[j]), float(sims[i, k])) for k, j in enumerate(idxs[i])]
            out.append(hits)
        return out

# ======================================================
# BM25 index
# ======================================================

class BM25Index:
    """
    BM25 wrapper that keeps an explicit mapping from internal positions
    to external doc_ids, so that sampling / reordering does not break
    evaluation or fusion.

    docs: list of document texts in the same order as `ids`
    ids:  list of external doc_ids (e.g., docs_df["doc_id"])
    """

    def __init__(self, docs: List[str], ids: Optional[List[int]] = None):
        self.docs = docs
        self.tokenized = [simple_tokens(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

        if ids is None:
            # Default to 0..N-1 if no ids supplied
            self.doc_ids = np.arange(len(docs), dtype=np.int64)
        else:
            if len(ids) != len(docs):
                raise ValueError(
                    f"BM25Index: len(ids)={len(ids)} does not match len(docs)={len(docs)}"
                )
            self.doc_ids = np.asarray(ids, dtype=np.int64)

    def search(self, queries: List[str], topk: int) -> List[List[Tuple[int, float]]]:
        out: List[List[Tuple[int, float]]] = []
        for q in queries:
            toks = simple_tokens(q)
            scores = self.bm25.get_scores(toks)
            if topk <= 0:
                idx = np.argsort(-scores)  # full ranking if needed
            else:
                idx = np.argsort(-scores)[:topk]
            hits = [(int(self.doc_ids[i]), float(scores[i])) for i in idx]
            out.append(hits)
        return out

# ======================================================
# RRF fusion
# ======================================================

def rrf_fusion(run_a: List[List[Tuple[int,float]]],
               run_b: List[List[Tuple[int,float]]],
               k: int, topk: int) -> List[List[Tuple[int, float]]]:
    out = []
    for ha, hb in zip(run_a, run_b):
        rank_a = {doc: r for r, (doc, _) in enumerate(ha)}
        rank_b = {doc: r for r, (doc, _) in enumerate(hb)}
        docs = set(rank_a.keys()) | set(rank_b.keys())
        scores = {}
        for d in docs:
            ra = rank_a.get(d, 10_000_000)
            rb = rank_b.get(d, 10_000_000)
            scores[d] = 1.0 / (k + 1 + ra) + 1.0 / (k + 1 + rb)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        out.append([(int(d), float(s)) for d, s in ranked])
    return out

# ======================================================
# Cross-encoder rerank (verified tier)
# ======================================================

class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", max_len: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_len = max_len

    @torch.no_grad()
    def rerank(self, query: str, cand_texts: List[str], topk: int) -> List[int]:
        if len(cand_texts) == 0:
            return []
        pairs = [(query, ct) for ct in cand_texts]
        enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits.squeeze(-1)
        scores = logits.detach().cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return order.tolist()

# ======================================================
# Metrics
# ======================================================

def dcg_at_k(rels: List[float], k: int) -> float:
    rels = np.array(rels[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2**rels - 1) * discounts))

def ndcg_at_k(rels: List[float], k: int) -> float:
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg

def precision_at_k(rels: List[float], k: int) -> float:
    rels = np.array(rels[:k], dtype=float)
    return float((rels > 0).mean()) if rels.size else 0.0

def recall_at_k(rels: List[float], total_rel: int, k: int) -> float:
    if total_rel <= 0:
        return 0.0
    rels = np.array(rels[:k], dtype=float)
    return float((rels > 0).sum() / total_rel)

# ======================================================
# Emotion & high-risk detection (transformer + compact lexicon)
# ======================================================

class EmotionDetector:
    """
    Multi-class emotion probabilities + compact high-risk phrase detector (with light negation).
    """
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base", device="cpu"):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.labels = self.model.config.id2label

        # Minimal high-risk lexicon
        self.risk_terms = [
            "suicidal", "suicide", "kill myself", "end my life",
            "no reason to live", "want to die", "hopeless",
            "self harm", "self-harm", "cutting myself", "hurt myself"
        ]
        self.neg_triggers = {"not", "n't", "no", "never", "without", "deny", "denying"}

    @torch.no_grad()
    def emotion_probs(self, texts: List[str]) -> np.ndarray:
        out = []
        bs = 16
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
        return np.vstack(out) if out else np.zeros((0, len(getattr(self.model.config, "id2label", {}))))

    def _negated(self, low_text: str, start_idx: int, end_idx: int) -> bool:
        toks = low_text.split()
        window = toks[max(0, start_idx-4): min(len(toks), end_idx+4)]
        return any(t in self.neg_triggers for t in window)

    def high_risk_score(self, text: str) -> float:
        low = (text or "").lower()
        toks = low.split()
        score = 0.0
        for term in self.risk_terms:
            pos = low.find(term)
            while pos != -1:
                pre = low[:pos]
                start = len(pre.split())
                end = start + len(term.split())
                if not self._negated(low, start, end):
                    score += 1.0
                pos = low.find(term, pos+1)
        return float(1 - math.exp(-score))  # compress to [0,1]

    def features_for_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        probs = self.emotion_probs(texts)
        if probs.size == 0:
            # handle empty input gracefully
            return {
                "emo_entropy": np.zeros(len(texts), dtype=float),
                "emo_dom": np.zeros(len(texts), dtype=float),
                "emo_maxprob": np.zeros(len(texts), dtype=float),
                "high_risk_prob": np.zeros(len(texts), dtype=float),
            }
        eps = 1e-8
        ent = -np.sum(probs * np.log(probs + eps), axis=1) / math.log(probs.shape[1])
        dom = probs.argmax(axis=1)
        risk = np.array([self.high_risk_score(t) for t in texts], dtype=float)
        return {
            "emo_entropy": ent.astype(float),
            "emo_dom": dom.astype(float),
            "emo_maxprob": probs.max(axis=1).astype(float),
            "high_risk_prob": risk
        }

# ======================================================
# QPP: pre-retrieval features
# ======================================================

def compute_query_features(queries: List[str]) -> pd.DataFrame:
    feats: Dict[str, Any] = {}
    qlens = np.array([len(q.split()) for q in queries], dtype=float)
    feats["q_len"] = qlens
    feats["q_avg_tok_len"] = np.array([0 if len(q.split())==0 else np.mean([len(t) for t in q.split()]) for q in queries], dtype=float)
    feats["q_exclam"] = np.array([q.count("!") for q in queries], dtype=float)
    feats["q_qmark"] = np.array([q.count("?") for q in queries], dtype=float)
    feats["q_first_person"] = np.array([1.0 if re.search(r"\b(i|i'm|i’ve|i feel|i am|my)\b", q.lower()) else 0.0 for q in queries], dtype=float)
    feats["q_negation"] = np.array([1.0 if re.search(r"\b(no|not|never|n't)\b", q.lower()) else 0.0 for q in queries], dtype=float)
    feats["q_future"] = np.array([1.0 if re.search(r"\b(will|going to|future)\b", q.lower()) else 0.0 for q in queries], dtype=float)
    feats["q_longtok_frac"] = np.array([0.0 if len(simple_tokens(q))==0 else np.mean([len(t)>=8 for t in simple_tokens(q)]) for q in queries], dtype=float)

    # fixed question-type one-hot
    def qtype(q):
        ql = q.lower().strip()
        for t in QTYPE_LIST[:-1]:
            if ql.startswith(t+" "): return t
        return "other"
    qtypes = np.array([qtype(q) for q in queries])
    for t in QTYPE_LIST:
        feats[f"qt_{t}"] = (qtypes==t).astype(float)

    return pd.DataFrame(feats)

def compute_embedding_feats(queries: List[str], dense_model: SentenceTransformer) -> pd.DataFrame:
    with torch.no_grad():
        emb = dense_model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return pd.DataFrame({
        "emb_norm": np.linalg.norm(emb, axis=1),
        "emb_var": emb.var(axis=1)
    })

def add_emotion_feats(df: pd.DataFrame, emotion_feats: Dict[str, np.ndarray]) -> pd.DataFrame:
    for k, v in emotion_feats.items():
        df[k] = v.astype(float)
    return df

# ======================================================
# QPP: post-retrieval dispersion & complementarity
# ======================================================

def _score_entropy(scores: np.ndarray) -> float:
    if scores.size == 0: return 0.0
    a = np.abs(scores)
    p = a / (a.sum() + 1e-9)
    return float(-(p * np.log(p + 1e-9)).sum())

def _nqc(scores: np.ndarray, k: int) -> float:
    if scores.size == 0: return 0.0
    top = scores[:min(k, scores.size)]
    denom = max(1e-9, abs(top.mean()))
    return float(top.std() / denom)

def postret_features_from_runs(bm25_hits: List[List[Tuple[int,float]]],
                               dense_hits: List[List[Tuple[int,float]]],
                               k: int = 50) -> pd.DataFrame:
    rows = []
    for b_hits, d_hits in zip(bm25_hits, dense_hits):
        b_scores = np.array([s for _, s in b_hits[:k]], dtype=float)
        d_scores = np.array([s for _, s in d_hits[:k]], dtype=float)

        # Dispersion features (BM25 and Dense)
        b_mean, b_std, b_ent, b_nqc = float(b_scores.mean()) if b_scores.size else 0.0, float(b_scores.std()) if b_scores.size else 0.0, _score_entropy(b_scores), _nqc(b_scores, 10)
        d_mean, d_std, d_ent, d_nqc = float(d_scores.mean()) if d_scores.size else 0.0, float(d_scores.std()) if d_scores.size else 0.0, _score_entropy(d_scores), _nqc(d_scores, 10)

        # Score gaps (top-1 vs 2 and vs k)
        def gaps(scores):
            if scores.size == 0: return 0.0, 0.0
            g12 = float(scores[0] - scores[1]) if scores.size >= 2 else float(scores[0])
            g1k = float(scores[0] - scores[min(9, scores.size-1)])
            return g12, g1k
        b_g12, b_g1k = gaps(b_scores)
        d_g12, d_g1k = gaps(d_scores)

        # Complementarity features (dense vs sparse)
        b_docs = [doc for doc,_ in b_hits[:k]]
        d_docs = [doc for doc,_ in d_hits[:k]]
        set_b, set_d = set(b_docs), set(d_docs)
        inter = len(set_b & set_d)
        union = len(set_b | set_d) if len(set_b | set_d) > 0 else 1
        overlap = inter / max(1, k)
        jaccard = inter / union

        # Mean rank gap over union (missing -> k+1)
        rank_b = {doc: i+1 for i, doc in enumerate(b_docs)}
        rank_d = {doc: i+1 for i, doc in enumerate(d_docs)}
        all_docs = list(set_b | set_d)
        kpad = k + 1
        mean_rank_gap = float(np.mean([abs(rank_b.get(doc, kpad) - rank_d.get(doc, kpad)) for doc in all_docs])) if all_docs else 0.0

        # Cross positions of top1
        pos_b1_in_d = rank_d.get(b_docs[0], kpad) if len(b_docs) else kpad
        pos_d1_in_b = rank_b.get(d_docs[0], kpad) if len(d_docs) else kpad

        rows.append({
            # BM25 dispersion
            "bm25_top_mean": b_mean, "bm25_top_std": b_std, "bm25_top_entropy": b_ent, "bm25_nqc": b_nqc,
            "bm25_gap12": b_g12, "bm25_gap1k": b_g1k,
            # Dense dispersion
            "dense_top_mean": d_mean, "dense_top_std": d_std, "dense_top_entropy": d_ent, "dense_nqc": d_nqc,
            "dense_gap12": d_g12, "dense_gap1k": d_g1k,
            # Complementarity
            "overlap_frac_k": overlap, "jaccard_k": jaccard,
            "mean_rank_gap": mean_rank_gap,
            "pos_b1_in_dense": float(pos_b1_in_d), "pos_d1_in_bm25": float(pos_d1_in_b),
            # Simple contrast features
            "nqc_diff_dense_minus_bm25": d_nqc - b_nqc,
            "std_diff_dense_minus_bm25": d_std - b_std
        })
    return pd.DataFrame(rows)

def postret_features_for_query(query: str,
                               bm25_index: "BM25Index",
                               dense_index: "DenseIndex",
                               k: int = 50) -> pd.DataFrame:
    # run small top-k searches (cheap) for features
    bm25_hits = bm25_index.search([query], topk=k)[0]
    dense_hits = dense_index.search([query], topk=k)[0]
    return postret_features_from_runs([bm25_hits], [dense_hits], k=k)

# ======================================================
# QPP: assemble full feature matrix
# ======================================================

def build_feature_frame(queries: List[str],
                        dense_model: SentenceTransformer,
                        emo_feats: Dict[str, np.ndarray],
                        bm25_hits: List[List[Tuple[int,float]]],
                        dense_hits: List[List[Tuple[int,float]]],
                        cfg: Config) -> pd.DataFrame:
    qf = compute_query_features(queries)
    ef = compute_embedding_feats(queries, dense_model)
    qf = pd.concat([qf, ef], axis=1)
    qf = add_emotion_feats(qf, emo_feats)
    pr = postret_features_from_runs(bm25_hits, dense_hits, k=cfg.qpp_post_k)
    X = pd.concat([qf.reset_index(drop=True), pr.reset_index(drop=True)], axis=1)
    return X

def ensure_feature_columns(df: pd.DataFrame, ref_columns: List[str]) -> pd.DataFrame:
    """Add any missing columns with 0.0, drop extras, and reorder to ref_columns."""
    for c in ref_columns:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in ref_columns]
    if extra:
        df = df.drop(columns=extra)
    return df[ref_columns]

# ======================================================
# Build runs
# ======================================================

def build_runs(queries: List[str],
               docs_df: pd.DataFrame,
               bm25: BM25Index,
               dense: DenseIndex,
               reranker: Optional[CrossEncoderReranker],
               cfg: Config) -> Dict[str, List[List[Tuple[int,float]]]]:
    id2doc = docs_df.set_index("doc_id")["text"].to_dict()

    bm25_hits = bm25.search(queries, topk=cfg.topk_bm25)
    dense_hits = dense.search(queries, topk=cfg.topk_dense)

    # choose hybrid strategy
    if cfg.hybrid_mode == "dense_only":
        hybrid_hits = dense_hits
    elif cfg.hybrid_mode == "bm25_only":
        hybrid_hits = bm25_hits
    else:  # "rrf"
        hybrid_hits = rrf_fusion_weighted(bm25_hits, dense_hits, k=cfg.rrf_k, topk=cfg.topk_fusion,
                                          w_bm25=cfg.rrf_w_bm25, w_dense=cfg.rrf_w_dense)

    # choose verified candidate source
    cand_map = {"hybrid": hybrid_hits, "dense": dense_hits, "sparse": bm25_hits}
    cand_source = cand_map.get(cfg.verified_source, hybrid_hits)

    if cfg.use_verified and reranker is not None:
        verified_hits = []
        for q, cands in zip(queries, cand_source):
            top = cands[:cfg.verified_topn]
            top_texts = [id2doc[doc_id] for doc_id, _ in top]
            if not top_texts:
                verified_hits.append([])
                continue
            order = reranker.rerank(q, top_texts, topk=min(cfg.topk_fusion, len(top_texts)))
            verified_hits.append([top[i] for i in order])
    else:
        verified_hits = hybrid_hits

    return {"sparse": bm25_hits, "dense": dense_hits, "hybrid": hybrid_hits, "hybrid_verified": verified_hits}


# ======================================================
# Evaluate runs (with bootstrap CI for nDCG@10)
# ======================================================

def evaluate_runs(runs: Dict[str, List[List[Tuple[int,float]]]],
                  qids: List[str], qrels_df: pd.DataFrame, cfg: Config) -> Dict[str, Dict[str, float]]:
    rels = {}
    for qid, grp in qrels_df.groupby("qid"):
        rels[qid] = {int(r.doc_id): float(r.rel) for r in grp.itertuples(index=False)}

    out = {}
    for name, run in runs.items():
        ndcgs, zeros, precs, recs = [], 0, [], []
        for qid, hits in zip(qids, run):
            gains = [rels.get(qid, {}).get(doc, 0.0) for doc,_ in hits]
            nd = ndcg_at_k(gains, cfg.eval_k)
            ndcgs.append(nd)
            if nd == 0.0: zeros += 1
            total_rel = len(rels.get(qid, {}))
            precs.append(precision_at_k(gains, cfg.eval_k))
            recs.append(recall_at_k(gains, total_rel, cfg.eval_k))
        nd_low, nd_high = bootstrap_ci_mean(ndcgs, n_resamples=1000, alpha=0.05, seed=SEED)
        out[name] = {
            "mean_ndcg@10": float(np.mean(ndcgs)),
            "mean_ndcg@10_ci_low": nd_low,
            "mean_ndcg@10_ci_high": nd_high,
            "precision@10": float(np.mean(precs)),
            "recall@10": float(np.mean(recs)),
            "zero_rate": float(zeros/len(qids))
        }
    return out

# ======================================================
# CV predictor with isotonic calibration (OOF) + bootstrap CIs for correlations
# ======================================================

def cv_fit_predictor(X: pd.DataFrame, y: np.ndarray, cfg: Config) -> Tuple[Any, IsotonicRegression, Dict[str,float]]:
    kf = KFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.seed)
    oof_raw = np.zeros_like(y, dtype=float)

    base_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=cfg.seed,
        verbose=-1,
    )

    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(**base_params)
        m.fit(X.iloc[tr], y[tr])
        oof_raw[va] = m.predict(X.iloc[va])

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_raw, y)

    # final model on full data
    model = lgb.LGBMRegressor(**base_params)
    model.fit(X, y)

    # metrics (OOF)
    oof_cal = iso.predict(oof_raw)
    def pearson(a,b):
        a = np.array(a); b = np.array(b)
        if a.std()==0 or b.std()==0: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    def spearman(a,b):
        a = np.array(a); b = np.array(b)
        ra = a.argsort().argsort()
        rb = b.argsort().argsort()
        if ra.std()==0 or rb.std()==0: return 0.0
        return float(np.corrcoef(ra, rb)[0,1])

    pear_raw = pearson(y, oof_raw)
    pear_cal = pearson(y, oof_cal)
    spear_raw = spearman(y, oof_raw)
    spear_cal = spearman(y, oof_cal)

    pr_low, pr_high = bootstrap_ci_corr(y, oof_raw, kind="pearson", n_resamples=1000, alpha=0.05, seed=SEED)
    pc_low, pc_high = bootstrap_ci_corr(y, oof_cal, kind="pearson", n_resamples=1000, alpha=0.05, seed=SEED)
    sr_low, sr_high = bootstrap_ci_corr(y, oof_raw, kind="spearman", n_resamples=1000, alpha=0.05, seed=SEED)
    sc_low, sc_high = bootstrap_ci_corr(y, oof_cal, kind="spearman", n_resamples=1000, alpha=0.05, seed=SEED)

    metrics = {
        "rmse_raw": float(np.sqrt(mean_squared_error(y, oof_raw))),
        "mae_raw": float(mean_absolute_error(y, oof_raw)),
        "rmse_cal": float(np.sqrt(mean_squared_error(y, oof_cal))),
        "mae_cal": float(mean_absolute_error(y, oof_cal)),
        "pearson_raw": pear_raw,
        "pearson_cal": pear_cal,
        "spearman_raw": spear_raw,
        "spearman_cal": spear_cal,
        "pearson_raw_ci_low": pr_low,
        "pearson_raw_ci_high": pr_high,
        "pearson_cal_ci_low": pc_low,
        "pearson_cal_ci_high": pc_high,
        "spearman_raw_ci_low": sr_low,
        "spearman_raw_ci_high": sr_high,
        "spearman_cal_ci_low": sc_low,
        "spearman_cal_ci_high": sc_high,
    }
    return model, iso, metrics
# ======================================================
# Safety-aware routing (single & dual)
# ======================================================

def route_one_single(pred_perf: float, high_risk_prob: float, cfg: Config) -> str:
    # Safety first
    if high_risk_prob >= cfg.high_risk_prob_thr and pred_perf < cfg.safety_override_thr:
        return "crisis_response" if cfg.enable_crisis_tier else "human_review"

    # Highest threshold first so it is not shadowed
    if pred_perf >= cfg.thr_hybrid:
        return "hybrid"
    if pred_perf >= cfg.thr_sparse:
        return "sparse"
    return "hybrid_verified"

def route_one_dual(pred_sparse: float, pred_hybrid: float, high_risk_prob: float, cfg: Config) -> str:
    # safety override first
    if high_risk_prob >= cfg.high_risk_prob_thr and max(pred_sparse, pred_hybrid) < cfg.safety_override_thr:
        return "crisis_response" if cfg.enable_crisis_tier else "human_review"
    # favor sparse if predicted >= hybrid + delta
    if pred_sparse >= (pred_hybrid + cfg.dual_delta) and pred_sparse >= cfg.thr_sparse:
        return "sparse"
    if pred_hybrid >= cfg.thr_hybrid:
        return "hybrid"
    return "hybrid_verified"

def simulate_routing(qids: List[str],
                     routes_for: List[str],
                     runs: Dict[str, List[List[Tuple[int,float]]]],
                     qrels_df: pd.DataFrame,
                     cfg: Config) -> Dict[str,float]:
    rels = {}
    for qid, grp in qrels_df.groupby("qid"):
        rels[qid] = {int(r.doc_id): float(r.rel) for r in grp.itertuples(index=False)}

    # Retrieval-tier costs only (for IR metrics)
    costs_ir = {
        "sparse": cfg.cost_sparse,
        "hybrid": cfg.cost_hybrid,
        "hybrid_verified": cfg.cost_verified,
    }

    total_cost_ir = 0.0          # cost over IR tiers only
    total_cost_overall = 0.0     # IR + crisis + (optionally) human_review
    ndcgs_auto: List[float] = []
    all_ndcgs: List[float] = []

    auto_count = 0
    crisis_count = 0
    human_count = 0

    for i, (qid, route) in enumerate(zip(qids, routes_for)):
        # Crisis tier: no IR, but we DO incur crisis LLM cost
        if route == "crisis_response":
            crisis_count += 1
            total_cost_overall += float(getattr(cfg, "cost_crisis", 0.0))
            all_ndcgs.append(0.0)
            continue

        # Human review: no IR; we treat cost as 0.0 here (could be extended later)
        if route == "human_review":
            human_count += 1
            all_ndcgs.append(0.0)
            continue

        # IR tiers: sparse / hybrid / hybrid_verified
        auto_count += 1
        tier_cost = float(costs_ir[route])
        total_cost_ir += tier_cost
        total_cost_overall += tier_cost

        if route == "sparse":
            chosen = runs["sparse"][i]
        elif route == "hybrid":
            chosen = runs["hybrid"][i]
        else:
            chosen = runs["hybrid_verified"][i]

        gains = [rels.get(qid, {}).get(doc, 0.0) for doc, _ in chosen]
        nd = ndcg_at_k(gains, cfg.eval_k)
        ndcgs_auto.append(nd)
        all_ndcgs.append(nd)

    # IR-only averages (backwards compatible)
    avg_cost_ir = total_cost_ir / max(1, auto_count)
    avg_ndcg_auto = float(np.mean(ndcgs_auto)) if ndcgs_auto else 0.0
    eff_ir = avg_ndcg_auto / avg_cost_ir if avg_cost_ir > 0 else 0.0

    # Overall (including crisis + human_review)
    N = max(1, len(routes_for))
    avg_ndcg_overall = float(np.mean(all_ndcgs)) if all_ndcgs else 0.0
    avg_cost_overall = total_cost_overall / N if total_cost_overall > 0 else 0.0
    eff_overall = avg_ndcg_overall / avg_cost_overall if avg_cost_overall > 0 else 0.0

    return {
        # IR-only (same semantics as before)
        "avg_cost": float(avg_cost_ir),
        "avg_ndcg@10_auto": avg_ndcg_auto,
        "avg_ndcg@10_overall": avg_ndcg_overall,
        "efficiency": float(eff_ir),

        # New: global accounting including crisis cost
        "avg_cost_overall": float(avg_cost_overall),
        "efficiency_overall": float(eff_overall),

        # Routing ratios
        "auto_ratio": float(auto_count / N),
        "crisis_ratio": float(crisis_count / N),
        "human_review_ratio": float(human_count / N),
    }


# ======================================================
# QPP Baselines: SCQ, Clarity, WIG, NQC (BM25-based)
# ======================================================

def corpus_stats_for_bm25(docs: List[str]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    N = len(docs)
    for d in docs:
        toks = set(simple_tokens(d))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    # idf-ish (BM25-like)
    idf = {t: math.log((N + 1) / (dfc + 0.5)) for t, dfc in df.items()}
    return idf

def clarity_score(query: str, docs: List[str]) -> float:
    toks = simple_tokens(query)
    if not toks: return 0.0
    qcounts: Dict[str, int] = {}
    for t in toks: qcounts[t] = qcounts.get(t, 0)+1
    qlm = {t: c/len(toks) for t,c in qcounts.items()}
    # collection LM (very simple)
    all_toks = list(itertools.chain.from_iterable(simple_tokens(d) for d in docs))
    if not all_toks: return 0.0
    ccounts: Dict[str, int] = {}
    for t in all_toks: ccounts[t] = ccounts.get(t, 0)+1
    Z = sum(ccounts.values())
    clm = {t: c/Z for t,c in ccounts.items()}
    kl = 0.0
    for t, p in qlm.items():
        q = clm.get(t, 1e-12)
        kl += p * math.log(max(p,1e-12)/q)
    return float(kl)

def nqc_score(scores: List[float], k: int) -> float:
    if not scores: return 0.0
    top = np.array(scores[:k], dtype=float)
    denom = max(1e-9, abs(top.mean()))
    return float(top.std() / denom)

def scq_score(query: str, idf: Dict[str,float]) -> float:
    toks = simple_tokens(query)
    if not toks: return 0.0
    s = sum(idf.get(t, 0.0) for t in toks)
    return float(s / max(1, len(toks)))

def compute_qpp_baselines(queries: List[str],
                          bm25_hits: List[List[Tuple[int,float]]],
                          bm25_index: BM25Index,
                          docs_text: List[str],
                          cfg: Config) -> Dict[str, List[float]]:
    idf = corpus_stats_for_bm25(docs_text)
    SCQ, CL, WIG, NQC = [], [], [], []
    for q, hits in zip(queries, bm25_hits):
        all_scores = bm25_index.bm25.get_scores(simple_tokens(q))  # corpus-wide scores for q
        top_scores = [s for _, s in hits]
        corp_avg = float(np.mean(all_scores)) if np.size(all_scores) else 0.0
        SCQ.append(scq_score(q, idf))
        CL.append(clarity_score(q, docs_text))
        WIG.append(float(np.mean(top_scores) - corp_avg))
        NQC.append(nqc_score(top_scores, cfg.eval_k))
    return {"SCQ": SCQ, "Clarity": CL, "WIG": WIG, "NQC": NQC}

# ======================================================
# Analyses helpers
# ======================================================

def adversarial_consistency(original_query: str,
                            router_fn,
                            route_args: Dict[str, Any]) -> Dict[str, Any]:
    perturbations = [
        original_query,
        original_query.replace(" I ", " I am ").replace(" I'm ", " I am "),
        original_query.replace(" feel ", " feel very "),
        original_query.lower().capitalize()
    ]
    routes = [router_fn(q, **route_args) for q in perturbations]
    return {
        "consistent": len(set(routes)) == 1,
        "perturbations": perturbations,
        "routes": routes
    }

def counterfactual_regret(qids: List[str],
                          runs: Dict[str, List[List[Tuple[int,float]]]],
                          qrels_df: pd.DataFrame,
                          chosen_routes: List[str],
                          cfg: Config) -> Dict[str,float]:
    rels: Dict[str, Dict[int, float]] = {}
    for qid, grp in qrels_df.groupby("qid"):
        rels[qid] = {int(r.doc_id): float(r.rel) for r in grp.itertuples(index=False)}

    regrets: List[float] = []
    for i, (qid, route) in enumerate(zip(qids, chosen_routes)):
        variants = {
            "sparse": runs["sparse"][i],
            "hybrid": runs["hybrid"][i],
            "hybrid_verified": runs["hybrid_verified"][i],
        }
        if route in ("human_review", "crisis_response"):
            nd_actual = 0.0
        else:
            gains_actual = [rels.get(qid, {}).get(doc, 0.0) for doc,_ in variants[route]]
            nd_actual = ndcg_at_k(gains_actual, cfg.eval_k)

        best = nd_actual
        for alt in ["sparse","hybrid","hybrid_verified"]:
            gains_alt = [rels.get(qid, {}).get(doc, 0.0) for doc,_ in variants[alt]]
            nd_alt = ndcg_at_k(gains_alt, cfg.eval_k)
            best = max(best, nd_alt)
        regrets.append(max(0.0, best - nd_actual))
    return {
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "max_regret": float(np.max(regrets)) if regrets else 0.0
    }

def compare_chunk_modes(df_raw: pd.DataFrame, cfg: Config, device: str) -> Dict[str, Dict[str,float]]:
    """
    Retrieval-only ablation across chunking modes.
    Returns: {mode: metrics-dict for the (verified if enabled else hybrid) run}
    """
    results: Dict[str, Dict[str,float]] = {}
    base_dense = "pritamdeka/S-PubMedBert-MS-MARCO"
    for mode in ["sentence", "semantic", "overlap"]:
        ctmp = Config(**{**vars(cfg), "chunk_mode": mode})
        docs_df, qrels_df = build_corpus(df_raw, ctmp)
        bm25 = BM25Index(docs_df["text"].tolist(), docs_df["doc_id"].tolist())
        dense_idx = DenseIndex(base_dense, device=device)
        dense_idx.build(docs_df["text"].tolist(), docs_df["doc_id"].tolist())
        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_len=ctmp.verified_max_len) if ctmp.use_verified else None

        q_meta = df_raw[["qid","qtext"]].drop_duplicates("qid").reset_index(drop=True)
        qids = q_meta["qid"].astype(str).tolist()
        queries = q_meta["qtext"].astype(str).tolist()

        runs = build_runs(queries, docs_df, bm25, dense_idx, reranker, ctmp)
        evals = evaluate_runs(runs, qids, qrels_df, ctmp)
        results[mode] = {k: v for k, v in evals["hybrid_verified" if ctmp.use_verified else "hybrid"].items()}
        # free memory
        del bm25, dense_idx, reranker
        gc.collect()
    return results

# ======================================================
# Predict function (train-time schema → inference-time parity)
# ======================================================

def make_predict_fn(model, iso, ref_columns: List[str],
                    bm25_index: BM25Index,
                    dense_index: DenseIndex,
                    emo_detector: EmotionDetector,
                    dense_model: SentenceTransformer,
                    cfg: Config):
    """
    Returns a function predict(query) -> (y_calibrated, high_risk_prob),
    computing the same feature schema used in training by running small (top-K) BM25/Dense searches.
    """
    def predict(query: str) -> Tuple[float, float]:
        # pre-retrieval
        qf = compute_query_features([query])
        ef = compute_embedding_feats([query], dense_model)
        qf = pd.concat([qf, ef], axis=1)
        emo = emo_detector.features_for_texts([query])
        qf = add_emotion_feats(qf, emo)

        # post-retrieval dispersion + complementarity
        pr = postret_features_for_query(query, bm25_index, dense_index, k=cfg.qpp_post_k)
        X_new = pd.concat([qf.reset_index(drop=True), pr.reset_index(drop=True)], axis=1)
        X_new = ensure_feature_columns(X_new, ref_columns)

        y_raw = model.predict(X_new.values)
        y_cal = iso.predict(y_raw)
        return float(y_cal[0]), float(emo["high_risk_prob"][0])
    return predict
# ======================================================
# Dense Model Comparison
# ======================================================

# ======================================================
# Dense Model Comparison
# ======================================================

def compare_dense_models(
    model_names: List[str],
    docs_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    queries: List[str],
    qids: List[str],
    cfg: Config,
    device: str,
    reranker: Optional[CrossEncoderReranker] = None,
) -> pd.DataFrame:
    """Compare multiple dense retrieval models."""
    print(f"\n[comparison] Testing {len(model_names)} dense models...")
    
    # Build BM25 once
    print("[comparison] Building shared BM25 index...")
    bm25 = BM25Index(docs_df["text"].tolist(), docs_df["doc_id"].tolist())
    
    results = []
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(model_names)}] Testing: {model_name}")
        print(f"{'='*80}\n")
        
        try:
            # Build dense index
            dense_idx = DenseIndex(model_name, device=device, batch_size=64)
            dense_idx.build(docs_df["text"].tolist(), docs_df["doc_id"].tolist())
            
            # Run retrieval
            runs = build_runs(queries, docs_df, bm25, dense_idx, reranker, cfg)
            
            # Evaluate
            evals = evaluate_runs(runs, qids, qrels_df, cfg)
            dense_metrics = evals["dense"]
            
            # Categorize
            name_lower = model_name.lower()
            if any(x in name_lower for x in ['bio', 'pubmed', 'clinical', 'sci']):
                category = 'Biomedical'
            elif any(x in name_lower for x in ['mini', 'distil']):
                category = 'Efficient'
            else:
                category = 'General'
            
            # Estimate size
            if 'large' in name_lower:
                params = '335M'
            elif 'minilm-l6' in name_lower:
                params = '22M'
            elif 'distil' in name_lower:
                params = '82M'
            else:
                params = '110M'
            
            results.append({
                'model': model_name,
                'category': category,
                'parameters': params,
                'ndcg@10': dense_metrics['mean_ndcg@10'],
                'ndcg@10_ci_low': dense_metrics['mean_ndcg@10_ci_low'],
                'ndcg@10_ci_high': dense_metrics['mean_ndcg@10_ci_high'],
                'precision@10': dense_metrics['precision@10'],
                'recall@10': dense_metrics['recall@10'],
                'zero_rate': dense_metrics['zero_rate'],
                'status': 'success',
            })
            
            print(f"✓ Success: nDCG@10 = {dense_metrics['mean_ndcg@10']:.3f}")
            
            del dense_idx
            gc.collect()
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'model': model_name,
                'category': 'Error',
                'parameters': 'N/A',
                'ndcg@10': 0.0,
                'ndcg@10_ci_low': 0.0,
                'ndcg@10_ci_high': 0.0,
                'precision@10': 0.0,
                'recall@10': 0.0,
                'zero_rate': 1.0,
                'status': f'error',
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('ndcg@10', ascending=False)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")
    print(df_results[['model', 'category', 'ndcg@10', 'status']].to_string(index=False))
    
    return df_results

#File 2: Integration into dar_core.py
#Add this function at the end of dar_core.py:
def compute_modern_qpp_baselines(
    queries: List[str],
    sparse_runs: List[List[Tuple[int, float]]],
    dense_model: SentenceTransformer,
    docs_texts: List[str],
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Wrapper to compute modern QPP baselines.
    Delegates to dar_modern_baselines.py
    """
    try:
        from dar_modern_baselines import compute_all_qpp_baselines
        return compute_all_qpp_baselines(
            queries=queries,
            sparse_runs=sparse_runs,
            dense_model=dense_model,
            docs_texts=docs_texts,
            device=device
        )
    except ImportError:
        print("[warning] dar_modern_baselines.py not found, skipping modern baselines")
        return {}
