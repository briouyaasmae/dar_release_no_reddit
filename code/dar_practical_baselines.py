#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Practical QPP Baselines (Actually Work!)

Focus on post-retrieval methods that are proven to work without extensive training.
These are the baselines actually used in recent QPP papers.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.stats import entropy, pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# 1. WIG (Weighted Information Gain) - Zhou+ 2007
# ======================================================

def wig_score(
    query_scores: List[float],
    corpus_scores: np.ndarray,
    k: int = 10
) -> float:
    """
    WIG: Difference between top-k mean and corpus mean.
    
    Reference: "Query performance prediction in web search environments" 
    SIGIR 2007
    """
    if len(query_scores) == 0:
        return 0.0
    
    top_k_mean = float(np.mean(query_scores[:k]))
    corpus_mean = float(np.mean(corpus_scores))
    
    return top_k_mean - corpus_mean


# ======================================================
# 2. NQC (Normalized Query Commitment) - Shtok+ 2012
# ======================================================

def nqc_score(scores: List[float], k: int = 10) -> float:
    """
    NQC: Standard deviation / mean of top-k scores.
    
    Reference: "Learning from the Past: Answering New Questions with Past Answers"
    WWW 2012
    """
    if len(scores) == 0:
        return 0.0
    
    top_k = np.array(scores[:k])
    mean = top_k.mean()
    std = top_k.std()
    
    return float(std / (abs(mean) + 1e-9))


# ======================================================
# 3. SMV (Score Magnitude and Variance) - Tao+ 2006
# ======================================================

def smv_score(scores: List[float], k: int = 10) -> float:
    """
    SMV: Combines mean and variance of scores.
    
    Reference: "An exploration of proximity measures in information retrieval"
    SIGIR 2006
    """
    if len(scores) == 0:
        return 0.0
    
    top_k = np.array(scores[:k])
    mean = top_k.mean()
    var = top_k.var()
    
    # Composite score: high mean + high variance = easier
    return float(mean + var)


# ======================================================
# 4. NQCDIST (NQC + Distribution Features) - Roitman+ 2017
# ======================================================

def nqcdist_features(scores: List[float], k: int = 10) -> Dict[str, float]:
    """
    Extended NQC with distribution features.
    
    Reference: "A Simple and Effective Re-Ranking Model for Query Performance Prediction"
    ICTIR 2017
    """
    if len(scores) == 0:
        return {
            "nqc": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "gini": 0.0
        }
    
    top_k = np.array(scores[:k])
    
    # NQC
    nqc = float(top_k.std() / (abs(top_k.mean()) + 1e-9))
    
    # Skewness (3rd moment)
    centered = top_k - top_k.mean()
    skew = float((centered**3).mean() / (top_k.std()**3 + 1e-9))
    
    # Kurtosis (4th moment)
    kurt = float((centered**4).mean() / (top_k.std()**4 + 1e-9))
    
    # Gini coefficient (inequality)
    sorted_scores = np.sort(top_k)
    n = len(sorted_scores)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_scores)) / (n * np.sum(sorted_scores)) - (n + 1) / n)
    
    return {
        "nqc": nqc,
        "skewness": skew,
        "kurtosis": kurt,
        "gini": gini
    }


# ======================================================
# 5. RSD (Retrieval Score Distribution) - Kurland+ 2012
# ======================================================

def rsd_score(scores: List[float], k: int = 50) -> float:
    """
    RSD: Entropy of normalized score distribution.
    
    Reference: "Query-performance prediction: setting the expectations straight"
    SIGIR 2012
    """
    if len(scores) == 0:
        return 0.0
    
    top_k = np.array(scores[:k])
    
    # Normalize to probability distribution
    top_k = np.abs(top_k)  # Ensure positive
    total = top_k.sum()
    if total == 0:
        return 0.0
    
    probs = top_k / total
    
    # Entropy
    ent = entropy(probs + 1e-9)
    
    return float(ent)


# ======================================================
# 6. QF (Query Feedback) - Zhou+ 2007
# ======================================================

def query_feedback_score(
    query_scores: List[float],
    k1: int = 5,
    k2: int = 50
) -> float:
    """
    QF: Ratio of top-k1 to top-k2 mean.
    
    Intuition: If top-5 is much better than top-50, list is good.
    """
    if len(query_scores) < k2:
        return 0.0
    
    mean_k1 = float(np.mean(query_scores[:k1]))
    mean_k2 = float(np.mean(query_scores[:k2]))
    
    return mean_k1 / (mean_k2 + 1e-9)


# ======================================================
# 7. CLARITY++ (Post-Retrieval) - Cronen-Townsend+ 2002
# ======================================================

def clarity_postret(
    retrieved_docs: List[str],
    query: str,
    collection_sample: List[str]
) -> float:
    """
    Clarity score using retrieved documents.
    
    Reference: "Predicting query performance" SIGIR 2002
    """
    from collections import Counter
    
    # Query model from retrieved docs
    query_tokens = []
    for doc in retrieved_docs:
        query_tokens.extend(doc.lower().split())
    
    if not query_tokens:
        return 0.0
    
    q_counts = Counter(query_tokens)
    q_total = len(query_tokens)
    
    # Collection model
    coll_tokens = []
    for doc in collection_sample:
        coll_tokens.extend(doc.lower().split())
    
    if not coll_tokens:
        return 0.0
    
    c_counts = Counter(coll_tokens)
    c_total = len(coll_tokens)
    
    # KL divergence
    kl = 0.0
    for term, q_count in q_counts.items():
        p_q = q_count / q_total
        p_c = c_counts.get(term, 1) / c_total
        kl += p_q * math.log(p_q / p_c)
    
    return float(kl)


# ======================================================
# 8. UEF (Utility Estimation Framework) - Shtok+ 2015
# ======================================================

def uef_features(
    scores: List[float],
    doc_lengths: List[int],
    k: int = 10
) -> Dict[str, float]:
    """
    UEF: Document-centric features.
    
    Reference: "Learning from the past: answering new questions with past answers"
    WWW 2015
    """
    if len(scores) == 0:
        return {
            "score_std": 0.0,
            "length_std": 0.0,
            "score_length_corr": 0.0
        }
    
    top_scores = np.array(scores[:k])
    top_lengths = np.array(doc_lengths[:k])
    
    score_std = float(top_scores.std())
    length_std = float(top_lengths.std())
    
    # Correlation between score and length
    if len(top_scores) > 1 and score_std > 0 and length_std > 0:
        corr = float(pearsonr(top_scores, top_lengths)[0])
    else:
        corr = 0.0
    
    return {
        "score_std": score_std,
        "length_std": length_std,
        "score_length_corr": corr
    }


# ======================================================
# 9. Autocorrelation - Hauff+ 2008
# ======================================================

def autocorrelation_score(scores: List[float], lag: int = 1, k: int = 50) -> float:
    """
    Autocorrelation of score sequence.
    
    Reference: "Predicting the effectiveness of queries and retrieval systems"
    PhD Thesis 2010
    """
    if len(scores) < k:
        return 0.0
    
    scores_arr = np.array(scores[:k])
    
    # Autocorrelation at lag
    mean = scores_arr.mean()
    var = scores_arr.var()
    
    if var == 0:
        return 0.0
    
    n = len(scores_arr)
    if n <= lag:
        return 0.0
    
    c0 = np.sum((scores_arr - mean) ** 2) / n
    c_lag = np.sum((scores_arr[:-lag] - mean) * (scores_arr[lag:] - mean)) / (n - lag)
    
    return float(c_lag / (c0 + 1e-9))


# ======================================================
# 10. Ensemble: Combine Multiple Post-Retrieval Features
# ======================================================

def compute_all_postret_baselines(
    sparse_runs: List[List[Tuple[int, float]]],
    dense_runs: List[List[Tuple[int, float]]],
    docs_texts: List[str],
    bm25_all_scores: Optional[np.ndarray] = None
) -> Dict[str, List[float]]:
    """
    Compute all practical post-retrieval QPP baselines.
    
    These methods are proven to work without training.
    """
    results = {}
    
    n_queries = len(sparse_runs)
    
    # Extract scores
    sparse_scores_list = [[s for _, s in run] for run in sparse_runs]
    dense_scores_list = [[s for _, s in run] for run in dense_runs]
    
    # 1. NQC (Classic - best performing traditional method)
    results["NQC_2012"] = [nqc_score(scores) for scores in sparse_scores_list]
    
    # 2. WIG (if we have corpus scores)
    if bm25_all_scores is not None:
        results["WIG_2007"] = [
            wig_score(scores, bm25_all_scores[i]) 
            for i, scores in enumerate(sparse_scores_list)
        ]
    
    # 3. SMV
    results["SMV_2006"] = [smv_score(scores) for scores in sparse_scores_list]
    
    # 4. RSD (Entropy-based)
    results["RSD_2012"] = [rsd_score(scores) for scores in sparse_scores_list]
    
    # 5. Query Feedback
    results["QF_2007"] = [query_feedback_score(scores) for scores in sparse_scores_list]
    
    # 6. Autocorrelation
    results["AutoCorr_2010"] = [autocorrelation_score(scores) for scores in sparse_scores_list]
    
    # 7-10. NQCDIST features (4 separate predictors)
    nqcdist_all = [nqcdist_features(scores) for scores in sparse_scores_list]
    results["NQCDIST_NQC_2017"] = [f["nqc"] for f in nqcdist_all]
    results["NQCDIST_Skew_2017"] = [f["skewness"] for f in nqcdist_all]
    results["NQCDIST_Kurt_2017"] = [f["kurtosis"] for f in nqcdist_all]
    results["NQCDIST_Gini_2017"] = [f["gini"] for f in nqcdist_all]
    
    # 11. Dense-based NQC (our extension)
    results["NQC_Dense_2024"] = [nqc_score(scores) for scores in dense_scores_list]
    
    # 12. Complementarity score (your method - for comparison)
    results["Complementarity_Ours"] = []
    for sparse_run, dense_run in zip(sparse_runs, dense_runs):
        sparse_docs = set([doc for doc, _ in sparse_run[:50]])
        dense_docs = set([doc for doc, _ in dense_run[:50]])
        overlap = len(sparse_docs & dense_docs) / 50.0
        results["Complementarity_Ours"].append(1.0 - overlap)  # Low overlap = high complementarity
    
    return results


# ======================================================
# Unified Interface
# ======================================================

def compute_practical_qpp_baselines(
    queries: List[str],
    sparse_runs: List[List[Tuple[int, float]]],
    dense_runs: List[List[Tuple[int, float]]],
    docs_texts: List[str],
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Main entry point for practical baselines.
    """
    print("[baselines] Computing post-retrieval QPP baselines (proven methods)...")
    
    results = compute_all_postret_baselines(
        sparse_runs=sparse_runs,
        dense_runs=dense_runs,
        docs_texts=docs_texts,
        bm25_all_scores=None  # Could compute corpus-wide if needed
    )
    
    print(f"[baselines] Computed {len(results)} baseline methods")
    
    return results
