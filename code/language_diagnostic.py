#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic cross-lingual retrieval stress test for the DAR revision.

Purpose
-------
This script addresses reviewer concerns about generalization scope by testing
non-English mental-health queries against the English CounselChat-style corpus.
It compares lexical BM25, an English dense encoder, and a multilingual dense
encoder using a pseudo-relevance alignment proxy.

Important interpretation
------------------------
The reported metric is pseudo-nDCG@10, not human-annotated nDCG@10. Pseudo-
relevant sets are derived from matched English reference queries using an
English dense encoder. When multiple non-English queries share the same topic,
they are evaluated against the same topic-level pseudo-relevant set. Therefore,
results should be reported as a diagnostic cross-lingual alignment stress test,
not as a definitive multilingual benchmark.

Suggested installation for Kaggle
---------------------------------
!pip -q install rank_bm25 sentence-transformers faiss-cpu pandas pyarrow scikit-learn langdetect
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu") from exc

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install rank_bm25: pip install rank_bm25") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from exc

try:
    from langdetect import detect
except Exception:  # pragma: no cover
    detect = None


# ============================================================
# Configuration
# ============================================================

SEED = 42
np.random.seed(SEED)

TOPK = 10
BATCH_SIZE = 64
OUTDIR = Path("/kaggle/working/multilingual_retrieval_study")
OUTDIR.mkdir(parents=True, exist_ok=True)

ENGLISH_DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
MULTILINGUAL_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MULTILINGUAL_QUERIES = [
    {
        "lang": "fr",
        "lang_name": "French",
        "query": "Je me sens très anxieuse depuis plusieurs semaines et je n'arrive pas à dormir. Que puis-je faire ?",
        "topic": "anxiety / sleep",
    },
    {
        "lang": "fr",
        "lang_name": "French",
        "query": "Comment gérer la dépression sans médicaments ?",
        "topic": "depression management",
    },
    {
        "lang": "fr",
        "lang_name": "French",
        "query": "J'ai des pensées négatives constantes et je me sens inutile.",
        "topic": "negative thoughts / self-worth",
    },
    {
        "lang": "es",
        "lang_name": "Spanish",
        "query": "Me siento muy solo y no sé cómo hablar con mi familia sobre mi depresión.",
        "topic": "loneliness / family communication",
    },
    {
        "lang": "es",
        "lang_name": "Spanish",
        "query": "Tengo ataques de pánico frecuentes. ¿Qué puedo hacer para calmarme?",
        "topic": "panic attacks",
    },
    {
        "lang": "es",
        "lang_name": "Spanish",
        "query": "Mi hijo adolescente tiene problemas de conducta. ¿Cómo puedo ayudarlo?",
        "topic": "adolescent behavior",
    },
    {
        "lang": "de",
        "lang_name": "German",
        "query": "Ich fühle mich seit Wochen deprimiert und weiß nicht, was ich tun soll.",
        "topic": "depression",
    },
    {
        "lang": "de",
        "lang_name": "German",
        "query": "Wie kann ich mit Angststörungen ohne Medikamente umgehen?",
        "topic": "anxiety management",
    },
    {
        "lang": "ar",
        "lang_name": "Arabic",
        "query": "أشعر بالقلق الشديد ولا أستطيع النوم. كيف يمكنني التعامل مع هذا؟",
        "topic": "anxiety / sleep",
    },
    {
        "lang": "ar",
        "lang_name": "Arabic",
        "query": "كيف أتعامل مع الاكتئاب والأفكار السلبية؟",
        "topic": "depression / negative thoughts",
    },
    {
        "lang": "pt",
        "lang_name": "Portuguese",
        "query": "Estou sofrendo de ansiedade severa e preciso de ajuda para controlar meus pensamentos.",
        "topic": "anxiety / thought control",
    },
    {
        "lang": "pt",
        "lang_name": "Portuguese",
        "query": "Como posso lidar com a depressão pós-parto?",
        "topic": "postpartum depression",
    },
]

ENGLISH_REFERENCE_QUERIES = [
    {"topic": "anxiety / sleep", "query": "I have been feeling very anxious for weeks and cannot sleep. What can I do?"},
    {"topic": "depression management", "query": "How can I manage depression without medication?"},
    {"topic": "negative thoughts / self-worth", "query": "I have constant negative thoughts and feel worthless."},
    {"topic": "loneliness / family communication", "query": "I feel very lonely and do not know how to talk to my family about my depression."},
    {"topic": "panic attacks", "query": "I have frequent panic attacks. What can I do to calm down?"},
    {"topic": "adolescent behavior", "query": "My teenage child has behavioral problems. How can I help them?"},
    {"topic": "depression", "query": "I have been feeling depressed for weeks and do not know what to do."},
    {"topic": "anxiety management", "query": "How can I cope with anxiety disorders without medication?"},
    {"topic": "anxiety / sleep", "query": "I feel severe anxiety and cannot sleep. How can I deal with this?"},
    {"topic": "depression / negative thoughts", "query": "How do I deal with depression and negative thoughts?"},
    {"topic": "anxiety / thought control", "query": "I am suffering from severe anxiety and need help controlling my thoughts."},
    {"topic": "postpartum depression", "query": "How can I deal with postpartum depression?"},
]


# ============================================================
# Corpus loading
# ============================================================

def first_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def build_text_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a corpus dataframe to columns doc_id and text."""
    df = df.copy()

    if "text" not in df.columns:
        text_candidates = [
            "answer",
            "answers",
            "response",
            "document",
            "doc",
            "content",
            "body",
            "questionText",
        ]
        available = [c for c in text_candidates if c in df.columns]

        if available:
            df = df.rename(columns={available[0]: "text"})
        else:
            # CounselChat CSV variants often contain question + answer fields.
            string_cols = [c for c in df.columns if df[c].dtype == "object"]
            if not string_cols:
                raise ValueError(f"Cannot infer text column. Columns: {df.columns.tolist()}")
            df["text"] = df[string_cols].fillna("").agg(" ".join, axis=1)

    if "doc_id" not in df.columns:
        for cand in ["id", "_id", "docid", "document_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "doc_id"})
                break

    if "doc_id" not in df.columns:
        df["doc_id"] = [f"D{i}" for i in range(len(df))]

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["doc_id"] = df["doc_id"].astype(str)
    df = df.reset_index(drop=True)
    return df[["doc_id", "text"]]


def load_corpus() -> tuple[pd.DataFrame, Path]:
    candidate_paths = [
        Path("/kaggle/working/paper_experiments/runs/exp1_dense_only/docs_corpus.parquet"),
        Path("/kaggle/working/paper_experiments/runs/exp2_rrf_equal/docs_corpus.parquet"),
        Path("/kaggle/working/paper_experiments/runs/exp4_sentence_chunks_mpnet/docs_corpus.parquet"),
        Path("/kaggle/working/paper_experiments/runs/exp4_sentence_chunks/docs_corpus.parquet"),
        Path("/kaggle/working/docs_corpus.parquet"),
        Path("/kaggle/working/combined-data.csv"),
        Path("/kaggle/working/counselchat.csv"),
        Path("combined-data.csv"),
        Path("counselchat.csv"),
    ]

    corpus_path = first_existing(candidate_paths)
    if corpus_path is None:
        raise FileNotFoundError(
            "No corpus found. Expected one of docs_corpus.parquet, combined-data.csv, or counselchat.csv."
        )

    if corpus_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(corpus_path)
    elif corpus_path.suffix.lower() == ".csv":
        raw = pd.read_csv(corpus_path)
    else:
        raise ValueError(f"Unsupported corpus format: {corpus_path}")

    return build_text_from_dataframe(raw), corpus_path


docs_df, corpus_path = load_corpus()
docs = docs_df["text"].tolist()
position_to_doc_id = docs_df["doc_id"].tolist()
doc_id_to_position = {doc_id: i for i, doc_id in enumerate(position_to_doc_id)}

print(f"[corpus] {len(docs_df)} documents from {corpus_path}")


# ============================================================
# Utilities
# ============================================================

def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t.strip()]


def show_text(x: str, n: int = 220) -> str:
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x[:n] + ("..." if len(x) > n else "")


def ndcg_at_k_binary(retrieved: List[Tuple[str, float]], relevant: Set[str], k: int = TOPK) -> float:
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id, _ in retrieved[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def jaccard(a: Set[str], b: Set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def best_method_label(row: pd.Series) -> str:
    vals = {
        "BM25": row["BM25_pseudo_nDCG@10"],
        "English dense": row["English_dense_pseudo_nDCG@10"],
        "Multilingual dense": row["Multilingual_dense_pseudo_nDCG@10"],
    }
    return max(vals, key=vals.get)


def latex_num(value: float, bold: bool = False) -> str:
    s = f"{float(value):.3f}"
    return f"\\textbf{{{s}}}" if bold else s


# ============================================================
# Indexing and retrieval
# ============================================================

print("[index] building BM25 ...")
tokenized_docs = [tokenize(t) for t in docs]
bm25 = BM25Okapi(tokenized_docs)

_dense_cache: Dict[str, Tuple[SentenceTransformer, faiss.IndexFlatIP]] = {}


def get_dense_index(model_name: str) -> Tuple[SentenceTransformer, faiss.IndexFlatIP]:
    if model_name in _dense_cache:
        return _dense_cache[model_name]

    print(f"[index] encoding corpus with {model_name} ...")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _dense_cache[model_name] = (model, index)
    return model, index


def retrieve_bm25(query: str, topk: int = TOPK) -> List[Tuple[str, float]]:
    scores = bm25.get_scores(tokenize(query))
    top_positions = np.argsort(-scores)[:topk]
    return [(position_to_doc_id[int(i)], float(scores[int(i)])) for i in top_positions]


def retrieve_dense(model_name: str, query: str, topk: int = TOPK) -> List[Tuple[str, float]]:
    model, index = get_dense_index(model_name)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, topk)
    out = []
    for rank, pos in enumerate(ids[0]):
        if int(pos) < 0:
            continue
        out.append((position_to_doc_id[int(pos)], float(scores[0][rank])))
    return out


def build_pseudo_qrels(topic: str, topk: int = TOPK) -> Set[str]:
    refs = [q["query"] for q in ENGLISH_REFERENCE_QUERIES if q["topic"] == topic]
    if not refs:
        return set()

    relevant: Set[str] = set()
    for ref_query in refs:
        relevant.update(doc_id for doc_id, _ in retrieve_dense(ENGLISH_DENSE_MODEL, ref_query, topk))
    return relevant


# ============================================================
# Main evaluation
# ============================================================

rows = []
detail_rows = []

for qi, qdict in enumerate(MULTILINGUAL_QUERIES, start=1):
    lang = qdict["lang"]
    lang_name = qdict["lang_name"]
    query = qdict["query"]
    topic = qdict["topic"]

    if detect is not None:
        try:
            detected = detect(query)
        except Exception:
            detected = "unknown"
    else:
        detected = "langdetect_not_installed"

    pseudo_rel = build_pseudo_qrels(topic, topk=TOPK)

    bm25_res = retrieve_bm25(query)
    eng_res = retrieve_dense(ENGLISH_DENSE_MODEL, query)
    multi_res = retrieve_dense(MULTILINGUAL_DENSE_MODEL, query)

    bm25_ids = {doc_id for doc_id, _ in bm25_res}
    eng_ids = {doc_id for doc_id, _ in eng_res}
    multi_ids = {doc_id for doc_id, _ in multi_res}

    bm25_score = ndcg_at_k_binary(bm25_res, pseudo_rel)
    eng_score = ndcg_at_k_binary(eng_res, pseudo_rel)
    multi_score = ndcg_at_k_binary(multi_res, pseudo_rel)

    row = {
        "lang": lang,
        "lang_name": lang_name,
        "topic": topic,
        "detected_lang": detected,
        "pseudo_relevant_docs": len(pseudo_rel),
        "BM25_pseudo_nDCG@10": round(bm25_score, 4),
        "English_dense_pseudo_nDCG@10": round(eng_score, 4),
        "Multilingual_dense_pseudo_nDCG@10": round(multi_score, 4),
        "English_vs_Multilingual_Jaccard@10": round(jaccard(eng_ids, multi_ids), 4),
        "BM25_vs_English_Jaccard@10": round(jaccard(bm25_ids, eng_ids), 4),
        "multilingual_gain_over_english": round(multi_score - eng_score, 4),
        "multilingual_gain_over_bm25": round(multi_score - bm25_score, 4),
    }
    row["best_method_by_proxy"] = best_method_label(pd.Series(row))
    rows.append(row)

    for method_name, res in [
        ("BM25", bm25_res),
        ("English dense", eng_res),
        ("Multilingual dense", multi_res),
    ]:
        for rank, (doc_id, score) in enumerate(res[:3], start=1):
            pos = doc_id_to_position.get(doc_id)
            text = docs_df.loc[pos, "text"] if pos is not None else ""
            detail_rows.append(
                {
                    "lang": lang,
                    "lang_name": lang_name,
                    "topic": topic,
                    "query": query,
                    "method": method_name,
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "in_pseudo_relevant": doc_id in pseudo_rel,
                    "text_snippet": show_text(text),
                }
            )

    print(
        f"[{qi:02d}/{len(MULTILINGUAL_QUERIES)}] {lang_name:10s} | {topic:35s} | "
        f"BM25={bm25_score:.3f}  English={eng_score:.3f}  Multi={multi_score:.3f} "
        f"(Multi-English={multi_score - eng_score:+.3f})"
    )

results_df = pd.DataFrame(rows)
detail_df = pd.DataFrame(detail_rows)


# ============================================================
# Aggregate summaries
# ============================================================

agg = (
    results_df.groupby("lang_name")
    .agg(
        n_queries=("topic", "count"),
        BM25_mean_pseudo_nDCG=("BM25_pseudo_nDCG@10", "mean"),
        English_dense_mean_pseudo_nDCG=("English_dense_pseudo_nDCG@10", "mean"),
        Multilingual_dense_mean_pseudo_nDCG=("Multilingual_dense_pseudo_nDCG@10", "mean"),
        multi_gain_over_english=("multilingual_gain_over_english", "mean"),
        multi_gain_over_bm25=("multilingual_gain_over_bm25", "mean"),
    )
    .round(4)
    .reset_index()
)

best_counts = results_df["best_method_by_proxy"].value_counts().to_dict()

overall = {
    "metric": "pseudo_nDCG@10",
    "interpretation": "Diagnostic cross-lingual alignment proxy, not human-annotated multilingual relevance. Shared topics use common topic-level pseudo-qrels derived from matched English reference queries.",
    "n_queries": int(len(results_df)),
    "topk": int(TOPK),
    "corpus_path": str(corpus_path),
    "n_documents": int(len(docs_df)),
    "english_reference_model_for_pseudo_qrels": ENGLISH_DENSE_MODEL,
    "english_dense_model": ENGLISH_DENSE_MODEL,
    "multilingual_dense_model": MULTILINGUAL_DENSE_MODEL,
    "BM25_mean_pseudo_nDCG@10": float(results_df["BM25_pseudo_nDCG@10"].mean()),
    "English_dense_mean_pseudo_nDCG@10": float(results_df["English_dense_pseudo_nDCG@10"].mean()),
    "Multilingual_dense_mean_pseudo_nDCG@10": float(results_df["Multilingual_dense_pseudo_nDCG@10"].mean()),
    "multilingual_gain_over_english_mean": float(results_df["multilingual_gain_over_english"].mean()),
    "multilingual_gain_over_bm25_mean": float(results_df["multilingual_gain_over_bm25"].mean()),
    "pct_queries_multilingual_beats_english": float((results_df["multilingual_gain_over_english"] > 0).mean() * 100),
    "pct_queries_multilingual_beats_bm25": float((results_df["multilingual_gain_over_bm25"] > 0).mean() * 100),
    "best_method_counts_by_proxy": best_counts,
}

print("\n" + "=" * 90)
print("AGGREGATE RESULTS BY LANGUAGE")
print("=" * 90)
print(agg.to_string(index=False))

print("\n" + "=" * 90)
print("OVERALL DIAGNOSTIC SUMMARY")
print("=" * 90)
print(json.dumps(overall, indent=2, ensure_ascii=False))


# ============================================================
# LaTeX table with best method bolded per row
# ============================================================

latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Diagnostic cross-lingual retrieval stress test on an English CounselChat-style corpus.",
    r"Pseudo-nDCG@10 is computed against topic-level pseudo-relevant sets obtained from",
    r"matched English reference queries using the English dense encoder. Queries sharing",
    r"the same topic use the same reference set. The metric should therefore be interpreted",
    r"as a cross-lingual alignment proxy rather than as human-annotated multilingual relevance.}",
    r"\label{tab:multilingual-diagnostic}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"\textbf{Language} & \textbf{Topic} & \textbf{BM25} & \textbf{English dense} & \textbf{Multilingual dense} \\",
    r"\midrule",
]

for _, r in results_df.iterrows():
    best = r["best_method_by_proxy"]
    latex_lines.append(
        f"{r['lang_name']} & {r['topic']} & "
        f"{latex_num(r['BM25_pseudo_nDCG@10'], best == 'BM25')} & "
        f"{latex_num(r['English_dense_pseudo_nDCG@10'], best == 'English dense')} & "
        f"{latex_num(r['Multilingual_dense_pseudo_nDCG@10'], best == 'Multilingual dense')} \\\\"
    )

mean_bm25 = results_df["BM25_pseudo_nDCG@10"].mean()
mean_eng = results_df["English_dense_pseudo_nDCG@10"].mean()
mean_multi = results_df["Multilingual_dense_pseudo_nDCG@10"].mean()
means = {"BM25": mean_bm25, "English dense": mean_eng, "Multilingual dense": mean_multi}
mean_best = max(means, key=means.get)

latex_lines.extend(
    [
        r"\midrule",
        f"\\textit{{Mean}} & --- & "
        f"{latex_num(mean_bm25, mean_best == 'BM25')} & "
        f"{latex_num(mean_eng, mean_best == 'English dense')} & "
        f"{latex_num(mean_multi, mean_best == 'Multilingual dense')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
)

latex_str = "\n".join(latex_lines)


# ============================================================
# Suggested manuscript text and README
# ============================================================

suggested_text = """Suggested manuscript wording
============================

As an additional diagnostic stress test, we evaluated non-English mental-health queries against the English CounselChat-style corpus using pseudo-relevance sets derived from matched English reference queries. The results should be interpreted as a cross-lingual alignment proxy rather than as human-annotated multilingual relevance. The stress test indicates that lexical BM25 is brittle under language mismatch, while dense retrieval degrades more gracefully and a multilingual encoder can improve semantic alignment for several non-English topics. We therefore treat multilingual deployment as outside the validated scope of the present system and identify multilingual routing, language-aware safety detection, and human-annotated multilingual evaluation as important future work.
"""

readme_text = f"""# Multilingual Retrieval Diagnostic Study

This folder contains a diagnostic cross-lingual retrieval stress test for the DAR revision.

## Purpose

The study addresses reviewer concerns about generalization scope by testing non-English mental-health queries against an English CounselChat-style corpus.

## Important limitation

The reported metric is **pseudo-nDCG@10**, not human-annotated nDCG@10. Pseudo-relevant documents are obtained from matched English reference queries using `{ENGLISH_DENSE_MODEL}`. Queries that share the same topic use a common topic-level pseudo-relevant set. Results should therefore be interpreted as a cross-lingual alignment proxy, not as definitive multilingual retrieval effectiveness.

## Compared methods

- BM25 lexical retrieval
- English dense encoder: `{ENGLISH_DENSE_MODEL}`
- Multilingual dense encoder: `{MULTILINGUAL_DENSE_MODEL}`

## Outputs

- `multilingual_pseudo_ndcg_summary.csv`: per-query diagnostic scores
- `multilingual_top3_results_detail.csv`: top-3 retrieved examples per method
- `multilingual_by_language_agg.csv`: language-level aggregation
- `multilingual_overall_summary.json`: overall diagnostic summary
- `multilingual_table.tex`: LaTeX table with best method bolded per row
- `suggested_manuscript_text.txt`: cautious wording for the paper
- `multilingual_retrieval_diagnostic_outputs.zip`: all outputs bundled

## Generated at

{datetime.now(timezone.utc).isoformat()}
"""


# ============================================================
# Save outputs
# ============================================================

results_df.to_csv(OUTDIR / "multilingual_pseudo_ndcg_summary.csv", index=False)
detail_df.to_csv(OUTDIR / "multilingual_top3_results_detail.csv", index=False)
agg.to_csv(OUTDIR / "multilingual_by_language_agg.csv", index=False)

with open(OUTDIR / "multilingual_overall_summary.json", "w", encoding="utf-8") as f:
    json.dump(overall, f, indent=2, ensure_ascii=False)

with open(OUTDIR / "multilingual_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)

with open(OUTDIR / "suggested_manuscript_text.txt", "w", encoding="utf-8") as f:
    f.write(suggested_text)

with open(OUTDIR / "README_multilingual_study.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

run_config = {
    "seed": SEED,
    "topk": TOPK,
    "batch_size": BATCH_SIZE,
    "corpus_path": str(corpus_path),
    "n_documents": int(len(docs_df)),
    "english_dense_model": ENGLISH_DENSE_MODEL,
    "multilingual_dense_model": MULTILINGUAL_DENSE_MODEL,
    "n_multilingual_queries": len(MULTILINGUAL_QUERIES),
    "queries": MULTILINGUAL_QUERIES,
    "english_reference_queries": ENGLISH_REFERENCE_QUERIES,
    "pseudo_qrels_policy": "topic_level_common_reference_set_for_queries_with_same_topic",
}
save_config_path = OUTDIR / "multilingual_run_config.json"
with open(save_config_path, "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2, ensure_ascii=False)

zip_path = OUTDIR / "multilingual_retrieval_diagnostic_outputs.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUTDIR.glob("*"):
        if p.is_file() and p.name != zip_path.name:
            z.write(p, arcname=p.name)

print("\n[saved]", OUTDIR / "multilingual_pseudo_ndcg_summary.csv")
print("[saved]", OUTDIR / "multilingual_top3_results_detail.csv")
print("[saved]", OUTDIR / "multilingual_by_language_agg.csv")
print("[saved]", OUTDIR / "multilingual_overall_summary.json")
print("[saved]", OUTDIR / "multilingual_table.tex")
print("[saved]", OUTDIR / "suggested_manuscript_text.txt")
print("[saved]", OUTDIR / "README_multilingual_study.md")
print("[zip]", zip_path)
print("\nLaTeX table:\n")
print(latex_str)
