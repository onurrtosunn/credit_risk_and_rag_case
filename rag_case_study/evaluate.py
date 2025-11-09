#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import warnings
from collections import Counter

# Import visualization functions
import sys
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from visualization import (
    plot_sentiment_distribution,
    plot_topic_frequency,
    plot_sentiment_by_score,
    plot_correlation_heatmap,
    plot_hidden_risks_and_strengths,
    detect_hidden_risks_and_strengths,
    ensure_dir as viz_ensure_dir
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def tr_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    return text.split() if text else []


def load_indexes(index_dir: str):
    with open(os.path.join(index_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bm25: BM25Okapi = pickle.load(open(os.path.join(index_dir, "bm25.pkl"), "rb"))
    faiss_index = faiss.read_index(os.path.join(index_dir, "faiss_hnsw_ip.index"))
    meta = pd.read_parquet(os.path.join(index_dir, "meta.parquet"))
    return cfg, bm25, faiss_index, meta


def load_embed_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_query(model: SentenceTransformer, query: str, use_e5_prefix: bool) -> np.ndarray:
    q = f"query: {query}" if use_e5_prefix else query
    v = model.encode([q], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype(np.float32)


def bm25_topk(bm25: BM25Okapi, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = tr_tokenize(query)
    scores = bm25.get_scores(tokens)
    scores = np.asarray(scores)
    if k >= len(scores):
        idxs = np.argsort(-scores)
    else:
        idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
    return idxs, scores[idxs]


def faiss_topk(index: faiss.Index, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(query_vec, k)
    return ids[0], scores[0]


def rrf_fuse(candidates: Dict[str, List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    rrf: Dict[int, float] = {}
    for _, docs in candidates.items():
        for r, (doc_id, _) in enumerate(docs, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + r)
    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return fused


def load_sentiment_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def infer_label_mapping(id2label: Dict[int, str]) -> Dict[int, float]:
    mapping_by_name: Dict[str, float] = {"negative": -1.0, "neg": -1.0, "positive": 1.0, "pos": 1.0, "neutral": 0.0, "neu": 0.0}
    label_map: Dict[int, float] = {}
    matched = set()
    star_detected = False
    for i, name in id2label.items():
        lname = name.lower()
        m = re.search(r"\b([1-5])\s*star", lname)
        if m:
            star_detected = True
            stars = int(m.group(1))
            star_to_val = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
            label_map[i] = float(star_to_val.get(stars, 0.0))
            matched.add(i)
            continue
        score = None
        for key, val in mapping_by_name.items():
            if key in lname:
                score = val
                break
        if score is not None:
            label_map[i] = score
            matched.add(i)
    num_labels = len(id2label)
    if len(matched) == num_labels and (star_detected or num_labels in (2, 3)):
        return label_map
    if num_labels == 2:
        return {0: -1.0 if 0 not in label_map else label_map[0], 1: 1.0 if 1 not in label_map else label_map[1]}
    if num_labels == 3:
        default_map = {0: -1.0, 1: 0.0, 2: 1.0}
        default_map.update(label_map)
        return default_map
    ordered = sorted(id2label.keys())
    values = np.linspace(-1.0, 1.0, num=len(ordered))
    return {i: float(v) for i, v in zip(ordered, values)}


@torch.inference_mode()
def sentiment_predict(tokenizer, model, texts: List[str], batch_size: int = 32, max_length: int = 256):
    if len(texts) == 0:
        return [], np.array([], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    id2label = {i: l for i, l in enumerate(model.config.id2label.values())} if isinstance(model.config.id2label, dict) else {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
    label_value_map = infer_label_mapping(id2label)
    all_scores = []
    all_labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        value_vec = np.array([label_value_map[j] for j in range(probs.shape[1])], dtype=np.float32)
        numeric = (probs * value_vec[None, :]).sum(axis=1)
        all_scores.append(numeric)
        all_labels.extend([id2label[int(pid)] for pid in pred_ids])
    sentiment_numeric = np.concatenate(all_scores, axis=0) if len(all_scores) > 0 else np.array([], dtype=np.float32)
    return all_labels, sentiment_numeric


def aggregate_metrics(df_subset: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(df_subset) == 0:
        return out
    if "sentiment_score" in df_subset.columns:
        out["sentiment_mean"] = float(df_subset["sentiment_score"].mean())
        out["sentiment_median"] = float(df_subset["sentiment_score"].median())
        out["sentiment_std"] = float(df_subset["sentiment_score"].std())
        if "sentiment_label" in df_subset.columns:
            dist = df_subset["sentiment_label"].value_counts(normalize=True)
            for k, v in dist.items():
                out[f"class_ratio_{k}"] = float(v)
        if "score" in df_subset.columns:
            mapped = (df_subset["score"].astype(float) - 3.0) / 2.0
            # Safe Pearson (avoid divide-by-zero / NaN when variance is zero)
            def safe_pearson(a: pd.Series, b: pd.Series) -> float:
                if len(a) < 2:
                    return float("nan")
                a_std = float(a.std())
                b_std = float(b.std())
                if a_std == 0.0 or b_std == 0.0:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])
            pearson = safe_pearson(mapped, df_subset["sentiment_score"])
            # Safe Spearman via ranks
            def safe_spearman(a: pd.Series, b: pd.Series) -> float:
                if len(a) < 2:
                    return float("nan")
                ra = a.rank(method="average")
                rb = b.rank(method="average")
                return safe_pearson(ra, rb)
            spearman = safe_spearman(mapped, df_subset["sentiment_score"])
            out["corr_pearson_score_vs_sentiment"] = pearson
            out["corr_spearman_score_vs_sentiment"] = spearman
            # Business-facing rates: % negative for score<=2, % positive for score>=3
            def to_class(x: float) -> str:
                if x <= -0.2:
                    return "negative"
                if x >= 0.2:
                    return "positive"
                return "neutral"
            if "sentiment_score" in df_subset.columns:
                sent_classes = df_subset["sentiment_score"].apply(to_class)
                df_tmp = df_subset.assign(_sent_class=sent_classes)
                low = df_tmp[df_tmp["score"] <= 2]
                high = df_tmp[df_tmp["score"] >= 3]
                neg_rate_low = float((low["_sent_class"] == "negative").mean()) if len(low) > 0 else float("nan")
                pos_rate_high = float((high["_sent_class"] == "positive").mean()) if len(high) > 0 else float("nan")
                out["rate_negative_for_score_le_2"] = neg_rate_low
                out["rate_positive_for_score_ge_3"] = pos_rate_high
    return out


def run_baseline(query: str, bm25: BM25Okapi, meta: pd.DataFrame, tokenizer, sent_model, k_lex: int, limit: int):
    t0 = time.perf_counter()
    lex_ids, lex_scores = bm25_topk(bm25, query, k=k_lex)
    retrieve_ms = (time.perf_counter() - t0) * 1000.0
    # Filter out zero-score matches to avoid unrelated results
    if len(lex_ids) > 0:
        mask = lex_scores > 0
        filtered_ids = lex_ids[mask]
        # If no positive BM25 scores, fallback to substring match over full corpus (feedback/title)
        if filtered_ids.size == 0:
            fb = meta["feedback"].astype(str).str.contains(query, case=False, na=False, regex=False)
            tt = meta["title"].astype(str).str.contains(query, case=False, na=False, regex=False) if "title" in meta.columns else pd.Series([False] * len(meta))
            sub_mask = (fb | tt).values
            fallback_ids = np.where(sub_mask)[0]
            if len(fallback_ids) > 0:
                filtered_ids = fallback_ids
    else:
        filtered_ids = lex_ids
    doc_ids = filtered_ids.tolist()
    if limit > 0:
        doc_ids = doc_ids[: limit]
    result_df = meta.iloc[doc_ids].copy().reset_index(drop=True) if len(doc_ids) > 0 else pd.DataFrame(columns=meta.columns)
    t1 = time.perf_counter()
    if len(result_df) > 0:
        # Check if pre-computed sentiment exists in meta
        if "sentiment_label" in meta.columns and "sentiment_score" in meta.columns:
            # Use pre-computed sentiment (fast lookup - T6 becomes ~0ms)
            result_df["sentiment_label"] = meta.iloc[doc_ids]["sentiment_label"].values
            result_df["sentiment_score"] = meta.iloc[doc_ids]["sentiment_score"].values
        else:
            # Fallback: Compute sentiment on-the-fly (backward compatibility)
            labels, numeric = sentiment_predict(tokenizer, sent_model, result_df["feedback"].astype(str).tolist())
            result_df["sentiment_label"] = labels
            result_df["sentiment_score"] = numeric
    else:
        result_df["sentiment_label"] = []
        result_df["sentiment_score"] = []
    sent_ms = (time.perf_counter() - t1) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    metrics = aggregate_metrics(result_df)
    metrics.update({"latency_ms_retrieval": retrieve_ms, "latency_ms_sentiment": sent_ms, "latency_ms_total": total_ms, "count": int(len(result_df))})
    return result_df, metrics


def run_rag(query: str, bm25: BM25Okapi, faiss_index, embed_model, use_e5_prefix: bool, meta: pd.DataFrame, tokenizer, sent_model, k_lex: int, k_vec: int, limit: int, use_reranker: bool, reranker_model_name: str, rrf_k: int, min_sim: float, max_sentiment: int = 1000):
    t0 = time.perf_counter()
    lex_ids, lex_scores = bm25_topk(bm25, query, k=k_lex)
    q_vec = encode_query(embed_model, query, use_e5_prefix=use_e5_prefix)
    vec_ids, vec_scores = faiss_topk(faiss_index, q_vec, k=k_vec)
    lex_pairs = list(zip(lex_ids.tolist(), lex_scores.tolist()))
    vec_pairs = list(zip(vec_ids.tolist(), vec_scores.tolist()))
    lex_pairs.sort(key=lambda x: x[1], reverse=True)
    vec_pairs.sort(key=lambda x: x[1], reverse=True)
    fused = rrf_fuse({"bm25": lex_pairs, "faiss": vec_pairs}, k=rrf_k)
    # Don't apply limit yet - apply after min_sim filtering for better quality
    fused_ids = [doc_id for doc_id, _ in fused]
    # Build FAISS similarity map for later lookup (index position -> score)
    faiss_score_map = {int(doc_id): float(score) for doc_id, score in vec_pairs}
    if use_reranker and len(fused_ids) > 0:
        reranker = CrossEncoder(reranker_model_name)
        pairs = [(query, str(meta.iloc[i]["feedback"])) for i in fused_ids]
        scores = reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        order = np.argsort(-scores)
        fused_ids = [fused_ids[i] for i in order]
    retrieve_ms = (time.perf_counter() - t0) * 1000.0
    result_df = meta.iloc[fused_ids].copy().reset_index(drop=True) if len(fused_ids) > 0 else pd.DataFrame(columns=meta.columns)
    # Add FAISS similarity scores to DataFrame (use only FAISS scores, no additional computation)
    if len(result_df) > 0:
        # Use only FAISS scores (fast lookup, no embedding computation)
        sim_list = [faiss_score_map.get(int(idx), 0.0) for idx in fused_ids]
        # Apply similarity threshold filtering only if min_sim > 0
        # Keep BM25-only results but limit them to prevent too many low-quality results
        if min_sim > 0:
            keep_positions = []
            bm25_only_count = 0
            max_bm25_only = 1000  # Limit BM25-only results to prevent spam
            for i, sim in enumerate(sim_list):
                # Keep if: has FAISS score and sim >= min_sim, OR no FAISS score (BM25-only, limited)
                has_faiss = int(fused_ids[i]) in faiss_score_map
                if has_faiss:
                    # FAISS result: apply min_sim filter
                    if sim >= min_sim:
                        keep_positions.append(i)
                else:
                    # BM25-only result: keep but limit count
                    if bm25_only_count < max_bm25_only:
                        keep_positions.append(i)
                        bm25_only_count += 1
            if len(keep_positions) < len(sim_list):
                fused_ids = [fused_ids[i] for i in keep_positions]
                result_df = result_df.iloc[keep_positions].reset_index(drop=True)
                sim_list = [sim_list[i] for i in keep_positions]
        # Apply limit after filtering (for quality)
        if limit > 0 and len(fused_ids) > limit:
            fused_ids = fused_ids[: limit]
            result_df = result_df.iloc[: limit].reset_index(drop=True)
            sim_list = sim_list[: limit]
        result_df["faiss_similarity"] = sim_list
    t1 = time.perf_counter()
    if len(result_df) > 0:
        # Check if pre-computed sentiment exists in meta
        if "sentiment_label" in meta.columns and "sentiment_score" in meta.columns:
            # Use pre-computed sentiment (fast lookup - T6 becomes ~0ms)
            result_df["sentiment_label"] = meta.iloc[fused_ids]["sentiment_label"].values
            result_df["sentiment_score"] = meta.iloc[fused_ids]["sentiment_score"].values
            # Reset index to align with result_df
            result_df = result_df.reset_index(drop=True)
        else:
            # Fallback: Compute sentiment on-the-fly (backward compatibility)
            # Limit sentiment analysis to first N results for speed (0 = all)
            n_sent = len(result_df) if (max_sentiment <= 0) else min(max_sentiment, len(result_df))
            if n_sent < len(result_df):
                # Only analyze first N, fill rest with NaN
                texts_sent = result_df["feedback"].astype(str).tolist()[:n_sent]
                labels, numeric = sentiment_predict(tokenizer, sent_model, texts_sent, batch_size=256)  # Increased batch size
                # Extend with NaN for remaining rows
                labels.extend([None] * (len(result_df) - n_sent))
                numeric = np.concatenate([numeric, np.full(len(result_df) - n_sent, np.nan, dtype=np.float32)])
            else:
                labels, numeric = sentiment_predict(tokenizer, sent_model, result_df["feedback"].astype(str).tolist(), batch_size=256)  # Increased batch size
            result_df["sentiment_label"] = labels
            result_df["sentiment_score"] = numeric
    else:
        result_df["sentiment_label"] = []
        result_df["sentiment_score"] = []
    sent_ms = (time.perf_counter() - t1) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    metrics = aggregate_metrics(result_df)
    metrics.update({"latency_ms_retrieval": retrieve_ms, "latency_ms_sentiment": sent_ms, "latency_ms_total": total_ms, "count": int(len(result_df))})
    return result_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG vs Non-RAG (BM25) on sentiment correlation and latency")
    parser.add_argument("--index-dir", type=str, default="/home/onur/GitHub/case/rag3/index")
    parser.add_argument("--queries", type=str, default="", help="Comma-separated queries")
    parser.add_argument("--queries-file", type=str, default="", help="File with one query per line")
    parser.add_argument("--sentiment_model", type=str, default="savasy/bert-base-turkish-sentiment-cased")
    parser.add_argument("--k_lex", type=int, default=50000, help="BM25 candidate count (0=all)")
    parser.add_argument("--k_vec", type=int, default=50000, help="FAISS candidate count (0=all)")
    parser.add_argument("--limit", type=int, default=50000, help="Final top-N after fusion/rerank (0=all, default=1000 for quality)")
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="/home/onur/GitHub/case/rag3/eval")
    parser.add_argument("--print_examples", type=int, default=5, help="Print first N RAG examples per query")
    parser.add_argument("--min_sim", type=float, default=0.7, help="Minimum semantic similarity to include a result (RAG only, only applies to FAISS results, BM25-only results are always kept, default=0.5 for quality)")
    parser.add_argument("--max_sentiment", type=int, default=500, help="Max number of results to run sentiment analysis on (0=all, for speed)")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cfg, bm25, faiss_index, meta = load_indexes(args.index_dir)
    embed_model_name = cfg.get("model_name", "intfloat/multilingual-e5-small")
    use_e5_prefix = bool(cfg.get("uses_e5_passage_prefix", True))
    embed_model = load_embed_model(embed_model_name)

    tokenizer, sent_model = load_sentiment_model(args.sentiment_model)

    queries: List[str] = []
    if args.queries:
        queries.extend([q.strip() for q in args.queries.split(",") if q.strip()])
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    queries.append(t)
    if not queries:
        raise SystemExit("Provide --queries or --queries-file")

    rows = []
    for q in queries:
        base_df, base_metrics = run_baseline(
            query=q,
            bm25=bm25,
            meta=meta,
            tokenizer=tokenizer,
            sent_model=sent_model,
            k_lex=args.k_lex,
            limit=args.limit,
        )
        rag_df, rag_metrics = run_rag(
            query=q,
            bm25=bm25,
            faiss_index=faiss_index,
            embed_model=embed_model,
            use_e5_prefix=use_e5_prefix,
            meta=meta,
            tokenizer=tokenizer,
            sent_model=sent_model,
            k_lex=args.k_lex,
            k_vec=args.k_vec,
            limit=args.limit,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            rrf_k=args.rrf_k,
            min_sim=args.min_sim,
            max_sentiment=args.max_sentiment,
        )
        safe_query = re.sub(r"\W+", "_", q)[:40]
        base_csv = os.path.join(args.out_dir, f"baseline_{safe_query}.csv")
        rag_csv = os.path.join(args.out_dir, f"rag_{safe_query}.csv")
        
        # Save only print_examples rows to CSV
        if len(base_df) > 0:
            base_df.head(args.print_examples).to_csv(base_csv, index=False)
        if len(rag_df) > 0:
            rag_df.head(args.print_examples).to_csv(rag_csv, index=False)
        # ---------- Standardized Output Structure ----------
        def label_dist(df: pd.DataFrame) -> Dict[str, float]:
            if "sentiment_label" not in df.columns or len(df) == 0:
                return {}
            vc = df["sentiment_label"].value_counts(normalize=True)
            return {k: float(v) for k, v in vc.items()}
        def mean_score(df: pd.DataFrame) -> float:
            return float(df["sentiment_score"].mean()) if "sentiment_score" in df.columns and len(df) > 0 else float("nan")
        def print_examples(df: pd.DataFrame, n: int, header: str, show_similarity: bool = False) -> None:
            print(header)
            n_show = min(max(n, 0), len(df))
            for i in range(n_show):
                row = df.iloc[i]
                snippet = str(row["feedback"])
                if len(snippet) > 160:
                    snippet = snippet[:160] + "..."
                if show_similarity and "faiss_similarity" in df.columns:
                    sim_score = row.get("faiss_similarity", 0.0)
                    print(f"- ID={row['id']} score={row['score']} sent=({row['sentiment_label']}, {row['sentiment_score']:.3f}) sim={sim_score:.3f} title={row['title']}\n  {snippet}")
                else:
                    print(f"- ID={row['id']} score={row['score']} sent=({row['sentiment_label']}, {row['sentiment_score']:.3f}) title={row['title']}\n  {snippet}")

        base_labels = label_dist(base_df)
        rag_labels = label_dist(rag_df)
        base_mean = mean_score(base_df)
        rag_mean = mean_score(rag_df)
        base_n = int(base_metrics.get("count", len(base_df)))
        rag_n = int(rag_metrics.get("count", len(rag_df)))

        # 1. Input Topic
        print("\n1. Input Topic")
        print(q)

        # 2. Non-RAG Retrieval Result
        print("\n2. Non-RAG Retrieval Result")
        print(f"Number of matched feedback entries: {base_n}")
        if args.print_examples > 0:
            print_examples(base_df, args.print_examples, "Comments:")
        print("Sentiment summary:")
        print(f"- Counts (ratios): {base_labels}")
        print(f"- Mean score: {base_mean:.3f}")
        print("Short interpretation: BM25-based matches; sensitive to word matching, may miss semantic variants.")

        # 3. RAG Retrieval Result
        print("\n3. RAG Retrieval Result")
        print(f"Number of matched feedback entries: {rag_n}")
        if args.print_examples > 0:
            print_examples(rag_df, args.print_examples, "Comments (semantic):", show_similarity=True)
        print("Sentiment summary:")
        print(f"- Counts (ratios): {rag_labels}")
        print(f"- Mean score: {rag_mean:.3f}")
        print("Short interpretation: Tends to capture broader/implicitly related comments through semantic similarity.")

        # 4. Comparison Summary (tabular style)
        print("\n4. Comparison Summary")
        def get_ratio(d: Dict[str, float], key: str) -> float:
            return float(d.get(key, 0.0))
        pos_diff = get_ratio(rag_labels, "positive") - get_ratio(base_labels, "positive")
        neg_diff = get_ratio(rag_labels, "negative") - get_ratio(base_labels, "negative")
        neu_diff = get_ratio(rag_labels, "neutral") - get_ratio(base_labels, "neutral")
        mean_diff = rag_mean - base_mean
        cover_diff = rag_n - base_n
        # ASCII table formatting
        headers = ["Aspect", "Non-RAG", "RAG", "Difference"]
        rows_tbl = [
            ["Coverage (N)", f"{base_n}", f"{rag_n}", f"RAG {('+' if cover_diff>=0 else '')}{cover_diff} (number of results difference)"],
            ["Positive ratio", f"{get_ratio(base_labels,'positive'):.2f}", f"{get_ratio(rag_labels,'positive'):.2f}", f"Δ {pos_diff:+.2f}"],
            ["Negative ratio", f"{get_ratio(base_labels,'negative'):.2f}", f"{get_ratio(rag_labels,'negative'):.2f}", f"Δ {neg_diff:+.2f}"],
            ["Neutral ratio", f"{get_ratio(base_labels,'neutral'):.2f}", f"{get_ratio(rag_labels,'neutral'):.2f}", f"Δ {neu_diff:+.2f}"],
            ["Mean sentiment", f"{base_mean:.3f}", f"{rag_mean:.3f}", f"Δ {mean_diff:+.3f}"],
            ["Latency (retrieval ms)", f"{base_metrics.get('latency_ms_retrieval', float('nan')):.0f}", f"{rag_metrics.get('latency_ms_retrieval', float('nan')):.0f}", " "],
            ["Latency (total ms)", f"{base_metrics.get('latency_ms_total', float('nan')):.0f}", f"{rag_metrics.get('latency_ms_total', float('nan')):.0f}", ""],
        ]
        col_widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows_tbl]) for i, h in enumerate(headers)]
        def print_sep():
            print("+" + "+".join("-" * (w + 2) for w in col_widths) + "+")
        def print_row(vals):
            print("| " + " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(vals)) + " |")
        print_sep()
        print_row(headers)
        print_sep()
        for r in rows_tbl:
            print_row(r)
        print_sep()

        # ---------- Per-query narrative insights ----------
        def to_class(x: float) -> str:
            if x <= -0.2:
                return "negative"
            if x >= 0.2:
                return "positive"
            return "neutral"
        rd = rag_df.copy()
        rd["_sent_class"] = rd["sentiment_score"].apply(to_class)
        neg_rate = float((rd["_sent_class"] == "negative").mean()) if len(rd) > 0 else float("nan")
        pos_rate = float((rd["_sent_class"] == "positive").mean()) if len(rd) > 0 else float("nan")
        # Keyword extraction on negative subset (unigram + bigram)
        STOP_TR = {
            "ve","veya","ile","ama","gibi","çok","daha","mi","mı","mu","mü","de","da","ki","bu","şu","o",
            "bir","birçok","her","hiç","olan","oldu","olması","olarak","için","üzere","ise","yada","yani",
            "neden","niye","çünkü","fakat","ancak","ben","biz","siz","onlar","hepsi","şey","şeyi",
            "var","yok","yine","az","en","olanlar","ettim","ettik","ettiniz","etti","etmek","hem",
            "şeklinde","konu","konuda","durum","durumda"
        }
        def tokenize_tr(text: str) -> List[str]:
            text = str(text).lower()
            text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
            toks = [t for t in re.sub(r"\s+"," ",text).strip().split(" ") if t and t not in STOP_TR and len(t) >= 2]
            return toks
        def top_terms(texts: List[str], topk: int = 10) -> List[str]:
            uni = Counter()
            bi = Counter()
            for t in texts:
                toks = tokenize_tr(t)
                uni.update(toks)
                if len(toks) >= 2:
                    bigs = [f"{a} {b}" for a, b in zip(toks[:-1], toks[1:])]
                    bi.update(bigs)
            items = [(k, v) for k, v in bi.items()] + [(k, v) for k, v in uni.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            return [k for k, _ in items[:topk]]
        neg_texts = rd.loc[rd["_sent_class"] == "negative", "feedback"].astype(str).tolist()
        neg_top = top_terms(neg_texts, topk=5) if neg_texts else []
        # Correlation call (reuse rag_metrics if present)
        pearson = rag_metrics.get("corr_pearson_score_vs_sentiment", float("nan"))
        corr_txt = "high" if isinstance(pearson, (float,int)) and abs(float(pearson)) >= 0.5 else "medium/low"
        corr_pct = abs(float(pearson)) * 100.0 if isinstance(pearson, (float, int)) else float("nan")
        # 5. Business Insight Summary (Actionable)
        print("\n5. Business Insight Summary")
        if neg_rate == neg_rate:
            if neg_rate >= 0.6:
                print("• Dissatisfaction is high; investigate process and communication steps in depth.")
            elif neg_rate >= 0.4:
                print("• Dissatisfaction is medium; plan quick improvements for frequently occurring complaint patterns.")
            else:
                print("• Dissatisfaction is low; processes are generally healthy but review outlier complaints.")
        if neg_top:
            print(f"• Root-cause hints (prominent in negatives): {', '.join(neg_top[:5])}")
        if pos_rate == pos_rate:
            if pos_rate >= 0.6:
                print("• Positive comments are dominant; scale up steps that produce good experiences.")
            elif pos_rate <= 0.3:
                print("• Positive rate is low; strengthen touchpoints that deliver value to customers.")
        if corr_pct == corr_pct:
            print(f"• Customer score vs sentiment correlation is {corr_txt} ({corr_pct:.0f}%); text-based sentiment aligns with scores.")
        else:
            print(f"• Customer score vs sentiment correlation is {corr_txt}.")

        # 6. Visualization Recommendations (text-only)
        print("\n6. Visualization Recommendations")
        print("- Pie chart: Sentiment distribution (positive/negative/neutral)")
        print("- Bar chart: Theme/keyword frequencies (negative subset)")

        # 7. Generate visualizations for RAG results
        print("\n7. Generating Visualizations for RAG Results")
        charts_dir = os.path.join(args.out_dir, "charts")
        viz_ensure_dir(charts_dir)
        
        # Generate visualizations with query name in filename
        query_prefix = safe_query
        
        try:
            # 1. Sentiment distribution
            plot_sentiment_distribution(
                rag_df,
                os.path.join(charts_dir, f"sentiment_distribution_{query_prefix}.png")
            )
            
            # 2. Topic frequency
            plot_topic_frequency(
                rag_df,
                os.path.join(charts_dir, f"topic_frequency_{query_prefix}.png"),
                topk=15
            )
            
            # 3. Sentiment by score
            plot_sentiment_by_score(
                rag_df,
                os.path.join(charts_dir, f"sentiment_by_score_{query_prefix}.png")
            )
            
            # 4. Correlation heatmap (Title x Score x sentiment_score)
            plot_correlation_heatmap(
                rag_df,
                os.path.join(charts_dir, f"correlation_heatmap_{query_prefix}.png"),
                top_topics=20
            )
            
            # 5. Hidden risks and strengths
            plot_hidden_risks_and_strengths(
                rag_df,
                os.path.join(charts_dir, f"hidden_risks_strengths_{query_prefix}.png"),
                topk=10
            )
            
            # Print hidden risks and strengths summary
            detection = detect_hidden_risks_and_strengths(rag_df)
            if detection["risk_count"] > 0 or detection["strength_count"] > 0:
                print(f"\nHidden Risks & Strengths:")
                print(f"- Hidden Risk Count (Score 4-5 but Sentiment Negative): {detection['risk_count']}")
                print(f"- Hidden Strength Count (Score 1-2 but Sentiment Positive): {detection['strength_count']}")
                
                if detection["risk_count"] > 0:
                    print(f"\n⚠️  Hidden Risks Detected:")
                    print(f"   Customers gave high scores (4-5) but content is negative.")
                    print(f"   This indicates customers are polite but actually dissatisfied.")
                
                if detection["strength_count"] > 0:
                    print(f"\n✅ Hidden Strengths Detected:")
                    print(f"   Customers gave low scores (1-2) but mentioned something positive in content.")
                    print(f"   This shows improvement opportunities.")
            
            print(f"\n✓ All visualizations saved to: {charts_dir}/")
            
        except Exception as e:
            print(f"\n⚠️  Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

        rows.append({
            "query": q,
            "baseline": base_metrics,
            "rag": rag_metrics,
        })

    # Flatten metrics into a table
    records = []
    def flat(prefix: str, d: Dict[str, float]) -> Dict[str, float]:
        return {f"{prefix}.{k}": v for k, v in d.items()}
    for r in rows:
        rec = {"query": r["query"]}
        rec.update(flat("baseline", r["baseline"]))
        rec.update(flat("rag", r["rag"]))
        records.append(rec)
    df_metrics = pd.DataFrame.from_records(records)
    metrics_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved per-query metrics to {metrics_csv}")

    # Print business commentary (averaged across queries) for RAG (guard if columns missing)
    col_neg = "rag.rate_negative_for_score_le_2"
    col_pos = "rag.rate_positive_for_score_ge_3"
    if col_neg in df_metrics.columns and col_pos in df_metrics.columns:
        rag_neg = df_metrics[col_neg].dropna()
        rag_pos = df_metrics[col_pos].dropna()
        if len(rag_neg) > 0 and len(rag_pos) > 0:
            neg_pct = 100.0 * float(rag_neg.mean())
            pos_pct = 100.0 * float(rag_pos.mean())
            print("\nBusiness commentary (RAG):")
            print(f"- {neg_pct:.0f}% of users who gave scores 1-2 had negative sentiment analysis.")
            print(f"- {pos_pct:.0f}% positive sentiment is seen in scores 3 and above. This correlation shows that our sentiment analysis aligns with scores.")


if __name__ == "__main__":
    main()


