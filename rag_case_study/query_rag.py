#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def ensure_dir(path: str) -> None:
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
    # tokens not required at query time; kept if needed:
    # tokenized_corpus = pickle.load(open(os.path.join(index_dir, "bm25_tokens.pkl"), "rb"))
    faiss_index = faiss.read_index(os.path.join(index_dir, "faiss_hnsw_ip.index"))
    meta = pd.read_parquet(os.path.join(index_dir, "meta.parquet"))
    return cfg, bm25, faiss_index, meta


def load_embed_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model


def encode_query(model: SentenceTransformer, query: str, use_e5_prefix: bool) -> np.ndarray:
    q = f"query: {query}" if use_e5_prefix else query
    v = model.encode([q], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    # normalize for inner product
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype(np.float32)


def bm25_topk(bm25: BM25Okapi, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = tr_tokenize(query)
    scores = bm25.get_scores(tokens)
    scores = np.asarray(scores)
    if k <= 0 or k >= len(scores):
        idxs = np.argsort(-scores)
    else:
        idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
    return idxs, scores[idxs]


def faiss_topk(index: faiss.Index, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k_eff = index.ntotal if (k is None or k <= 0) else k
    scores, ids = index.search(query_vec, k_eff)
    return ids[0], scores[0]


def rrf_fuse(candidates: Dict[str, List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    # candidates: name -> list of (doc_id, rank_score_descending)
    # Convert to ranks first, then sum 1/(k + rank)
    rrf: Dict[int, float] = {}
    for name, docs in candidates.items():
        # docs assumed sorted by score desc
        for r, (doc_id, _) in enumerate(docs, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + r)
    # sort by fused score desc
    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return fused


def load_sentiment_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def infer_label_mapping(id2label: Dict[int, str]) -> Dict[int, float]:
    # Map labels to numeric sentiment in [-1, 1]
    # Try to detect 'neg', 'neu', 'pos' by name
    mapping_by_name: Dict[str, float] = {"negative": -1.0, "neg": -1.0, "positive": 1.0, "pos": 1.0, "neutral": 0.0, "neu": 0.0}
    label_map: Dict[int, float] = {}
    matched = set()
    # Special-case: labels like '1 star', '2 stars', ... (e.g., nlptown 1-5 star models)
    star_detected = False
    for i, name in id2label.items():
        lname = name.lower()
        # Try star mapping first
        m = re.search(r"\b([1-5])\s*star", lname)
        if m:
            star_detected = True
            stars = int(m.group(1))
            # Map 1..5 stars to [-1, -0.5, 0, 0.5, 1]
            star_to_val = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
            label_map[i] = float(star_to_val.get(stars, 0.0))
            matched.add(i)
            continue
        # Otherwise, map by sentiment keywords
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
    # Fallback heuristic: 2-class -> [neg, pos] as [0, 1]; 3-class -> [neg, neu, pos] as [0,1,2]
    if num_labels == 2:
        return {0: -1.0 if 0 not in label_map else label_map[0], 1: 1.0 if 1 not in label_map else label_map[1]}
    if num_labels == 3:
        default_map = {0: -1.0, 1: 0.0, 2: 1.0}
        default_map.update(label_map)
        return default_map
    # Otherwise linearly space across [-1,1]
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
        # numeric expected value in [-1,1]
        value_vec = np.array([label_value_map[j] for j in range(probs.shape[1])], dtype=np.float32)
        numeric = (probs * value_vec[None, :]).sum(axis=1)
        all_scores.append(numeric)
        all_labels.extend([id2label[int(pid)] for pid in pred_ids])
    sentiment_numeric = np.concatenate(all_scores, axis=0) if len(all_scores) > 0 else np.array([], dtype=np.float32)
    return all_labels, sentiment_numeric


def aggregate_results(df_subset: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(df_subset) == 0:
        return out
    if "sentiment_score" in df_subset.columns:
        out["sentiment_mean"] = float(df_subset["sentiment_score"].mean())
        out["sentiment_median"] = float(df_subset["sentiment_score"].median())
        out["sentiment_std"] = float(df_subset["sentiment_score"].std())
        # Class distribution if available
        if "sentiment_label" in df_subset.columns:
            dist = df_subset["sentiment_label"].value_counts(normalize=True)
            for k, v in dist.items():
                out[f"class_ratio_{k}"] = float(v)
        # Correlation with Score (map score 1..5 to [-1,1])
        if "score" in df_subset.columns:
            mapped = (df_subset["score"].astype(float) - 3.0) / 2.0
            pearson = float(np.corrcoef(mapped, df_subset["sentiment_score"])[0, 1]) if len(df_subset) >= 2 else float("nan")
            # Spearman via ranks (no scipy dependency)
            rank_score = mapped.rank(method="average")
            rank_sent = df_subset["sentiment_score"].rank(method="average")
            spearman = float(np.corrcoef(rank_score, rank_sent)[0, 1]) if len(df_subset) >= 2 else float("nan")
            out["corr_pearson_score_vs_sentiment"] = pearson
            out["corr_spearman_score_vs_sentiment"] = spearman
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query: BM25 + FAISS + (optional) rerank + Turkish sentiment + aggregation")
    parser.add_argument("--index-dir", type=str, default="/home/onur/GitHub/case/rag3/index", help="Directory containing FAISS/BM25/meta/config")
    parser.add_argument("--query", type=str, required=True, help="User query (feature/keyword/phrase)")
    parser.add_argument("--k_lex", type=int, default=0, help="BM25 candidate count (0=all)")
    parser.add_argument("--k_vec", type=int, default=0, help="FAISS candidate count (0=all)")
    parser.add_argument("--limit", type=int, default=0, help="Final top-N after fusion/rerank (0=all)")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF hyperparameter k")
    parser.add_argument("--use_reranker", action="store_true", help="Use cross-encoder reranker on fused candidates")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3", help="Cross-encoder reranker model")
    parser.add_argument("--sentiment_model", type=str, default="savasy/bert-base-turkish-sentiment-cased", help="Turkish sentiment model")
    parser.add_argument("--out_csv", type=str, default="", help="Optional path to save results CSV")
    parser.add_argument("--print_examples", type=int, default=5, help="Print first N examples")
    parser.add_argument("--min_sim", type=float, default=0.5, help="Minimum semantic similarity to include a result (RAG only, only applies to FAISS results, BM25-only results are always kept, default=0.5 for quality; 0=disable)")
    parser.add_argument("--max_sentiment", type=int, default=500, help="Max number of results to run sentiment analysis on (0=all, for speed)")
    args = parser.parse_args()

    cfg, bm25, faiss_index, meta = load_indexes(args.index_dir)
    embed_model_name = cfg.get("model_name", "trmteb/turkish-embedding-model")
    use_e5_prefix = bool(cfg.get("uses_e5_passage_prefix", True))

    embed_model = load_embed_model(embed_model_name)
    q_vec = encode_query(embed_model, args.query, use_e5_prefix=use_e5_prefix)

    # Retrieve candidates
    lex_ids, lex_scores = bm25_topk(bm25, args.query, k=args.k_lex)
    vec_ids, vec_scores = faiss_topk(faiss_index, q_vec, k=args.k_vec)

    # Prepare lists sorted by score desc
    lex_pairs = list(zip(lex_ids.tolist(), lex_scores.tolist()))
    vec_pairs = list(zip(vec_ids.tolist(), vec_scores.tolist()))
    lex_pairs.sort(key=lambda x: x[1], reverse=True)
    vec_pairs.sort(key=lambda x: x[1], reverse=True)

    fused = rrf_fuse({"bm25": lex_pairs, "faiss": vec_pairs}, k=args.rrf_k)
    # Don't apply limit yet - apply after min_sim filtering for better quality
    fused_ids = [doc_id for doc_id, _ in fused]
    # Build FAISS similarity map for later lookup (use index position, not ID column)
    # vec_ids are indices into meta DataFrame, so we map index -> score
    faiss_score_map = {int(doc_id): float(score) for doc_id, score in vec_pairs}

    # Optional rerank with cross-encoder
    if args.use_reranker and len(fused_ids) > 0:
        reranker = CrossEncoder(args.reranker_model)
        pairs = [(args.query, str(meta.iloc[i]["feedback"])) for i in fused_ids]
        scores = reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        order = np.argsort(-scores)
        fused_ids = [fused_ids[i] for i in order]

    # Slice meta
    result_df = meta.iloc[fused_ids].copy().reset_index().rename(columns={"index": "row_id"}) if len(fused_ids) > 0 else pd.DataFrame(columns=meta.columns)
    # Add FAISS similarity scores to DataFrame (use only FAISS scores, no additional computation)
    # fused_ids are indices into meta, so we can directly use them to look up FAISS scores.
    if len(result_df) > 0:
        # Use only FAISS scores (fast lookup, no embedding computation)
        sim_list = [faiss_score_map.get(int(idx), 0.0) for idx in fused_ids]
        # Apply similarity threshold filtering only if min_sim > 0
        # Keep BM25-only results but limit them to prevent too many low-quality results
        if (args.min_sim is not None) and (args.min_sim > 0.0):
            keep_positions = []
            bm25_only_count = 0
            max_bm25_only = 1000  # Limit BM25-only results to prevent spam
            for i, sim in enumerate(sim_list):
                # Keep if: has FAISS score and sim >= min_sim, OR no FAISS score (BM25-only, limited)
                has_faiss = int(fused_ids[i]) in faiss_score_map
                if has_faiss:
                    # FAISS result: apply min_sim filter
                    if sim >= args.min_sim:
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
        if args.limit > 0 and len(fused_ids) > args.limit:
            fused_ids = fused_ids[: args.limit]
            result_df = result_df.iloc[: args.limit].reset_index(drop=True)
            sim_list = sim_list[: args.limit]
        result_df["faiss_similarity"] = sim_list

    # Sentiment inference: Use pre-computed if available, otherwise compute
    if len(result_df) > 0:
        # Check if pre-computed sentiment exists in meta
        if "sentiment_label" in meta.columns and "sentiment_score" in meta.columns:
            # Use pre-computed sentiment (fast lookup)
            result_df["sentiment_label"] = meta.iloc[fused_ids]["sentiment_label"].values
            result_df["sentiment_score"] = meta.iloc[fused_ids]["sentiment_score"].values
            # Reset index to align with result_df
            result_df = result_df.reset_index(drop=True)
        else:
            # Fallback: Compute sentiment on-the-fly (backward compatibility)
            tokenizer, sentiment_model = load_sentiment_model(args.sentiment_model)
            # Limit sentiment analysis to first N results for speed (0 = all)
            n_sent = len(result_df) if (args.max_sentiment <= 0) else min(args.max_sentiment, len(result_df))
            if n_sent < len(result_df):
                # Only analyze first N, fill rest with NaN
                texts_sent = result_df["feedback"].astype(str).tolist()[:n_sent]
                labels, numeric = sentiment_predict(tokenizer, sentiment_model, texts_sent, batch_size=256)
                # Extend with NaN for remaining rows
                labels.extend([None] * (len(result_df) - n_sent))
                numeric = np.concatenate([numeric, np.full(len(result_df) - n_sent, np.nan, dtype=np.float32)])
            else:
                labels, numeric = sentiment_predict(tokenizer, sentiment_model, result_df["feedback"].astype(str).tolist(), batch_size=256)
            result_df["sentiment_label"] = labels
            result_df["sentiment_score"] = numeric
    else:
        result_df["sentiment_label"] = []
        result_df["sentiment_score"] = []

    # Aggregation
    summary = aggregate_results(result_df)
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Print examples
    if args.print_examples > 0 and len(result_df) > 0:
        print("\nExamples:")
        for i in range(min(args.print_examples, len(result_df))):
            row = result_df.iloc[i]
            snippet = str(row["feedback"])
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            if "faiss_similarity" in result_df.columns:
                sim_score = row.get("faiss_similarity", 0.0)
                print(f"- ID={row['id']} score={row['score']} sent=({row['sentiment_label']}, {row['sentiment_score']:.3f}) sim={sim_score:.3f} title={row['title']}\n  {snippet}")
            else:
                print(f"- ID={row['id']} score={row['score']} sent=({row['sentiment_label']}, {row['sentiment_score']:.3f}) title={row['title']}\n  {snippet}")

    # Save CSV if requested
    if args.out_csv:
        ensure_dir(os.path.dirname(args.out_csv)) if os.path.dirname(args.out_csv) else None
        result_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved results to {args.out_csv}")


if __name__ == "__main__":
    main()


