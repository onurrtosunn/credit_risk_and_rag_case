#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Tuple
import functools
try:
    from zeyrek import MorphAnalyzer
    _TR_LEM_BASE = MorphAnalyzer()
except Exception:
    _TR_LEM_BASE = None

import numpy as np
import pandas as pd
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@functools.lru_cache(maxsize=200_000)
def _lemma_base(tok: str) -> str:
    if not _TR_LEM_BASE:
        return tok
    try:
        lem = _TR_LEM_BASE.lemmatize(tok)
        if isinstance(lem, list) and len(lem) > 0:
            cand = lem[0][0] if isinstance(lem[0], (list, tuple)) and len(lem[0]) > 0 else lem[0]
            return str(cand)
    except Exception:
        pass
    try:
        analyses = _TR_LEM_BASE.analyze(tok)
        if analyses:
            first = analyses[0]
            cand = getattr(first, "lemma", None)
            if cand:
                return str(cand)
    except Exception:
        pass
    return tok

def tr_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    toks = text.split() if text else []
    return [_lemma_base(t) for t in toks]


def load_bm25_and_meta(index_dir: str):
    bm25: BM25Okapi = pickle.load(open(os.path.join(index_dir, "bm25.pkl"), "rb"))
    meta = pd.read_parquet(os.path.join(index_dir, "meta.parquet"))
    return bm25, meta


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


def load_sentiment_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def infer_label_mapping(id2label: Dict[int, str]) -> Dict[int, float]:
    # Robust mapping for 2/3-class and 1..5 stars
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


def aggregate_results(df_subset: pd.DataFrame) -> Dict[str, float]:
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
            pearson = float(np.corrcoef(mapped, df_subset["sentiment_score"])[0, 1]) if len(df_subset) >= 2 else float("nan")
            rank_score = mapped.rank(method="average")
            rank_sent = df_subset["sentiment_score"].rank(method="average")
            spearman = float(np.corrcoef(rank_score, rank_sent)[0, 1]) if len(df_subset) >= 2 else float("nan")
            out["corr_pearson_score_vs_sentiment"] = pearson
            out["corr_spearman_score_vs_sentiment"] = spearman
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-RAG baseline: BM25 retrieval + sentiment + aggregation")
    parser.add_argument("--index-dir", type=str, default="/home/onur/GitHub/case/rag3/index", help="Directory containing BM25/meta")
    parser.add_argument("--query", type=str, required=True, help="User query (feature/keyword/phrase)")
    parser.add_argument("--k_lex", type=int, default=0, help="BM25 candidate count (0=all)")
    parser.add_argument("--limit", type=int, default=0, help="Final top-N to score (0=all)")
    parser.add_argument("--sentiment_model", type=str, default="savasy/bert-base-turkish-sentiment-cased", help="Turkish sentiment model")
    parser.add_argument("--out_csv", type=str, default="", help="Optional path to save results CSV")
    parser.add_argument("--print_examples", type=int, default=5, help="Print first N examples")
    args = parser.parse_args()

    bm25, meta = load_bm25_and_meta(args.index_dir)

    lex_ids, lex_scores = bm25_topk(bm25, args.query, k=args.k_lex)
    # Filter zero-score matches
    if len(lex_ids) > 0:
        mask = lex_scores > 0
        filtered_ids = lex_ids[mask]
    else:
        filtered_ids = lex_ids
    doc_ids = filtered_ids.tolist()
    if args.limit > 0:
        doc_ids = doc_ids[: args.limit]

    result_df = meta.iloc[doc_ids].copy().reset_index().rename(columns={"index": "row_id"}) if len(doc_ids) > 0 else pd.DataFrame(columns=meta.columns)
    tokenizer, sentiment_model = load_sentiment_model(args.sentiment_model)
    if len(result_df) > 0:
        labels, numeric = sentiment_predict(tokenizer, sentiment_model, result_df["feedback"].astype(str).tolist())
        result_df["sentiment_label"] = labels
        result_df["sentiment_score"] = numeric
    else:
        result_df["sentiment_label"] = []
        result_df["sentiment_score"] = []

    summary = aggregate_results(result_df)
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.print_examples > 0 and len(result_df) > 0:
        print("\nExamples:")
        for i in range(min(args.print_examples, len(result_df))):
            row = result_df.iloc[i]
            snippet = str(row["feedback"])
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"- ID={row['id']} score={row['score']} sent=({row['sentiment_label']}, {row['sentiment_score']:.3f}) title={row['title']}\n  {snippet}")

    if args.out_csv:
        ensure_dir(os.path.dirname(args.out_csv)) if os.path.dirname(args.out_csv) else None
        result_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved results to {args.out_csv}")


if __name__ == "__main__":
    main()


