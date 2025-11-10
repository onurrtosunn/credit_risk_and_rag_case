#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from typing import List, Tuple
import functools
try:
    from zeyrek import MorphAnalyzer
    _TR_LEMMA = MorphAnalyzer()
except Exception:
    _TR_LEMMA = None

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

@functools.lru_cache(maxsize=200_000)
def _lemma_tr(tok: str) -> str:
    if not _TR_LEMMA:
        return tok
    try:
        lem = _TR_LEMMA.lemmatize(tok)
        if isinstance(lem, list) and len(lem) > 0:
            cand = lem[0][0] if isinstance(lem[0], (list, tuple)) and len(lem[0]) > 0 else lem[0]
            return str(cand)
    except Exception:
        pass
    try:
        analyses = _TR_LEMMA.analyze(tok)
        if analyses:
            first = analyses[0]
            cand = getattr(first, "lemma", None)
            if cand:
                return str(cand)
    except Exception:
        pass
    return tok

def tr_tokenize(text: str) -> List[str]:
    # Lowercase, remove punctuation-like chars, split on whitespace, then lemmatize tokens
    text = text.lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    toks = text.split() if text else []
    return [_lemma_tr(t) for t in toks]

def build_bm25(corpus: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized_corpus = [tr_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def encode_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 128,
    use_e5_prefix: bool = True,
) -> np.ndarray:
    if use_e5_prefix:
        texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings

def build_faiss_hnsw(embeddings: np.ndarray, m: int = 32, ef_construction: int = 200) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.M = m
    index.verbose = False
    index.add(embeddings.astype(np.float32))
    return index

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS (HNSW, IP) and BM25 indexes from cleaned Parquet.")
    parser.add_argument("--input-parquet", type=str, default="/home/onur/GitHub/case/rag3/data/clean.parquet", help="Cleaned parquet path")
    parser.add_argument("--out-dir", type=str, default="/home/onur/GitHub/case/rag3/index", help="Directory to save indexes and metadata")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-e5-prefix", action="store_true", help="Disable E5 'passage:' prefix for embeddings")
    parser.add_argument("--save-embeddings", action="store_true", help="Also save raw embeddings .npy (optional)")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_parquet(args.input_parquet)
    required_cols = {"id", "score", "title", "feedback", "timestamp"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Parquet missing columns. Expected {required_cols}, found {set(df.columns)}")

    df = df.reset_index(drop=True)
    texts = df["feedback"].astype(str).tolist()

    print("Building BM25...")
    print(f"BM25 tokenization uses lemma: {'enabled' if _TR_LEMMA else 'disabled'}")
    bm25, tokenized_corpus = build_bm25(texts)
    with open(os.path.join(args.out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.out_dir, "bm25_tokens.pkl"), "wb") as f:
        pickle.dump(tokenized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print("Encoding embeddings...")
    embeddings = encode_embeddings(
        model=model,
        texts=texts,
        batch_size=args.batch_size,
        use_e5_prefix=not args.no_e5_prefix,
    )
    embeddings = l2_normalize(embeddings).astype(np.float32)

    if args.save_embeddings:
        np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)

    print("Building FAISS HNSW (IP) index...")
    index = build_faiss_hnsw(embeddings, m=32, ef_construction=200)
    index.hnsw.efSearch = 64

    faiss_path = os.path.join(args.out_dir, "faiss_hnsw_ip.index")
    print(f"Saving FAISS index to {faiss_path}")
    faiss.write_index(index, faiss_path)

    # Load pre-computed sentiment results if available
    precomputed_sentiment_path = "outputs/full_analysis_results.parquet"
    if os.path.exists(precomputed_sentiment_path):
        print(f"Loading pre-computed sentiment results from {precomputed_sentiment_path}...")
        try:
            precomputed = pd.read_parquet(precomputed_sentiment_path)
            # Merge sentiment columns if they exist
            if "sentiment_label" in precomputed.columns and "sentiment_score" in precomputed.columns:
                df = df.merge(
                    precomputed[["id", "sentiment_label", "sentiment_score"]],
                    on="id",
                    how="left"
                )
                print(f"Merged pre-computed sentiment results: {df['sentiment_label'].notna().sum()} rows have sentiment")
            else:
                print("Warning: Pre-computed file exists but missing sentiment columns")
        except Exception as e:
            print(f"Warning: Could not load pre-computed sentiment results: {e}")
    else:
        print(f"Pre-computed sentiment results not found at {precomputed_sentiment_path}")
        print("Run src/analyze_full_dataset.py first to enable sentiment pre-computation")
    
    meta_cols = ["id", "score", "title", "feedback", "timestamp"]
    if "sentiment_label" in df.columns:
        meta_cols.append("sentiment_label")
    if "sentiment_score" in df.columns:
        meta_cols.append("sentiment_score")
    meta_path = os.path.join(args.out_dir, "meta.parquet")
    df[meta_cols].to_parquet(meta_path, index=False)

    config = {
        "model_name": args.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "metric": "inner_product_l2_normalized",
        "hnsw_m": 32,
        "hnsw_efConstruction": 200,
        "hnsw_efSearch": 64,
        "count": int(len(df)),
        "bm25": {
            "tokenizer": "tr_tokenize_lemma",
        },
        "uses_e5_passage_prefix": not args.no_e5_prefix,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Index build complete:")
    print(f"- FAISS: {faiss_path}")
    print(f"- BM25:  {os.path.join(args.out_dir, 'bm25.pkl')}")
    print(f"- Meta:  {meta_path}")
    if args.save_embeddings:
        print(f"- Embeddings: {os.path.join(args.out_dir, 'embeddings.npy')}")

if __name__ == "__main__":
    main()


