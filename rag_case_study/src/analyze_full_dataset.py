#!/usr/bin/env python3
"""
Pre-compute sentiment analysis for the entire dataset.
This script analyzes all feedback entries once and saves results for fast lookup in RAG queries.
"""
import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def load_sentiment_model(model_name: str, quantize: bool = True):
    """Load sentiment model and apply dynamic quantization for CPU speedup."""
    print(f"Loading sentiment model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Apply dynamic quantization for CPU speedup (2-4x faster)
    if quantize and not torch.cuda.is_available():
        print("Applying dynamic quantization for CPU speedup...")
        # Quantize only the linear layers (most compute-intensive)
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        print("Quantization applied successfully.")
    elif torch.cuda.is_available():
        print("GPU available, skipping quantization (GPU is faster).")
    
    return tokenizer, model


def infer_label_mapping(id2label: dict) -> dict:
    """Map label IDs to numeric sentiment values in [-1, 1]."""
    import re
    mapping_by_name = {
        "negative": -1.0, "neg": -1.0,
        "positive": 1.0, "pos": 1.0,
        "neutral": 0.0, "neu": 0.0
    }
    label_map = {}
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
        return {
            0: -1.0 if 0 not in label_map else label_map[0],
            1: 1.0 if 1 not in label_map else label_map[1]
        }
    if num_labels == 3:
        default_map = {0: -1.0, 1: 0.0, 2: 1.0}
        default_map.update(label_map)
        return default_map
    
    ordered = sorted(id2label.keys())
    values = np.linspace(-1.0, 1.0, num=len(ordered))
    return {i: float(v) for i, v in zip(ordered, values)}


@torch.inference_mode()
def sentiment_predict_batch(
    tokenizer, 
    model, 
    texts: List[str], 
    batch_size: int = 256,
    max_length: int = 256
):
    """Predict sentiment for a batch of texts."""
    if len(texts) == 0:
        return [], np.array([], dtype=np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        # For quantized models, keep on CPU
        model = model.to(device)
    else:
        model = model.to(device)
    
    id2label = (
        {i: l for i, l in enumerate(model.config.id2label.values())}
        if isinstance(model.config.id2label, dict)
        else {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
    )
    label_value_map = infer_label_mapping(id2label)
    
    all_scores = []
    all_labels = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        value_vec = np.array(
            [label_value_map[j] for j in range(probs.shape[1])],
            dtype=np.float32
        )
        numeric = (probs * value_vec[None, :]).sum(axis=1)
        all_scores.append(numeric)
        all_labels.extend([id2label[int(pid)] for pid in pred_ids])
    
    sentiment_numeric = (
        np.concatenate(all_scores, axis=0)
        if len(all_scores) > 0
        else np.array([], dtype=np.float32)
    )
    return all_labels, sentiment_numeric


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute sentiment analysis for entire dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/clean.parquet",
        help="Input file path (Excel, CSV, or Parquet)"
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Sheet1",
        help="Sheet name in Excel file (ignored for Parquet/CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/full_analysis_results.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--sentiment_model",
        type=str,
        default="savasy/bert-base-turkish-sentiment-cased",
        help="Sentiment analysis model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for sentiment analysis"
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable quantization (use if GPU available)"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    elif args.input.endswith('.xlsx') or args.input.endswith('.xls'):
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    elif args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        raise ValueError(f"Unsupported file format: {args.input}")
    
    print(f"Loaded {len(df)} rows")
    
    # Normalize column names (handle both uppercase and lowercase)
    feedback_col = None
    id_col = None
    
    # Find feedback column (case-insensitive)
    for col in df.columns:
        if col.lower() == "feedback":
            feedback_col = col
            break
    if feedback_col is None:
        raise ValueError(f"Dataset must contain 'Feedback' column. Found columns: {list(df.columns)}")
    
    # Find id column (case-insensitive)
    for col in df.columns:
        if col.lower() == "id":
            id_col = col
            break
    if id_col is None:
        raise ValueError(f"Dataset must contain 'id' column for ID matching. Found columns: {list(df.columns)}")
    
    # Normalize column names to lowercase for consistency
    df = df.rename(columns={feedback_col: "feedback", id_col: "id"})
    
    # Filter out empty feedback
    df = df[df["feedback"].notna() & (df["feedback"].astype(str).str.strip() != "")]
    print(f"After filtering empty feedback: {len(df)} rows")
    
    # Load sentiment model with quantization
    start_time = time.perf_counter()
    tokenizer, model = load_sentiment_model(
        args.sentiment_model,
        quantize=not args.no_quantize
    )
    model_load_time = time.perf_counter() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Perform sentiment analysis on all feedback
    print("\nStarting sentiment analysis on all feedback entries...")
    analysis_start = time.perf_counter()
    
    feedback_texts = df["feedback"].astype(str).tolist()
    labels, numeric_scores = sentiment_predict_batch(
        tokenizer,
        model,
        feedback_texts,
        batch_size=args.batch_size
    )
    
    analysis_end = time.perf_counter()
    analysis_time = analysis_end - analysis_start
    
    # Add results to DataFrame
    df["sentiment_label"] = labels
    df["sentiment_score"] = numeric_scores
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    df.to_parquet(args.output, index=False)
    print(f"Results saved successfully")
    
    # Print summary
    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total rows analyzed: {len(df)}")
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print(f"Sentiment analysis time: {analysis_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per row: {total_time/len(df)*1000:.2f} ms")
    print(f"\nFull dataset analysis (Non-RAG) completed in {total_time:.1f} seconds.")
    print(f"Results saved to: {args.output}")
    
    # Print sentiment distribution
    if len(labels) > 0:
        print("\nSentiment distribution:")
        sentiment_counts = pd.Series(labels).value_counts()
        for label, count in sentiment_counts.items():
            pct = count / len(labels) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        print(f"\nMean sentiment score: {numeric_scores.mean():.3f}")
        print(f"Std sentiment score: {numeric_scores.std():.3f}")


if __name__ == "__main__":
    main()

