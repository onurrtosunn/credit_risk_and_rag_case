#!/usr/bin/env python3
import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
    return text.strip()


def remove_urls(text: str) -> str:
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.sub(url_pattern, " ", text)


def remove_emails(text: str) -> str:
    email_pattern = r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b"
    return re.sub(email_pattern, " ", text)


def remove_phones(text: str) -> str:
    phone_pattern = r"\+?\d[\d\-\s]{7,}\d"
    return re.sub(phone_pattern, " ", text)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def basic_clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phones(text)
    text = normalize_whitespace(text)
    return text


def read_excel(input_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    # Normalize columns: expected [ID, Score, Title, Feedback, Timestamp]
    col_map = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in col_map:
                return col_map[n.lower()]
        return None

    id_col = pick("ID")
    score_col = pick("Score")
    title_col = pick("Title")
    feedback_col = pick("Feedback")
    ts_col = pick("Timestamp")

    missing = [name for name, val in [("ID", id_col), ("Score", score_col), ("Title", title_col), ("Feedback", feedback_col), ("Timestamp", ts_col)] if val is None]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}. Found columns: {list(df.columns)}")

    df = df[[id_col, score_col, title_col, feedback_col, ts_col]].rename(
        columns={
            id_col: "id",
            score_col: "score",
            title_col: "title",
            feedback_col: "feedback",
            ts_col: "timestamp",
        }
    )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic types
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["title"] = df["title"].astype(str)
    df["feedback"] = df["feedback"].astype(str)
    # Clean text columns
    tqdm.pandas(desc="cleaning_feedback")
    df["title"] = df["title"].progress_apply(basic_clean_text)
    df["feedback"] = df["feedback"].progress_apply(basic_clean_text)
    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, infer_datetime_format=True)
    # Drop bad rows
    df = df.dropna(subset=["id", "score", "feedback"])
    df = df[df["feedback"].str.len() >= 5]
    # Score bounds (1..5) if applicable
    df = df[(df["score"] >= 1) & (df["score"] <= 5)]
    # Deduplicate by id, keep latest timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values(["id", "timestamp"]).drop_duplicates(subset=["id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["id"], keep="last")
    # Sort final
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and clean customer feedback Excel, write Parquet/CSV.")
    parser.add_argument("--input", type=str, default="/home/onur/GitHub/case/rag3/musteriyorumlari.xlsx", help="Path to input Excel file")
    parser.add_argument("--out-dir", type=str, default="/home/onur/GitHub/case/rag3/data", help="Output directory for cleaned data")
    parser.add_argument("--out-name", type=str, default="clean", help="Base name for outputs (without extension)")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = read_excel(args.input)
    df = clean_dataframe(df)

    parquet_path = os.path.join(args.out_dir, f"{args.out_name}.parquet")
    csv_path = os.path.join(args.out_dir, f"{args.out_name}.csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"Saved cleaned data:\n- {parquet_path}\n- {csv_path}\nRows: {len(df)}")


if __name__ == "__main__":
    main()


