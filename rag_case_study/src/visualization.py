#!/usr/bin/env python3
"""
Visualization script for generating business insights from full dataset analysis.
Generates charts for stakeholders: sentiment distribution, timeline trends, topic frequency.
"""
import argparse
import os
import re
from collections import Counter
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Turkish stopwords
STOP_TR = {
    "ve", "veya", "ile", "ama", "gibi", "çok", "daha", "mi", "mı", "mu", "mü", "de", "da", "ki", "bu", "şu", "o",
    "bir", "birçok", "her", "hiç", "olan", "oldu", "olması", "olarak", "için", "üzere", "ise", "yada", "yani",
    "neden", "niye", "çünkü", "fakat", "ancak", "ben", "biz", "siz", "onlar", "hepsi", "şey", "şeyi",
    "var", "yok", "yine", "az", "en", "olanlar", "ettim", "ettik", "ettiniz", "etti", "etmek", "hem",
    "şeklinde", "konu", "konuda", "durum", "durumda"
}


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if path:
        os.makedirs(path, exist_ok=True)


def tokenize_tr(text: str) -> List[str]:
    """Tokenize Turkish text."""
    text = str(text).lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    toks = [t for t in re.sub(r"\s+", " ", text).strip().split(" ") if t and t not in STOP_TR and len(t) >= 2]
    return toks


def extract_top_terms(texts: List[str], topk: int = 10) -> List[tuple]:
    """Extract top terms (unigrams + bigrams) from texts."""
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
    return items[:topk]


def plot_sentiment_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plot sentiment distribution as bar chart."""
    if "sentiment_label" not in df.columns:
        print("Warning: sentiment_label column not found, skipping sentiment distribution plot")
        return
    
    sentiment_counts = df["sentiment_label"].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    bar_colors = [colors.get(label.lower(), "#3498db") for label in sentiment_counts.index]
    
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Duygu Sınıfı (Sentiment Class)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Yorum Sayısı (Number of Reviews)', fontsize=12, fontweight='bold')
    ax.set_title('Genel Duygu Dağılımı (Overall Sentiment Distribution)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment distribution chart to {output_path}")


def plot_timeline_trends(df: pd.DataFrame, output_path: str) -> None:
    """Plot sentiment trends over time."""
    if "timestamp" not in df.columns or "sentiment_score" not in df.columns:
        print("Warning: timestamp or sentiment_score column not found, skipping timeline trends plot")
        return
    
    # Convert timestamp to datetime if needed
    df_timeline = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_timeline["timestamp"]):
        df_timeline["timestamp"] = pd.to_datetime(df_timeline["timestamp"], errors='coerce')
    
    # Remove rows with invalid timestamps
    df_timeline = df_timeline[df_timeline["timestamp"].notna()]
    
    if len(df_timeline) == 0:
        print("Warning: No valid timestamps found, skipping timeline trends plot")
        return
    
    # Group by date (daily aggregation)
    df_timeline["date"] = df_timeline["timestamp"].dt.date
    daily_stats = df_timeline.groupby("date").agg({
        "sentiment_score": ["mean", "count"]
    }).reset_index()
    daily_stats.columns = ["date", "mean_sentiment", "count"]
    daily_stats = daily_stats[daily_stats["count"] > 0]  # Filter days with no data
    
    if len(daily_stats) == 0:
        print("Warning: No daily statistics available, skipping timeline trends plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Mean sentiment over time
    ax1.plot(daily_stats["date"], daily_stats["mean_sentiment"], 
             marker='o', linewidth=2, markersize=4, color='#3498db', alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(daily_stats["date"], daily_stats["mean_sentiment"], 0, 
                     alpha=0.3, color='#3498db', where=(daily_stats["mean_sentiment"] >= 0))
    ax1.fill_between(daily_stats["date"], daily_stats["mean_sentiment"], 0, 
                     alpha=0.3, color='#e74c3c', where=(daily_stats["mean_sentiment"] < 0))
    ax1.set_ylabel('Ortalama Duygu Skoru\n(Mean Sentiment Score)', fontsize=12, fontweight='bold')
    ax1.set_title('Zamana Göre Şikayet/Memnuniyet Trendi (Timeline: Complaint/Satisfaction Trend)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([-1.1, 1.1])
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_stats) // 10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Review count over time
    ax2.bar(daily_stats["date"], daily_stats["count"], 
           color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Tarih (Date)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Yorum Sayısı\n(Number of Reviews)', fontsize=12, fontweight='bold')
    ax2.set_title('Günlük Yorum Sayısı (Daily Review Count)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_stats) // 10)))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline trends chart to {output_path}")


def plot_topic_frequency(df: pd.DataFrame, output_path: str, topk: int = 15) -> None:
    """Plot most complained about topics (from title column)."""
    if "title" not in df.columns:
        print("Warning: title column not found, skipping topic frequency plot")
        return
    
    # Filter negative sentiment reviews
    if "sentiment_label" in df.columns:
        negative_df = df[df["sentiment_label"].str.lower() == "negative"].copy()
    elif "sentiment_score" in df.columns:
        negative_df = df[df["sentiment_score"] < -0.2].copy()
    else:
        negative_df = df.copy()
    
    if len(negative_df) == 0:
        print("Warning: No negative reviews found, using all reviews for topic frequency")
        negative_df = df.copy()
    
    # Extract top terms from titles
    titles = negative_df["title"].astype(str).tolist()
    top_terms = extract_top_terms(titles, topk=topk)
    
    if len(top_terms) == 0:
        print("Warning: No terms extracted, skipping topic frequency plot")
        return
    
    terms, counts = zip(*top_terms)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Reds_r", len(terms))
    bars = ax.barh(range(len(terms)), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=11)
    ax.set_xlabel('Frekans (Frequency)', fontsize=12, fontweight='bold')
    ax.set_title('En Çok Şikayet Edilen Konular (Most Complained About Topics)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Top item at top
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved topic frequency chart to {output_path}")


def plot_sentiment_by_score(df: pd.DataFrame, output_path: str) -> None:
    """Plot sentiment distribution by score (1-5)."""
    if "score" not in df.columns or "sentiment_label" not in df.columns:
        print("Warning: score or sentiment_label column not found, skipping sentiment by score plot")
        return
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df["score"], df["sentiment_label"], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    crosstab.plot(kind='bar', stacked=True, ax=ax, 
                  color={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Müşteri Puanı (Customer Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Yüzde (%)', fontsize=12, fontweight='bold')
    ax.set_title('Puanlara Göre Duygu Dağılımı (Sentiment Distribution by Score)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Duygu (Sentiment)', title_fontsize=11, fontsize=10, loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment by score chart to {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str, top_topics: int = 20) -> None:
    """
    Plot correlation heatmap: Title (Topic) x Score (1-5) x sentiment_score.
    Shows which topics produce strong negative/positive sentiment for each score.
    """
    if "title" not in df.columns or "score" not in df.columns or "sentiment_score" not in df.columns:
        print("Warning: title, score, or sentiment_score column not found, skipping correlation heatmap")
        return
    
    # Filter out rows with missing data
    df_clean = df[df["title"].notna() & df["score"].notna() & df["sentiment_score"].notna()].copy()
    
    if len(df_clean) == 0:
        print("Warning: No valid data for correlation heatmap")
        return
    
    # Get top topics by frequency
    topic_counts = df_clean["title"].value_counts()
    top_topics_list = topic_counts.head(top_topics).index.tolist()
    df_filtered = df_clean[df_clean["title"].isin(top_topics_list)].copy()
    
    if len(df_filtered) == 0:
        print("Warning: No data after filtering top topics")
        return
    
    # Create pivot table: Title x Score -> mean sentiment_score
    pivot = df_filtered.pivot_table(
        index="title",
        columns="score",
        values="sentiment_score",
        aggfunc="mean"
    )
    
    # Sort by average sentiment (most negative first)
    pivot["avg_sentiment"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg_sentiment", ascending=True)
    pivot = pivot.drop(columns=["avg_sentiment"])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(10, len(pivot) * 0.5)))
    
    # Use diverging colormap: red (negative) to green (positive)
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Ortalama Duygu Skoru\n(Mean Sentiment Score)", "shrink": 0.8},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('Müşteri Puanı (Customer Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Konu Başlığı (Topic Title)', fontsize=12, fontweight='bold')
    ax.set_title('Konu x Puan Duygu Korelasyonu (Topic x Score Sentiment Correlation)\n'
                'Kırmızı=Negatif, Yeşil=Pozitif (Red=Negative, Green=Positive)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")


def detect_hidden_risks_and_strengths(df: pd.DataFrame) -> Dict:
    """
    Detect hidden risks and strengths:
    - Hidden Risk: Score 4-5 but sentiment negative (customer is polite but content is negative)
    - Hidden Strength: Score 1-2 but sentiment positive (customer is angry but mentions something positive)
    """
    if "score" not in df.columns or "sentiment_label" not in df.columns:
        return {"hidden_risks": [], "hidden_strengths": []}
    
    df_clean = df[df["score"].notna() & df["sentiment_label"].notna()].copy()
    
    # Convert score to numeric if needed
    if df_clean["score"].dtype == 'object':
        df_clean["score"] = pd.to_numeric(df_clean["score"], errors='coerce')
    
    # Hidden Risks: Score 4-5 but sentiment negative
    hidden_risks = df_clean[
        (df_clean["score"] >= 4) & 
        (df_clean["sentiment_label"].str.lower() == "negative")
    ].copy()
    
    # Hidden Strengths: Score 1-2 but sentiment positive
    hidden_strengths = df_clean[
        (df_clean["score"] <= 2) & 
        (df_clean["sentiment_label"].str.lower() == "positive")
    ].copy()
    
    return {
        "hidden_risks": hidden_risks,
        "hidden_strengths": hidden_strengths,
        "risk_count": len(hidden_risks),
        "strength_count": len(hidden_strengths)
    }


def plot_hidden_risks_and_strengths(df: pd.DataFrame, output_path: str, topk: int = 10) -> None:
    """Plot hidden risks and strengths analysis."""
    detection = detect_hidden_risks_and_strengths(df)
    
    hidden_risks = detection["hidden_risks"]
    hidden_strengths = detection["hidden_strengths"]
    
    if len(hidden_risks) == 0 and len(hidden_strengths) == 0:
        print("Warning: No hidden risks or strengths found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Hidden Risks by Topic
    if len(hidden_risks) > 0 and "title" in hidden_risks.columns:
        risk_topics = hidden_risks["title"].value_counts().head(topk)
        if len(risk_topics) > 0:
            ax1.barh(range(len(risk_topics)), risk_topics.values, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
            ax1.set_yticks(range(len(risk_topics)))
            ax1.set_yticklabels(risk_topics.index, fontsize=10)
            ax1.set_xlabel('Gizli Risk Sayısı (Hidden Risk Count)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Gizli Riskler (Hidden Risks)\nScore 4-5 ama Sentiment Negatif\n(Total: {len(hidden_risks)})', 
                         fontsize=13, fontweight='bold', pad=15)
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            ax1.invert_yaxis()
            # Add value labels
            for i, v in enumerate(risk_topics.values):
                ax1.text(v, i, f' {v}', va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Hidden Strengths by Topic
    if len(hidden_strengths) > 0 and "title" in hidden_strengths.columns:
        strength_topics = hidden_strengths["title"].value_counts().head(topk)
        if len(strength_topics) > 0:
            ax2.barh(range(len(strength_topics)), strength_topics.values, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
            ax2.set_yticks(range(len(strength_topics)))
            ax2.set_yticklabels(strength_topics.index, fontsize=10)
            ax2.set_xlabel('Gizli Güçlü Yan Sayısı (Hidden Strength Count)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Gizli Güçlü Yanlar (Hidden Strengths)\nScore 1-2 ama Sentiment Pozitif\n(Total: {len(hidden_strengths)})', 
                         fontsize=13, fontweight='bold', pad=15)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            ax2.invert_yaxis()
            # Add value labels
            for i, v in enumerate(strength_topics.values):
                ax2.text(v, i, f' {v}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hidden risks and strengths chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts from full dataset analysis"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/full_analysis_results.parquet",
        help="Input parquet file with full analysis results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/charts",
        help="Output directory for charts"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Number of top topics to show in topic frequency chart"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Load data
    print(f"Loading data from {args.input}...")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}\n"
                              f"Please run src/analyze_full_dataset.py first to generate the analysis results.")
    
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")
    
    # Generate charts
    print("\nGenerating charts...")
    
    # 1. Sentiment distribution
    plot_sentiment_distribution(
        df, 
        os.path.join(args.output_dir, "sentiment_distribution.png")
    )
    
    # 2. Timeline trends
    plot_timeline_trends(
        df, 
        os.path.join(args.output_dir, "timeline_trends.png")
    )
    
    # 3. Topic frequency (most complained about topics)
    plot_topic_frequency(
        df, 
        os.path.join(args.output_dir, "topic_frequency.png"),
        topk=args.topk
    )
    
    # 4. Sentiment by score
    plot_sentiment_by_score(
        df,
        os.path.join(args.output_dir, "sentiment_by_score.png")
    )
    
    # 5. Correlation heatmap (Title x Score x sentiment_score)
    plot_correlation_heatmap(
        df,
        os.path.join(args.output_dir, "correlation_heatmap.png"),
        top_topics=20
    )
    
    # 6. Hidden risks and strengths
    plot_hidden_risks_and_strengths(
        df,
        os.path.join(args.output_dir, "hidden_risks_strengths.png"),
        topk=10
    )
    
    # Print hidden risks and strengths summary
    detection = detect_hidden_risks_and_strengths(df)
    if detection["risk_count"] > 0 or detection["strength_count"] > 0:
        print(f"\nGizli Riskler ve Güçlü Yanlar (Hidden Risks & Strengths):")
        print(f"- Gizli Risk Sayısı (Score 4-5 ama Sentiment Negatif): {detection['risk_count']}")
        print(f"- Gizli Güçlü Yan Sayısı (Score 1-2 ama Sentiment Pozitif): {detection['strength_count']}")
    
    print(f"\nAll charts saved to {args.output_dir}/")
    print("Charts generated successfully!")


if __name__ == "__main__":
    main()

