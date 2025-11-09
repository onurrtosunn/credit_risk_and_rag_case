# Data Scientist Case Studies

This repository contains two case studies:

1. **[Credit Risk Analysis](./credit_risk_analysis_case_study/)** - Credit risk prediction models
2. **[RAG Customer Review Analysis](./rag_case_study/)** - RAG-based and Non-RAG sentiment analysis for customer reviews

## 📁 Project Structure

```
credit_risk_and_rag_case/
├── README.md                          # This file
├── Data_Scientist_Case.pdf            # Case study document
│
├── credit_risk_analysis_case_study/   # Task 1: Credit Risk Analysis
│   ├── README.md                      # Detailed documentation
│   ├── requirements.txt               # Dependencies
│   ├── config.py                      # Configuration parameters
│   ├── data_cleaning.py               # Data cleaning module
│   ├── feature_engineering.py         # Feature engineering module
│   ├── eda.py                         # Exploratory data analysis
│   ├── train.py                       # Model training module
│   ├── testing.py                     # Model evaluation module
│   ├── main.ipynb                     # Main working notebook
│   ├── feature_engineering_test.ipynb # Feature engineering test notebook
│   ├── data/                          # Data files
│   │   └── credit_risk_case.xlsx
│   ├── models/                        # Trained models
│   │   ├── pd_logistic_pipeline.pkl
│   │   ├── pd_gbm_pipeline.pkl
│   │   ├── pd_knn_pipeline.pkl
│   │   ├── pd_rf_pipeline.pkl
│   │   └── pd_xgb_pipeline.pkl
│   ├── plots/                         # Visualizations
│   └── reports/                       # Evaluation results
│       ├── model_comparison_summary.csv
│       ├── model_cv_results.csv
│       └── model_test_results.csv
│
└── rag_case_study/                    # Task 2: RAG Customer Review Analysis
    ├── README.md                      # Detailed documentation
    ├── requirements.txt               # Dependencies
    ├── ingest_clean.py                # Data ingestion and cleaning
    ├── build_index.py                 # Index building (BM25 + FAISS)
    ├── query_rag.py                   # RAG query pipeline
    ├── query_baseline.py              # Baseline (BM25-only) query
    ├── evaluate.py                    # RAG vs Baseline evaluation
    ├── benchmark.py                   # RAG vs Non-RAG benchmark
    ├── musteriyorumlari.xlsx          # Raw data file
    ├── data/                          # Cleaned data files
    │   ├── clean.csv
    │   └── clean.parquet
    ├── index/                         # Generated indexes (BM25 + FAISS)
    │   ├── bm25.pkl
    │   ├── bm25_tokens.pkl
    │   ├── faiss_hnsw_ip.index
    │   ├── meta.parquet
    │   └── config.json
    ├── eval/                          # Evaluation results and charts
    │   ├── baseline_kredi.csv
    │   ├── rag_kredi.csv
    │   ├── rag_takım.csv
    │   ├── rag_zaman.csv
    │   ├── summary_metrics.csv
    │   └── charts/                    # Visualization charts
    └── src/                           # Source code modules
        ├── analyze_full_dataset.py    # Full dataset analysis
        ├── visualization.py          # Visualization generation
        └── utils.py                   # Utility functions
```

## 🚀 Quick Start

### Installation

```bash
pip install -r credit_risk_analysis_case_study/requirements.txt
pip install -r rag_case_study/requirements.txt
```

### Project 1: Credit Risk Analysis

```bash
cd credit_risk_analysis_case_study
jupyter notebook main.ipynb
```

**Details**: [README](./credit_risk_analysis_case_study/README.md)

### Project 2: RAG Customer Review Analysis

```bash
cd rag_case_study
python ingest_clean.py --input musteriyorumlari.xlsx --out-dir data
python build_index.py --input-parquet data/clean.parquet --out-dir index
python query_rag.py --query "kredi" --index-dir index --limit 1000
```

**Details**: [README](./rag_case_study/README.md)
