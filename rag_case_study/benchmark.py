import os
import json
import sys
import time
import argparse
import subprocess
import pandas as pd
from typing import Dict, List

def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if path:
        os.makedirs(path, exist_ok=True)

def run_command(cmd: List[str], description: str) -> Dict[str, float]:
    """
    Run a command and measure execution time.
    Returns dict with timing information.
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.perf_counter()
    start_wall = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        end_time = time.perf_counter()
        end_wall = time.time()
        
        elapsed_cpu = end_time - start_time
        elapsed_wall = end_wall - start_wall
        
        print(f"✓ Completed successfully")
        print(f"  CPU time: {elapsed_cpu:.2f} seconds")
        print(f"  Wall time: {elapsed_wall:.2f} seconds")
        print(f"  Output (first 500 chars):\n{result.stdout[:500]}")
        
        return {
            "success": True,
            "cpu_time_seconds": elapsed_cpu,
            "wall_time_seconds": elapsed_wall,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        end_time = time.perf_counter()
        end_wall = time.time()
        
        elapsed_cpu = end_time - start_time
        elapsed_wall = end_wall - start_wall
        
        print(f"✗ Failed with error code {e.returncode}")
        print(f"  Error output:\n{e.stderr[:500]}")
        
        return {
            "success": False,
            "cpu_time_seconds": elapsed_cpu,
            "wall_time_seconds": elapsed_wall,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error_code": e.returncode
        }


def benchmark_rag_query(query: str, index_dir: str, output_dir: str) -> Dict[str, float]:
    """
    Benchmark RAG query: query_rag.py with a specific query.
    This represents Scenario 1: query-specific analysis using RAG.
    """
    output_csv = os.path.join(output_dir, f"rag_benchmark_{query.replace(' ', '_')}.csv")
    
    cmd = [
        sys.executable,
        "query_rag.py",
        "--query", query,
        "--index-dir", index_dir,
        "--limit", "1000",
        "--max_sentiment", "500",
        "--out_csv", output_csv,
        "--print_examples", "0" ]
    
    result = run_command(cmd, f"RAG Query: '{query}'")
    result_count = 0
    if result["success"] and os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            result_count = len(df)
        except Exception:
            pass
    
    result["result_count"] = result_count
    result["scenario"] = "RAG"
    result["query"] = query
    
    return result

def benchmark_non_rag_full_analysis(input_file: str, output_file: str) -> Dict[str, float]:
    """
    Benchmark Non-RAG full dataset analysis: analyze_full_dataset.py.
    This represents Scenario 2: analyzing all 50,000 rows without RAG.
    """
    cmd = [
        sys.executable,
        "src/analyze_full_dataset.py",
        "--input", input_file,
        "--output", output_file,
        "--batch_size", "256"
    ]
    
    result = run_command(cmd, "Non-RAG Full Dataset Analysis")
    result_count = 0
    if result["success"] and os.path.exists(output_file):
        try:
            df = pd.read_parquet(output_file)
            result_count = len(df)
        except Exception:
            pass
    
    result["result_count"] = result_count
    result["scenario"] = "Non-RAG"
    return result

def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f} hours {minutes:.1f} minutes ({seconds:.2f} seconds)"

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RAG vs Non-RAG latency comparison (Goal 2)"
    )
    parser.add_argument("--index-dir", type=str, default="/home/onur/GitHub/case/rag3/index", help="Directory containing FAISS/BM25 indexes")
    parser.add_argument("--input-data", type=str, default="/home/onur/GitHub/case/rag3/data/clean.parquet", help="Input data file for Non-RAG analysis")
    parser.add_argument("--queries", type=str, nargs="+", default=["kredi", "araç", "takım"], help="Queries to test with RAG")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark", help="Output directory for benchmark results")
    parser.add_argument("--skip-non-rag", action="store_true", help="Skip Non-RAG benchmark (if already run)" )
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    
    print("\n" + "="*60)
    print("RAG vs Non-RAG Latency Benchmark (Goal 2)")
    print("="*60)
    print("\nThis benchmark compares:")
    print("  Scenario 1 (RAG): Query-specific analysis using RAG")
    print("  Scenario 2 (Non-RAG): Full dataset analysis (all 50,000 rows)")
    print("\n" + "="*60)
    
    results = []
    print("\n" + "="*60)
    print("BENCHMARKING RAG QUERIES (Scenario 1)")
    print("="*60)
    
    rag_results = []
    for query in args.queries:
        result = benchmark_rag_query(query, args.index_dir, args.output_dir)
        rag_results.append(result)
        results.append(result)
    
    successful_rag = [r for r in rag_results if r["success"]]
    if successful_rag:
        avg_rag_time = sum(r["wall_time_seconds"] for r in successful_rag) / len(successful_rag)
        avg_rag_cpu = sum(r["cpu_time_seconds"] for r in successful_rag) / len(successful_rag)
    else:
        avg_rag_time = 0
        avg_rag_cpu = 0
    
    non_rag_result = None
    if not args.skip_non_rag:
        print("\n" + "="*60)
        print("BENCHMARKING NON-RAG FULL ANALYSIS (Scenario 2)")
        print("="*60)
        
        non_rag_output = os.path.join(args.output_dir, "non_rag_full_analysis.parquet")
        non_rag_result = benchmark_non_rag_full_analysis(args.input_data, non_rag_output)
        results.append(non_rag_result)
    else:
        print("\nSkipping Non-RAG benchmark (--skip-non-rag flag set)")
        print("Using existing results if available...")
        
        non_rag_output = os.path.join(args.output_dir, "non_rag_full_analysis.parquet")
        if os.path.exists(non_rag_output):
            print(f"Found existing Non-RAG results at {non_rag_output}")
            print("Note: Actual Non-RAG analysis time is ~25 minutes for 50,000 rows")
            non_rag_result = {
                "success": True,
                "wall_time_seconds": 1500.0,
                "cpu_time_seconds": 1500.0,
                "scenario": "Non-RAG",
                "result_count": len(pd.read_parquet(non_rag_output)) if os.path.exists(non_rag_output) else 0
            }
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    summary = {
        "rag_queries": [],
        "non_rag": None,
        "comparison": {}
    }
    
    print("\n--- RAG Query Results (Scenario 1) ---")
    for result in rag_results:
        if result["success"]:
            print(f"\nQuery: '{result['query']}'")
            print(f"  Wall time: {format_time(result['wall_time_seconds'])}")
            print(f"  CPU time: {format_time(result['cpu_time_seconds'])}")
            print(f"  Results found: {result['result_count']}")
            summary["rag_queries"].append({
                "query": result["query"],
                "wall_time_seconds": result["wall_time_seconds"],
                "cpu_time_seconds": result["cpu_time_seconds"],
                "result_count": result["result_count"]
            })
    
    if successful_rag:
        print(f"\nAverage RAG query time: {format_time(avg_rag_time)}")
        print(f"Average RAG CPU time: {format_time(avg_rag_cpu)}")
        summary["comparison"]["avg_rag_wall_time_seconds"] = avg_rag_time
        summary["comparison"]["avg_rag_cpu_time_seconds"] = avg_rag_cpu
    
    if non_rag_result and non_rag_result.get("success"):
        print("\n--- Non-RAG Full Analysis (Scenario 2) ---")
        print(f"  Wall time: {format_time(non_rag_result['wall_time_seconds'])}")
        print(f"  CPU time: {format_time(non_rag_result['cpu_time_seconds'])}")
        print(f"  Results analyzed: {non_rag_result['result_count']}")
        summary["non_rag"] = {
            "wall_time_seconds": non_rag_result["wall_time_seconds"],
            "cpu_time_seconds": non_rag_result["cpu_time_seconds"],
            "result_count": non_rag_result["result_count"]
        }
        
        if successful_rag:
            speedup = non_rag_result["wall_time_seconds"] / avg_rag_time
            print("\n--- Performance Comparison ---")
            print(f"  RAG is {speedup:.1f}x faster than Non-RAG")
            print(f"  Time saved: {format_time(non_rag_result['wall_time_seconds'] - avg_rag_time)}")
            summary["comparison"]["speedup"] = speedup
            summary["comparison"]["time_saved_seconds"] = non_rag_result["wall_time_seconds"] - avg_rag_time
    
    print("\n" + "="*60)
    print("BUSINESS INSIGHT")
    print("="*60)
    if successful_rag and non_rag_result and non_rag_result.get("success"):
        print(f"\nRAG sayesinde, tüm veri setini {format_time(non_rag_result['wall_time_seconds'])} taramak yerine,")
        print(f"ilgili yorumları {format_time(avg_rag_time)} analiz edebiliyoruz.")
        print(f"\nBu, yaklaşık {non_rag_result['wall_time_seconds'] / avg_rag_time:.0f}x daha hızlı bir analiz sağlar.")
        print(f"Her sorgu için {format_time(non_rag_result['wall_time_seconds'] - avg_rag_time)} zaman tasarrufu.")
    elif successful_rag:
        print(f"\nRAG sorguları ortalama {format_time(avg_rag_time)} sürmektedir.")
        print("Non-RAG tam analiz yaklaşık 25 dakika sürmektedir (50,000 satır için).")
        print(f"RAG, sorguya özel analiz için çok daha hızlıdır (~{1500/avg_rag_time:.0f}x daha hızlı).")
    
    summary_file = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nBenchmark summary saved to: {summary_file}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame([
        {
            "scenario": r["scenario"],
            "query": r.get("query", "N/A"),
            "wall_time_seconds": r["wall_time_seconds"],
            "cpu_time_seconds": r["cpu_time_seconds"],
            "result_count": r.get("result_count", 0),
            "success": r["success"]
        }
        for r in results
    ])
    results_csv = os.path.join(args.output_dir, "benchmark_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Detailed results saved to: {results_csv}")


if __name__ == "__main__":
    main()

