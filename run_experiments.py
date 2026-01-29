#!/usr/bin/env python
"""Run All CarbonX Experiments"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run CarbonX experiments")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer samples")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--comparison-only", action="store_true", help="Only run comparisons")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CARBONX EXPERIMENT SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    if not args.comparison_only and not args.figures_only:
        print("\nPHASE 1: Running Benchmarks")
        print("-" * 40)
        
        from experiments.benchmark import BenchmarkRunner
        
        runner = BenchmarkRunner(output_dir=str(results_dir))
        summaries = runner.run_all_benchmarks()
        runner.print_summary(summaries)
        
        benchmark_path = runner.save_results(summaries, "benchmark_results.json")
        all_results["benchmarks"] = str(benchmark_path)
        
        print(f"Benchmark results saved to {benchmark_path}")
    
    if not args.benchmark_only and not args.figures_only:
        print("\nPHASE 2: Running Strategy Comparisons")
        print("-" * 40)
        
        from experiments.comparison import ComparisonExperiment
        
        num_requests = 50 if args.quick else 100
        num_runs = 3 if args.quick else 5
        
        experiment = ComparisonExperiment(output_dir=str(results_dir))
        comparison_results = experiment.run_all_comparisons(
            num_requests=num_requests,
            num_runs=num_runs
        )
        experiment.print_comparison_table(comparison_results)
        
        comparison_path = experiment.save_results(comparison_results, "comparison_results.json")
        all_results["comparisons"] = str(comparison_path)
        
        print(f"Comparison results saved to {comparison_path}")
    
    if not args.benchmark_only and not args.comparison_only:
        print("\nPHASE 3: Generating Figures")
        print("-" * 40)
        
        try:
            from experiments.visualize import generate_all_figures
            
            comparison_path = results_dir / "comparison_results.json"
            if comparison_path.exists():
                generate_all_figures(str(comparison_path))
            else:
                generate_all_figures()
            
            all_results["figures"] = "experiments/figures/"
            
        except ImportError as e:
            print(f"Could not generate figures: {e}")
            print("  Install matplotlib: pip install matplotlib")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput files:")
    for name, path in all_results.items():
        print(f"  - {name}: {path}")
    
    master_path = results_dir / "experiment_summary.json"
    with open(master_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "outputs": all_results,
        }, f, indent=2)
    
    print(f"\nMaster summary: {master_path}")
    print("\nAll experiments complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
