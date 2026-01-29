"""CarbonX Comparison Experiments"""

import json
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger()


@dataclass
class ComparisonResult:
    """Result from comparison experiment."""
    
    strategy: str
    total_requests: int
    total_carbon_gco2: float
    avg_carbon_gco2: float
    total_latency_ms: float
    avg_latency_ms: float
    accuracy: float
    model_distribution: dict


class ComparisonExperiment:
    """Compare CarbonX against baselines."""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_strategy(
        self,
        strategy: Literal["carbonx", "always_large", "always_small", "random"],
        num_requests: int = 100,
        seed: int = 42,
    ) -> ComparisonResult:
        """Run a specific strategy."""
        random.seed(seed)
        
        from carbonx.simulator import DigitalTwin, WorkloadGenerator
        
        # Generate workload
        generator = WorkloadGenerator(seed=seed)
        requests = generator.generate_batch(num_requests)
        
        # Simulate based on strategy
        total_carbon = 0.0
        total_latency = 0.0
        correct = 0
        model_dist = {}
        
        for req in requests:
            if strategy == "carbonx":
                # Use adaptive selection based on expected_complexity
                complexity = req.expected_complexity
                if complexity == "low":
                    model = "small"
                    energy = 0.00001
                elif complexity == "medium":
                    model = "medium"
                    energy = 0.00003
                else:
                    model = "large"
                    energy = 0.00006
            elif strategy == "always_large":
                model = "large"
                energy = 0.00006
            elif strategy == "always_small":
                model = "small"
                energy = 0.00001
            else:  # random
                model = random.choice(["small", "medium", "large"])
                energy = {"small": 0.00001, "medium": 0.00003, "large": 0.00006}[model]
            
            # Calculate carbon (simulated)
            carbon = energy * 400  # gCO2/kWh
            latency = energy * 50000  # proportional latency
            
            total_carbon += carbon
            total_latency += latency
            model_dist[model] = model_dist.get(model, 0) + 1
            
            # Accuracy (larger models more accurate for complex queries)
            model_quality = {"small": 0.6, "medium": 0.8, "large": 0.95}[model]
            if random.random() < model_quality:
                correct += 1
        
        return ComparisonResult(
            strategy=strategy,
            total_requests=num_requests,
            total_carbon_gco2=total_carbon,
            avg_carbon_gco2=total_carbon / num_requests,
            total_latency_ms=total_latency,
            avg_latency_ms=total_latency / num_requests,
            accuracy=correct / num_requests,
            model_distribution=model_dist,
        )
    
    def run_all_comparisons(self, num_requests: int = 100, num_runs: int = 5) -> dict:
        """Run all strategies multiple times for statistical significance."""
        strategies = ["carbonx", "always_large", "always_small", "random"]
        
        all_results = {s: [] for s in strategies}
        
        for run in range(num_runs):
            seed = 42 + run
            log.info("comparison_run", run=run + 1, seed=seed)
            
            for strategy in strategies:
                result = self.run_strategy(strategy, num_requests, seed)
                all_results[strategy].append(result)
        
        # Aggregate results
        aggregated = {}
        for strategy, results in all_results.items():
            carbons = [r.total_carbon_gco2 for r in results]
            latencies = [r.avg_latency_ms for r in results]
            accuracies = [r.accuracy for r in results]
            
            aggregated[strategy] = {
                "carbon_mean": statistics.mean(carbons),
                "carbon_std": statistics.stdev(carbons) if len(carbons) > 1 else 0,
                "latency_mean": statistics.mean(latencies),
                "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "accuracy_mean": statistics.mean(accuracies),
                "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                "model_distribution": results[0].model_distribution,
            }
        
        return aggregated
    
    def save_results(self, results: dict, filename: str = None):
        """Save comparison results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        log.info("results_saved", path=str(output_path))
        return output_path
    
    def print_comparison_table(self, results: dict):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Strategy':<15} {'Carbon (gCO₂)':<20} {'Latency (ms)':<18} {'Accuracy':<12}")
        print("-" * 80)
        
        for strategy, data in results.items():
            carbon_str = f"{data['carbon_mean']:.4f} ± {data['carbon_std']:.4f}"
            latency_str = f"{data['latency_mean']:.1f} ± {data['latency_std']:.1f}"
            accuracy_str = f"{data['accuracy_mean']:.1%} ± {data['accuracy_std']:.1%}"
            print(f"{strategy:<15} {carbon_str:<20} {latency_str:<18} {accuracy_str:<12}")
        
        # Calculate reduction vs always_large
        if "carbonx" in results and "always_large" in results:
            reduction = (1 - results["carbonx"]["carbon_mean"] / results["always_large"]["carbon_mean"]) * 100
            print("-" * 80)
            print(f"CarbonX achieves {reduction:.1f}% carbon reduction vs always-large baseline")
        
        print("=" * 80)


def main():
    """Run comparison experiments."""
    print("CarbonX Comparison Experiments")
    print("=" * 50)
    
    experiment = ComparisonExperiment()
    
    # Run comparisons
    results = experiment.run_all_comparisons(num_requests=100, num_runs=5)
    
    # Print table
    experiment.print_comparison_table(results)
    
    # Save results
    output_path = experiment.save_results(results)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
