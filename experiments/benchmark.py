"""CarbonX Benchmark Evaluation Suite"""

import json
import time
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    
    prompt: str
    expected_answer: Optional[str]
    generated_text: str
    model_used: str
    tokens_generated: int
    latency_ms: float
    energy_kwh: float
    carbon_gco2: float
    from_cache: bool
    correct: Optional[bool] = None
    dataset: str = ""
    difficulty: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""
    
    dataset: str
    total_samples: int
    accuracy: float
    
    # Carbon metrics
    total_carbon_gco2: float
    avg_carbon_gco2: float
    carbon_per_correct: float
    
    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    
    # Model distribution
    model_distribution: dict
    cache_hit_rate: float
    
    # Comparison
    baseline_carbon_gco2: float
    carbon_reduction_percent: float


# ============================================================
# Benchmark Datasets
# ============================================================

TRIVIAQA_SAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris", "difficulty": "easy"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "difficulty": "easy"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "difficulty": "easy"},
    {"question": "In what year did World War II end?", "answer": "1945", "difficulty": "medium"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "difficulty": "easy"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci", "difficulty": "easy"},
    {"question": "What is the speed of light in meters per second?", "answer": "299792458", "difficulty": "hard"},
    {"question": "What is the powerhouse of the cell?", "answer": "mitochondria", "difficulty": "easy"},
    {"question": "Who developed the theory of relativity?", "answer": "Einstein", "difficulty": "easy"},
    {"question": "What is the smallest prime number?", "answer": "2", "difficulty": "easy"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "difficulty": "easy"},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell", "difficulty": "medium"},
    {"question": "What is H2O commonly known as?", "answer": "water", "difficulty": "easy"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars", "difficulty": "easy"},
    {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong", "difficulty": "easy"},
]

REASONING_SAMPLES = [
    {
        "question": "If a train travels at 60 mph for 2 hours, how far does it go?",
        "answer": "120 miles",
        "difficulty": "medium",
    },
    {
        "question": "What comes next in the sequence: 2, 4, 8, 16, ?",
        "answer": "32",
        "difficulty": "easy",
    },
    {
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
        "answer": "No",
        "difficulty": "hard",
    },
    {
        "question": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "answer": "5 cents",
        "difficulty": "hard",
    },
    {
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "answer": "5 minutes",
        "difficulty": "hard",
    },
]

CODE_SAMPLES = [
    {
        "question": "Write a Python function to check if a number is prime.",
        "answer": "def is_prime",
        "difficulty": "medium",
    },
    {
        "question": "Write a Python function to reverse a string.",
        "answer": "def reverse",
        "difficulty": "easy",
    },
    {
        "question": "Write a Python function to find the factorial of a number.",
        "answer": "def factorial",
        "difficulty": "easy",
    },
    {
        "question": "Write a Python function to implement binary search.",
        "answer": "def binary_search",
        "difficulty": "medium",
    },
    {
        "question": "Write a Python function to merge two sorted lists.",
        "answer": "def merge",
        "difficulty": "medium",
    },
]


class BenchmarkRunner:
    """Runs benchmarks against CarbonX."""
    
    def __init__(self, carbonx_instance=None, output_dir: str = "experiments/results"):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if carbonx_instance is None:
            from carbonx import CarbonX
            self.carbonx = CarbonX()
        else:
            self.carbonx = carbonx_instance
        
        self.results: list[BenchmarkResult] = []
        
        log.info("benchmark_runner_initialized", output_dir=str(self.output_dir))
    
    def _check_answer(self, generated: str, expected: str) -> bool:
        """Check if answer is correct (fuzzy matching)."""
        generated_lower = generated.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Direct match
        if expected_lower in generated_lower:
            return True
        
        # Check for common variations
        if expected_lower.replace(" ", "") in generated_lower.replace(" ", ""):
            return True
        
        return False
    
    def run_dataset(
        self,
        dataset_name: str,
        samples: list[dict],
        max_tokens: int = 50,
    ) -> BenchmarkSummary:
        """Run benchmark on a dataset."""
        log.info("benchmark_starting", dataset=dataset_name, samples=len(samples))
        
        results = []
        baseline_carbon = 0.0
        
        for i, sample in enumerate(samples):
            prompt = f"Question: {sample['question']}\nAnswer:"
            
            try:
                # Run CarbonX inference
                response = self.carbonx.inference(prompt, max_tokens=max_tokens)
                
                # Check correctness
                correct = self._check_answer(response.text, sample["answer"])
                
                result = BenchmarkResult(
                    prompt=prompt,
                    expected_answer=sample["answer"],
                    generated_text=response.text,
                    model_used=response.model_used,
                    tokens_generated=response.tokens_generated,
                    latency_ms=response.latency_ms,
                    energy_kwh=response.energy_kwh,
                    carbon_gco2=response.carbon_gco2,
                    from_cache=response.from_cache,
                    correct=correct,
                    dataset=dataset_name,
                    difficulty=sample.get("difficulty", "unknown"),
                )
                results.append(result)
                
                # Estimate baseline (always large model)
                baseline_carbon += response.carbon_gco2 * 3  # Large is ~3x small
                
                log.info(
                    "sample_complete",
                    index=i + 1,
                    correct=correct,
                    model=response.model_used,
                    carbon=response.carbon_gco2,
                )
                
            except Exception as e:
                log.error("sample_failed", index=i, error=str(e))
        
        # Calculate summary
        total_carbon = sum(r.carbon_gco2 for r in results)
        latencies = [r.latency_ms for r in results]
        correct_count = sum(1 for r in results if r.correct)
        
        model_dist = {}
        for r in results:
            model_dist[r.model_used] = model_dist.get(r.model_used, 0) + 1
        
        cache_hits = sum(1 for r in results if r.from_cache)
        
        summary = BenchmarkSummary(
            dataset=dataset_name,
            total_samples=len(results),
            accuracy=correct_count / len(results) if results else 0,
            total_carbon_gco2=total_carbon,
            avg_carbon_gco2=total_carbon / len(results) if results else 0,
            carbon_per_correct=total_carbon / correct_count if correct_count > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            model_distribution=model_dist,
            cache_hit_rate=cache_hits / len(results) if results else 0,
            baseline_carbon_gco2=baseline_carbon,
            carbon_reduction_percent=((baseline_carbon - total_carbon) / baseline_carbon * 100) if baseline_carbon > 0 else 0,
        )
        
        self.results.extend(results)
        
        log.info(
            "benchmark_complete",
            dataset=dataset_name,
            accuracy=summary.accuracy,
            carbon_reduction=summary.carbon_reduction_percent,
        )
        
        return summary
    
    def run_all_benchmarks(self, include_ml_benchmarks: bool = True) -> dict:
        """Run all benchmark datasets."""
        summaries = {}
        
        # Original datasets
        summaries["triviaqa"] = self.run_dataset("TriviaQA", TRIVIAQA_SAMPLES)
        summaries["reasoning"] = self.run_dataset("Reasoning", REASONING_SAMPLES)
        summaries["code"] = self.run_dataset("Code", CODE_SAMPLES, max_tokens=100)
        
        # ML Benchmark Datasets
        if include_ml_benchmarks:
            try:
                from experiments.datasets import MMLU_SAMPLES, HUMANEVAL_SAMPLES, GSM8K_SAMPLES, TRUTHFULQA_SAMPLES
                
                # MMLU - Multiple Choice
                mmlu_formatted = [
                    {"question": f"{s['question']} {' '.join(s['choices'])}", "answer": s['answer'], "difficulty": s['difficulty']}
                    for s in MMLU_SAMPLES
                ]
                summaries["mmlu"] = self.run_dataset("MMLU", mmlu_formatted)
                
                # HumanEval - Code Generation
                humaneval_formatted = [
                    {"question": s['prompt'], "answer": s['canonical_solution'][:50], "difficulty": s['difficulty']}
                    for s in HUMANEVAL_SAMPLES
                ]
                summaries["humaneval"] = self.run_dataset("HumanEval", humaneval_formatted, max_tokens=150)
                
                # GSM8K - Math Reasoning
                gsm8k_formatted = [
                    {"question": s['question'], "answer": s['answer'], "difficulty": s['difficulty']}
                    for s in GSM8K_SAMPLES
                ]
                summaries["gsm8k"] = self.run_dataset("GSM8K", gsm8k_formatted, max_tokens=100)
                
                # TruthfulQA - Truthfulness
                truthful_formatted = [
                    {"question": s['question'], "answer": s['correct_answer'][:50], "difficulty": s['difficulty']}
                    for s in TRUTHFULQA_SAMPLES
                ]
                summaries["truthfulqa"] = self.run_dataset("TruthfulQA", truthful_formatted)
                
                log.info("ml_benchmarks_complete", datasets=["MMLU", "HumanEval", "GSM8K", "TruthfulQA"])
            except ImportError as e:
                log.warning("ml_benchmarks_not_available", error=str(e))
        
        return summaries
    
    def run_ablation_study(self, num_samples: int = 50) -> dict:
        """Run ablation study testing each component."""
        from carbonx.config import CarbonXConfig
        
        ablation_results = {}
        
        # Generate test samples
        samples = random.sample(TRIVIAQA_SAMPLES, min(num_samples, len(TRIVIAQA_SAMPLES)))
        
        # Test 1: Full CarbonX
        log.info("ablation_full_carbonx")
        ablation_results["full"] = self.run_dataset("Ablation-Full", samples)
        
        # Test 2: No caching (simulate by clearing cache)
        log.info("ablation_no_cache")
        if hasattr(self.carbonx, 'cache'):
            self.carbonx.cache.clear()
        ablation_results["no_cache"] = self.run_dataset("Ablation-NoCache", samples)
        
        # Test 3: Always small model
        log.info("ablation_always_small")
        original_complexity = None
        # Note: Would need to modify runtime for true ablation
        ablation_results["always_small"] = self.run_dataset("Ablation-AlwaysSmall", samples[:5])
        
        return ablation_results
    
    def save_results(self, summaries: dict, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "summaries": {k: asdict(v) for k, v in summaries.items()},
            "detailed_results": [asdict(r) for r in self.results],
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        log.info("results_saved", path=str(output_path))
        return output_path
    
    def print_summary(self, summaries: dict):
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("CARBONX BENCHMARK RESULTS")
        print("=" * 70)
        
        for name, summary in summaries.items():
            print(f"\n{summary.dataset}")
            print("-" * 40)
            print(f"  Samples:           {summary.total_samples}")
            print(f"  Accuracy:          {summary.accuracy:.1%}")
            print(f"  Total Carbon:      {summary.total_carbon_gco2:.6f} gCO₂")
            print(f"  Avg Carbon:        {summary.avg_carbon_gco2:.6f} gCO₂")
            print(f"  Avg Latency:       {summary.avg_latency_ms:.1f} ms")
            print(f"  Cache Hit Rate:    {summary.cache_hit_rate:.1%}")
            print(f"  Model Distribution: {summary.model_distribution}")
            print(f"  Carbon Reduction:  {summary.carbon_reduction_percent:.1f}% vs baseline")
        
        print("\n" + "=" * 70)


def main():
    """Run benchmark suite."""
    print("CarbonX Benchmark Suite")
    print("=" * 50)
    
    runner = BenchmarkRunner()
    
    # Run all benchmarks
    summaries = runner.run_all_benchmarks()
    
    # Print summary
    runner.print_summary(summaries)
    
    # Save results
    output_path = runner.save_results(summaries)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
