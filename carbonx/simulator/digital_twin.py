"""Digital Twin Simulator Module"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import random
import structlog

from carbonx.config import ModelSize
from carbonx.inference.complexity_estimator import ComplexityLevel
from carbonx.simulator.workload_generator import WorkloadGenerator, SyntheticRequest

logger = structlog.get_logger()


@dataclass
class SimulatedInference:
    """Result of a simulated inference."""
    request_id: str
    model_used: str
    tokens_generated: int
    latency_ms: float
    energy_kwh: float
    carbon_gco2: float
    complexity: str


@dataclass
class SimulationSummary:
    """Summary of a simulation run."""
    total_requests: int
    total_carbon_gco2: float
    total_energy_kwh: float
    avg_latency_ms: float
    model_distribution: dict[str, int]
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_carbon_gco2": self.total_carbon_gco2,
            "total_energy_kwh": self.total_energy_kwh,
            "avg_latency_ms": self.avg_latency_ms,
            "model_distribution": self.model_distribution,
        }


class DigitalTwin:
    """
    Digital Twin Simulator for CarbonX evaluation.
    
    Enables:
    - What-if analysis of policies
    - Comparison with baseline systems
    - Reproducible evaluation without hardware
    """
    
    # Model characteristics (energy per token in kWh, latency per token in ms)
    MODEL_SPECS = {
        ModelSize.SMALL: {"energy_per_token": 1e-8, "latency_per_token": 5, "quality": 0.7},
        ModelSize.MEDIUM: {"energy_per_token": 3e-8, "latency_per_token": 15, "quality": 0.85},
        ModelSize.LARGE: {"energy_per_token": 8e-8, "latency_per_token": 40, "quality": 1.0},
    }
    
    # Complexity to model mapping for CarbonX adaptive behavior
    COMPLEXITY_MODEL_MAP = {
        "low": ModelSize.SMALL,
        "medium": ModelSize.MEDIUM,
        "high": ModelSize.LARGE,
    }
    
    def __init__(
        self,
        carbon_intensity_range: tuple[float, float] = (200.0, 600.0),
        seed: Optional[int] = 42,
    ):
        """
        Initialize the digital twin.
        
        Args:
            carbon_intensity_range: Range of carbon intensities to simulate
            seed: Random seed for reproducibility
        """
        self.carbon_intensity_range = carbon_intensity_range
        self.workload_generator = WorkloadGenerator(seed=seed)
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(
            "digital_twin_initialized",
            carbon_range=carbon_intensity_range,
        )
    
    def simulate_baseline(
        self,
        requests: list[SyntheticRequest],
        model: ModelSize = ModelSize.LARGE,
    ) -> list[SimulatedInference]:
        """
        Simulate baseline inference (always uses large model).
        
        Args:
            requests: List of requests to simulate
            model: Model to use for all requests
            
        Returns:
            List of simulated inference results
        """
        results = []
        spec = self.MODEL_SPECS[model]
        
        for req in requests:
            carbon_intensity = random.uniform(*self.carbon_intensity_range)
            
            # Simulate inference
            tokens = self._estimate_output_tokens(req)
            latency = spec["latency_per_token"] * tokens
            energy = spec["energy_per_token"] * tokens
            carbon = energy * carbon_intensity
            
            results.append(SimulatedInference(
                request_id=req.request_id,
                model_used=model.value,
                tokens_generated=tokens,
                latency_ms=latency,
                energy_kwh=energy,
                carbon_gco2=carbon,
                complexity=req.expected_complexity,
            ))
        
        return results
    
    def simulate_carbonx(
        self,
        requests: list[SyntheticRequest],
        early_exit_enabled: bool = True,
        cache_hit_rate: float = 0.15,
    ) -> list[SimulatedInference]:
        """
        Simulate CarbonX adaptive inference.
        
        Args:
            requests: List of requests to simulate
            early_exit_enabled: Whether to simulate early exit
            cache_hit_rate: Simulated cache hit rate
            
        Returns:
            List of simulated inference results
        """
        results = []
        
        for req in requests:
            carbon_intensity = random.uniform(*self.carbon_intensity_range)
            
            # Simulate cache hit
            if random.random() < cache_hit_rate:
                results.append(SimulatedInference(
                    request_id=req.request_id,
                    model_used="cache",
                    tokens_generated=0,
                    latency_ms=1.0,
                    energy_kwh=0.0,
                    carbon_gco2=0.0,
                    complexity=req.expected_complexity,
                ))
                continue
            
            # Select model based on complexity (CarbonX adaptive behavior)
            model = self.COMPLEXITY_MODEL_MAP.get(
                req.expected_complexity,
                ModelSize.MEDIUM,
            )
            spec = self.MODEL_SPECS[model]
            
            # Estimate tokens
            tokens = self._estimate_output_tokens(req)
            
            # Base calculations
            latency = spec["latency_per_token"] * tokens
            energy = spec["energy_per_token"] * tokens
            
            # Apply early exit savings for simpler queries
            if early_exit_enabled and req.expected_complexity in ["low", "medium"]:
                exit_savings = random.uniform(0.1, 0.3)
                energy *= (1 - exit_savings)
                latency *= (1 - exit_savings * 0.5)  # Less latency savings
            
            carbon = energy * carbon_intensity
            
            results.append(SimulatedInference(
                request_id=req.request_id,
                model_used=model.value,
                tokens_generated=tokens,
                latency_ms=latency,
                energy_kwh=energy,
                carbon_gco2=carbon,
                complexity=req.expected_complexity,
            ))
        
        return results
    
    def _estimate_output_tokens(self, req: SyntheticRequest) -> int:
        """Estimate output tokens for a request."""
        base = req.max_tokens
        
        # Add some variance
        variance = random.uniform(0.5, 1.0)
        return int(base * variance)
    
    def summarize_results(self, results: list[SimulatedInference]) -> SimulationSummary:
        """Create a summary of simulation results."""
        if not results:
            return SimulationSummary(
                total_requests=0,
                total_carbon_gco2=0.0,
                total_energy_kwh=0.0,
                avg_latency_ms=0.0,
                model_distribution={},
            )
        
        total_carbon = sum(r.carbon_gco2 for r in results)
        total_energy = sum(r.energy_kwh for r in results)
        total_latency = sum(r.latency_ms for r in results)
        
        model_dist = {}
        for r in results:
            model_dist[r.model_used] = model_dist.get(r.model_used, 0) + 1
        
        return SimulationSummary(
            total_requests=len(results),
            total_carbon_gco2=total_carbon,
            total_energy_kwh=total_energy,
            avg_latency_ms=total_latency / len(results),
            model_distribution=model_dist,
        )
    
    def run_comparison(
        self,
        num_requests: int = 1000,
        complexity_distribution: Optional[dict[str, float]] = None,
    ) -> dict:
        """
        Run a comparison between baseline and CarbonX.
        
        Args:
            num_requests: Number of requests to simulate
            complexity_distribution: Optional complexity distribution
            
        Returns:
            Comparison results dict
        """
        # Update generator if distribution provided
        if complexity_distribution:
            self.workload_generator.complexity_dist = complexity_distribution
        
        # Generate workload
        requests = self.workload_generator.generate_batch(num_requests)
        
        # Run simulations
        baseline_results = self.simulate_baseline(requests)
        carbonx_results = self.simulate_carbonx(requests)
        
        # Summarize
        baseline_summary = self.summarize_results(baseline_results)
        carbonx_summary = self.summarize_results(carbonx_results)
        
        # Calculate reduction
        if baseline_summary.total_carbon_gco2 > 0:
            reduction = (
                (baseline_summary.total_carbon_gco2 - carbonx_summary.total_carbon_gco2)
                / baseline_summary.total_carbon_gco2
            ) * 100
        else:
            reduction = 0.0
        
        result = {
            "total_requests": num_requests,
            "baseline_carbon_gco2": baseline_summary.total_carbon_gco2,
            "carbonx_carbon_gco2": carbonx_summary.total_carbon_gco2,
            "carbon_reduction_percent": reduction,
            "baseline_energy_kwh": baseline_summary.total_energy_kwh,
            "carbonx_energy_kwh": carbonx_summary.total_energy_kwh,
            "avg_latency_ms": carbonx_summary.avg_latency_ms,
            "model_distribution": carbonx_summary.model_distribution,
            "workload_stats": self.workload_generator.get_workload_stats(requests),
        }
        
        logger.info(
            "simulation_complete",
            num_requests=num_requests,
            baseline_carbon=baseline_summary.total_carbon_gco2,
            carbonx_carbon=carbonx_summary.total_carbon_gco2,
            reduction_percent=reduction,
        )
        
        return result
    
    def what_if_analysis(
        self,
        num_requests: int = 1000,
        scenarios: Optional[dict[str, dict]] = None,
    ) -> dict[str, dict]:
        """
        Run what-if analysis with different scenarios.
        
        Args:
            num_requests: Requests per scenario
            scenarios: Dict of scenario_name -> config
            
        Returns:
            Dict of scenario_name -> results
        """
        default_scenarios = {
            "all_low_complexity": {"low": 1.0, "medium": 0.0, "high": 0.0},
            "all_high_complexity": {"low": 0.0, "medium": 0.0, "high": 1.0},
            "balanced": {"low": 0.33, "medium": 0.34, "high": 0.33},
            "realistic": {"low": 0.4, "medium": 0.4, "high": 0.2},
        }
        
        scenarios = scenarios or default_scenarios
        results = {}
        
        for name, dist in scenarios.items():
            results[name] = self.run_comparison(
                num_requests=num_requests,
                complexity_distribution=dist,
            )
        
        return results


# CLI support
if __name__ == "__main__":
    import sys
    import json
    
    simulator = DigitalTwin()
    
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    print(f"\nCarbonX Digital Twin Simulation")
    print(f"{'=' * 50}")
    print(f"Simulating {num_requests} requests...\n")
    
    result = simulator.run_comparison(num_requests)
    
    print(f"Results:")
    print(f"  Baseline Carbon: {result['baseline_carbon_gco2']:.4f} gCO2")
    print(f"  CarbonX Carbon:  {result['carbonx_carbon_gco2']:.4f} gCO2")
    print(f"  Reduction:       {result['carbon_reduction_percent']:.1f}%")
    print(f"\n  Model Distribution: {result['model_distribution']}")
    print(f"  Avg Latency:     {result['avg_latency_ms']:.1f} ms")
