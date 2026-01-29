"""
Adaptive Runtime Module

Implements dynamic model selection based on request complexity,
carbon budget, and latency constraints.

Formula: m* = argmin C(m, r) subject to Latency(m, r) ≤ Lᵣ
"""

from dataclasses import dataclass
from typing import Optional
import time
import structlog

from carbonx.config import CarbonXConfig, ModelSize
from carbonx.inference.model_pool import ModelPool, InferenceResult
from carbonx.inference.complexity_estimator import (
    ComplexityEstimator,
    ComplexityEstimate,
    ComplexityLevel,
)
from carbonx.inference.early_exit import EarlyExitWrapper, EarlyExitResult
from carbonx.budget_manager import CarbonBudgetManager, BudgetStatus

logger = structlog.get_logger()


@dataclass
class AdaptiveInferenceResult:
    """Result of adaptive inference with full metadata."""
    text: str
    tokens_generated: int
    input_tokens: int
    model_used: str
    model_selected_by: str  # "complexity", "budget", "latency", "default"
    complexity: ComplexityEstimate
    latency_ms: float
    energy_kwh: float
    carbon_gco2: float
    early_exit_layer: Optional[int] = None
    computation_saved_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "input_tokens": self.input_tokens,
            "model_used": self.model_used,
            "model_selected_by": self.model_selected_by,
            "complexity": self.complexity.to_dict(),
            "latency_ms": self.latency_ms,
            "energy_kwh": self.energy_kwh,
            "carbon_gco2": self.carbon_gco2,
            "early_exit_layer": self.early_exit_layer,
            "computation_saved_ratio": self.computation_saved_ratio,
        }


class AdaptiveRuntime:
    """
    Adaptive inference runtime that dynamically selects models
    based on complexity, budget, and constraints.
    
    Implements the optimization:
        m* = argmin C(m, r) subject to Latency(m, r) ≤ Lᵣ
    """
    
    # Mapping from complexity level to preferred model size
    COMPLEXITY_MODEL_MAP = {
        ComplexityLevel.LOW: ModelSize.SMALL,
        ComplexityLevel.MEDIUM: ModelSize.MEDIUM,
        ComplexityLevel.HIGH: ModelSize.LARGE,
    }
    
    def __init__(
        self,
        config: Optional[CarbonXConfig] = None,
        model_pool: Optional[ModelPool] = None,
        complexity_estimator: Optional[ComplexityEstimator] = None,
        early_exit_wrapper: Optional[EarlyExitWrapper] = None,
        budget_manager: Optional[CarbonBudgetManager] = None,
        carbon_intensity_gco2_kwh: float = 400.0,
    ):
        """
        Initialize the adaptive runtime.
        
        Args:
            config: CarbonX configuration
            model_pool: Model pool instance
            complexity_estimator: Complexity estimator instance
            early_exit_wrapper: Early exit wrapper instance
            budget_manager: Carbon budget manager instance
            carbon_intensity_gco2_kwh: Current carbon intensity
        """
        self.config = config or CarbonXConfig()
        self.model_pool = model_pool or ModelPool()
        self.complexity_estimator = complexity_estimator or ComplexityEstimator()
        self.early_exit = early_exit_wrapper or EarlyExitWrapper(
            confidence_threshold=self.config.early_exit.confidence_threshold,
            enabled=self.config.early_exit.enabled,
        )
        self.budget_manager = budget_manager
        self.carbon_intensity = carbon_intensity_gco2_kwh
    
    def set_carbon_intensity(self, intensity_gco2_kwh: float) -> None:
        """Update the current carbon intensity."""
        self.carbon_intensity = intensity_gco2_kwh
        logger.info("carbon_intensity_updated", intensity=intensity_gco2_kwh)
    
    def select_model(
        self,
        complexity: ComplexityEstimate,
        tenant_id: Optional[str] = None,
        latency_constraint_ms: Optional[float] = None,
    ) -> tuple[str, str]:
        """
        Select the optimal model based on constraints.
        
        Args:
            complexity: Estimated request complexity
            tenant_id: Optional tenant ID for budget checking
            latency_constraint_ms: Optional latency constraint
            
        Returns:
            Tuple of (model_name, selection_reason)
        """
        # Start with complexity-based selection
        preferred_size = self.COMPLEXITY_MODEL_MAP[complexity.level]
        preferred_name = preferred_size.value
        selection_reason = "complexity"
        
        # Check budget constraints if budget manager available
        if self.budget_manager and tenant_id:
            # Estimate emissions for each model
            estimated_tokens = complexity.estimated_tokens
            emissions = {}
            
            for model_name in self.model_pool.list_models():
                energy = self.model_pool.estimate_energy(model_name, estimated_tokens)
                carbon = energy * self.carbon_intensity
                emissions[model_name] = carbon
            
            # Get budget-constrained recommendation
            recommended = self.budget_manager.get_recommended_model_size(
                tenant_id, emissions
            )
            
            if recommended and recommended != preferred_name:
                # Check if we need to downgrade due to budget
                budget_state = self.budget_manager.get_budget_state(tenant_id)
                
                if budget_state.status in [BudgetStatus.CRITICAL, BudgetStatus.EXHAUSTED]:
                    preferred_name = recommended
                    selection_reason = "budget"
                    logger.info(
                        "model_downgraded_for_budget",
                        original=preferred_size.value,
                        selected=recommended,
                        budget_status=budget_state.status.value,
                    )
        
        # Verify model is available
        available = self.model_pool.get_available_models()
        if preferred_name not in available:
            # Load the model
            self.model_pool.load_model(preferred_name)
        
        logger.debug(
            "model_selected",
            model=preferred_name,
            reason=selection_reason,
            complexity=complexity.level.value,
        )
        
        return preferred_name, selection_reason
    
    def inference(
        self,
        prompt: str,
        max_tokens: int = 256,
        tenant_id: Optional[str] = None,
        latency_constraint_ms: Optional[float] = None,
        temperature: float = 0.7,
    ) -> AdaptiveInferenceResult:
        """
        Run adaptive inference with automatic model selection.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            tenant_id: Optional tenant ID for budget tracking
            latency_constraint_ms: Optional latency constraint
            temperature: Sampling temperature
            
        Returns:
            AdaptiveInferenceResult with full metadata
        """
        start_time = time.time()
        
        # Step 1: Estimate complexity
        complexity = self.complexity_estimator.estimate(prompt, max_tokens)
        
        # Step 2: Select model
        model_name, selection_reason = self.select_model(
            complexity, tenant_id, latency_constraint_ms
        )
        
        # Step 3: Run inference with early exit
        inference_fn = lambda **kwargs: self.model_pool.inference(
            name=model_name,
            temperature=temperature,
            **kwargs,
        )
        
        early_result = self.early_exit.wrap_inference(
            inference_fn,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        
        # Step 4: Calculate energy and carbon
        base_energy = self.model_pool.estimate_energy(
            model_name,
            early_result.tokens_generated,
        )
        
        # Adjust for early exit savings
        adjusted_energy = self.early_exit.adjust_energy_for_early_exit(
            base_energy,
            early_result.computation_saved_ratio,
        )
        
        carbon = adjusted_energy * self.carbon_intensity
        
        total_latency = (time.time() - start_time) * 1000
        
        # Get input tokens from the inference
        model_info = self.model_pool.get_model_info(model_name)
        
        result = AdaptiveInferenceResult(
            text=early_result.text,
            tokens_generated=early_result.tokens_generated,
            input_tokens=len(prompt.split()),  # Rough estimate
            model_used=model_name,
            model_selected_by=selection_reason,
            complexity=complexity,
            latency_ms=total_latency,
            energy_kwh=adjusted_energy,
            carbon_gco2=carbon,
            early_exit_layer=early_result.exit_layer,
            computation_saved_ratio=early_result.computation_saved_ratio,
        )
        
        logger.info(
            "adaptive_inference_complete",
            model=model_name,
            selection_reason=selection_reason,
            complexity=complexity.level.value,
            tokens=early_result.tokens_generated,
            carbon_gco2=carbon,
            latency_ms=total_latency,
        )
        
        return result
    
    def estimate_emissions(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> dict[str, float]:
        """
        Estimate emissions for a prompt across all models.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            
        Returns:
            Dict mapping model name to estimated carbon (gCO2)
        """
        complexity = self.complexity_estimator.estimate(prompt, max_tokens)
        estimated_tokens = complexity.estimated_tokens
        
        emissions = {}
        for model_name in self.model_pool.list_models():
            energy = self.model_pool.estimate_energy(model_name, estimated_tokens)
            carbon = energy * self.carbon_intensity
            emissions[model_name] = carbon
        
        return emissions
