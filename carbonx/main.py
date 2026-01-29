"""
Main CarbonX Module

Provides the unified CarbonX class that integrates all components
into a single easy-to-use interface.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid
import structlog

from carbonx.config import CarbonXConfig, ModelSize
from carbonx.carbon_accounting import CarbonAccountant, CarbonMeasurement
from carbonx.budget_manager import CarbonBudgetManager, BudgetState
from carbonx.inference.model_pool import ModelPool
from carbonx.inference.complexity_estimator import ComplexityEstimator
from carbonx.inference.early_exit import EarlyExitWrapper
from carbonx.inference.adaptive_runtime import AdaptiveRuntime, AdaptiveInferenceResult
from carbonx.scheduler.green_scheduler import GreenScheduler, RequestContext, SchedulingAction
from carbonx.scheduler.demand_shaper import DemandShaper
from carbonx.data.carbon_intensity import CarbonIntensityProvider
from carbonx.data.energy_telemetry import EnergyTelemetry
from carbonx.cache.semantic_cache import SemanticCache
from carbonx.telemetry.collector import TelemetryCollector
from carbonx.telemetry.prometheus_exporter import PrometheusExporter

logger = structlog.get_logger()


@dataclass
class InferenceResponse:
    """Response from CarbonX inference."""
    text: str
    tokens_generated: int
    model_used: str
    carbon_gco2: float
    energy_kwh: float
    latency_ms: float
    from_cache: bool = False
    request_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "model_used": self.model_used,
            "carbon_gco2": self.carbon_gco2,
            "energy_kwh": self.energy_kwh,
            "latency_ms": self.latency_ms,
            "from_cache": self.from_cache,
            "request_id": self.request_id,
        }


class CarbonX:
    """
    CarbonX: A Carbon-First Framework for Sustainable LLM Inference
    
    Integrates carbon accounting, adaptive model selection, early-exit
    inference, and budget enforcement into a unified system.
    
    Example:
        ```python
        cx = CarbonX(carbon_budget_gco2=100.0, tenant_id="my-app")
        
        response = cx.inference("What is climate change?")
        
        print(f"Response: {response.text}")
        print(f"Carbon: {response.carbon_gco2:.4f} gCO2")
        print(f"Budget remaining: {cx.budget_remaining:.2f} gCO2")
        ```
    """
    
    def __init__(
        self,
        config: Optional[CarbonXConfig] = None,
        carbon_budget_gco2: Optional[float] = None,
        tenant_id: str = "default",
        region: str = "US",
        enable_cache: bool = True,
        enable_early_exit: bool = True,
    ):
        """
        Initialize CarbonX.
        
        Args:
            config: Optional configuration object
            carbon_budget_gco2: Carbon budget in gCO2 (overrides config)
            tenant_id: Tenant identifier for budget tracking
            region: Region for carbon intensity data
            enable_cache: Whether to enable response caching
            enable_early_exit: Whether to enable early-exit inference
        """
        self.config = config or CarbonXConfig.from_env()
        self.tenant_id = tenant_id
        self.region = region
        
        # Override budget if provided
        if carbon_budget_gco2 is not None:
            self.config.default_carbon_budget_gco2 = carbon_budget_gco2
        
        # Initialize components
        self._init_components(enable_cache, enable_early_exit)
        
        logger.info(
            "carbonx_initialized",
            tenant_id=tenant_id,
            budget_gco2=self.config.default_carbon_budget_gco2,
            region=region,
        )
    
    def _init_components(self, enable_cache: bool, enable_early_exit: bool) -> None:
        """Initialize all framework components."""
        # Carbon accounting
        self.accountant = CarbonAccountant()
        
        # Budget management
        self.budget_manager = CarbonBudgetManager(
            default_budget_gco2=self.config.default_carbon_budget_gco2,
            window_hours=self.config.budget_window_hours,
        )
        
        # Carbon intensity data
        self.carbon_provider = CarbonIntensityProvider(
            region=self.region,
            api_key=self.config.electricity_maps_api_key,
        )
        
        # Energy telemetry
        self.energy_telemetry = EnergyTelemetry()
        
        # Model pool
        self.model_pool = ModelPool()
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator()
        
        # Early exit
        self.early_exit = EarlyExitWrapper(
            confidence_threshold=self.config.early_exit.confidence_threshold,
            enabled=enable_early_exit,
        )
        
        # Adaptive runtime
        self.runtime = AdaptiveRuntime(
            config=self.config,
            model_pool=self.model_pool,
            complexity_estimator=self.complexity_estimator,
            early_exit_wrapper=self.early_exit,
            budget_manager=self.budget_manager,
        )
        
        # Scheduler
        self.scheduler = GreenScheduler(weights=self.config.scheduler_weights)
        
        # Demand shaper
        self.demand_shaper = DemandShaper()
        
        # Cache
        self.cache = SemanticCache(
            similarity_threshold=self.config.cache.similarity_threshold,
            use_embeddings=False,  # Start simple
        ) if enable_cache else None
        
        # Telemetry
        self.telemetry = TelemetryCollector()
        self.prometheus = PrometheusExporter()
    
    def inference(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        is_urgent: bool = True,
        latency_constraint_ms: Optional[float] = None,
    ) -> InferenceResponse:
        """
        Run carbon-aware inference.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            is_urgent: Whether the request is time-sensitive
            latency_constraint_ms: Optional latency constraint
            
        Returns:
            InferenceResponse with result and carbon metrics
        """
        request_id = str(uuid.uuid4())[:8]
        
        # Check cache first
        if self.cache:
            cache_result = self.cache.lookup(prompt)
            if cache_result.hit:
                self.telemetry.record_cache_hit()
                self.prometheus.record_cache_hit()
                
                logger.info(
                    "cache_hit",
                    request_id=request_id,
                    similarity=cache_result.similarity,
                )
                
                return InferenceResponse(
                    text=cache_result.entry.response,
                    tokens_generated=len(cache_result.entry.response.split()),
                    model_used=cache_result.entry.model_used,
                    carbon_gco2=0.0,  # No new emissions
                    energy_kwh=0.0,
                    latency_ms=1.0,  # Minimal cache lookup time
                    from_cache=True,
                    request_id=request_id,
                )
            else:
                self.telemetry.record_cache_miss()
                self.prometheus.record_cache_miss()
        
        # Get current carbon intensity
        carbon_intensity = self.carbon_provider.get_intensity_value(self.region)
        self.runtime.set_carbon_intensity(carbon_intensity)
        
        # Estimate complexity
        complexity = self.complexity_estimator.estimate(prompt, max_tokens)
        
        # Get budget state
        budget_state = self.budget_manager.get_budget_state(self.tenant_id)
        
        # Check scheduling decision
        context = RequestContext(
            complexity_level=complexity.level,
            estimated_tokens=complexity.estimated_tokens,
            is_urgent=is_urgent,
            latency_constraint_ms=latency_constraint_ms,
            tenant_id=self.tenant_id,
            budget_remaining_gco2=budget_state.remaining_gco2,
            current_carbon_intensity=carbon_intensity,
        )
        
        schedule_decision = self.scheduler.schedule(context)
        
        # Handle deferral (for now, just log - actual deferral requires async)
        if schedule_decision.action == SchedulingAction.DEFER:
            logger.info(
                "request_would_defer",
                request_id=request_id,
                defer_seconds=schedule_decision.defer_seconds,
            )
            # For synchronous API, we proceed anyway
        
        # Run adaptive inference
        result = self.runtime.inference(
            prompt=prompt,
            max_tokens=max_tokens,
            tenant_id=self.tenant_id,
            latency_constraint_ms=latency_constraint_ms,
            temperature=temperature,
        )
        
        # Record measurement
        measurement = self.accountant.record_measurement(
            energy_kwh=result.energy_kwh,
            carbon_intensity_gco2_kwh=carbon_intensity,
            model_used=result.model_used,
            tokens_generated=result.tokens_generated,
            latency_ms=result.latency_ms,
            tenant_id=self.tenant_id,
            request_id=request_id,
        )
        
        # Update budget
        self.budget_manager.record_emission(measurement, self.tenant_id)
        
        # Record telemetry
        self.telemetry.record(measurement)
        self.prometheus.record_request(
            model=result.model_used,
            tenant=self.tenant_id,
            tokens=result.tokens_generated,
            carbon_gco2=result.carbon_gco2,
            energy_kwh=result.energy_kwh,
            latency_seconds=result.latency_ms / 1000,
        )
        
        # Store in cache
        if self.cache:
            self.cache.store(
                prompt=prompt,
                response=result.text,
                model_used=result.model_used,
                carbon_gco2=result.carbon_gco2,
            )
        
        # Update prometheus gauges
        new_budget_state = self.budget_manager.get_budget_state(self.tenant_id)
        self.prometheus.set_budget_remaining(self.tenant_id, new_budget_state.remaining_gco2)
        self.prometheus.set_carbon_intensity(self.region, carbon_intensity)
        
        logger.info(
            "inference_complete",
            request_id=request_id,
            model=result.model_used,
            tokens=result.tokens_generated,
            carbon_gco2=result.carbon_gco2,
            latency_ms=result.latency_ms,
        )
        
        return InferenceResponse(
            text=result.text,
            tokens_generated=result.tokens_generated,
            model_used=result.model_used,
            carbon_gco2=result.carbon_gco2,
            energy_kwh=result.energy_kwh,
            latency_ms=result.latency_ms,
            from_cache=False,
            request_id=request_id,
        )
    
    @property
    def budget_remaining(self) -> float:
        """Get remaining carbon budget in gCO2."""
        state = self.budget_manager.get_budget_state(self.tenant_id)
        return state.remaining_gco2
    
    @property
    def budget_state(self) -> BudgetState:
        """Get full budget state."""
        return self.budget_manager.get_budget_state(self.tenant_id)
    
    def get_metrics(self) -> dict:
        """Get current telemetry metrics."""
        return self.telemetry.get_totals()
    
    def get_current_carbon_intensity(self) -> float:
        """Get current carbon intensity."""
        return self.carbon_provider.get_intensity_value(self.region)
    
    def set_carbon_budget(self, budget_gco2: float) -> None:
        """Update the carbon budget for this tenant."""
        self.budget_manager.set_tenant_budget(self.tenant_id, budget_gco2)
        logger.info("budget_updated", tenant_id=self.tenant_id, budget=budget_gco2)
    
    def load_model(self, model_name: str) -> bool:
        """Pre-load a model into memory."""
        return self.model_pool.load_model(model_name)
    
    def load_all_models(self) -> None:
        """Pre-load all models."""
        for model in self.model_pool.list_models():
            self.model_pool.load_model(model)
    
    def estimate_carbon(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> dict[str, float]:
        """
        Estimate carbon emissions without running inference.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            
        Returns:
            Dict mapping model name to estimated carbon (gCO2)
        """
        carbon_intensity = self.carbon_provider.get_intensity_value(self.region)
        self.runtime.set_carbon_intensity(carbon_intensity)
        return self.runtime.estimate_emissions(prompt, max_tokens)
