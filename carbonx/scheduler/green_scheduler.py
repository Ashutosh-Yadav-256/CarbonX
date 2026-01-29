"""
Green Scheduler Module

Implements carbon-aware scheduling by solving multi-objective optimization:
    min α·C + β·T + γ·Q
Where:
    C = Carbon emissions
    T = Latency
    Q = Quality degradation
    α, β, γ = Tunable weights
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import structlog

from carbonx.config import SchedulerWeights, ModelSize
from carbonx.inference.complexity_estimator import ComplexityLevel

logger = structlog.get_logger()


class SchedulingAction(str, Enum):
    """Possible scheduling actions."""
    EXECUTE_IMMEDIATELY = "execute_immediately"
    EXECUTE_WITH_SMALL_MODEL = "execute_with_small_model"
    DEFER = "defer"
    REJECT = "reject"


@dataclass
class SchedulingDecision:
    """Result of the scheduling decision."""
    action: SchedulingAction
    recommended_model: Optional[str] = None
    defer_seconds: Optional[float] = None
    reason: str = ""
    score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "recommended_model": self.recommended_model,
            "defer_seconds": self.defer_seconds,
            "reason": self.reason,
            "score": self.score,
        }


@dataclass
class RequestContext:
    """Context for scheduling decision."""
    complexity_level: ComplexityLevel
    estimated_tokens: int
    is_urgent: bool = True
    latency_constraint_ms: Optional[float] = None
    tenant_id: Optional[str] = None
    budget_remaining_gco2: Optional[float] = None
    current_carbon_intensity: float = 400.0


class GreenScheduler:
    """
    Carbon-aware scheduler that optimizes for minimal emissions
    while respecting latency and quality constraints.
    
    Implements: min α·C + β·T + γ·Q
    """
    
    def __init__(
        self,
        weights: Optional[SchedulerWeights] = None,
        high_carbon_threshold: float = 600.0,  # gCO2/kWh
        low_carbon_threshold: float = 200.0,   # gCO2/kWh
    ):
        """
        Initialize the green scheduler.
        
        Args:
            weights: Optimization weights (α, β, γ)
            high_carbon_threshold: Carbon intensity above which to consider deferring
            low_carbon_threshold: Carbon intensity below which to prioritize execution
        """
        self.weights = weights or SchedulerWeights()
        self.high_carbon_threshold = high_carbon_threshold
        self.low_carbon_threshold = low_carbon_threshold
        
        # Energy per token estimates for each model size (kWh)
        self.model_energy = {
            ModelSize.SMALL: 1e-8,
            ModelSize.MEDIUM: 3e-8,
            ModelSize.LARGE: 8e-8,
        }
        
        logger.info(
            "green_scheduler_initialized",
            alpha=self.weights.alpha_carbon,
            beta=self.weights.beta_latency,
            gamma=self.weights.gamma_quality,
        )
    
    def schedule(self, context: RequestContext) -> SchedulingDecision:
        """
        Make a scheduling decision for a request.
        
        Args:
            context: Request context with complexity, constraints, etc.
            
        Returns:
            SchedulingDecision with recommended action
        """
        # Calculate scores for each possible action
        candidates = self._generate_candidates(context)
        
        # Score each candidate
        scored = []
        for model_size, action in candidates:
            score = self._calculate_score(model_size, action, context)
            scored.append((score, model_size, action))
        
        # Select best (lowest score)
        scored.sort(key=lambda x: x[0])
        best_score, best_model, best_action = scored[0]
        
        # Build decision
        decision = SchedulingDecision(
            action=best_action,
            recommended_model=best_model.value if best_model else None,
            reason=self._explain_decision(best_model, best_action, context),
            score=best_score,
        )
        
        if best_action == SchedulingAction.DEFER:
            decision.defer_seconds = self._estimate_defer_time(context)
        
        logger.info(
            "scheduling_decision",
            action=best_action.value,
            model=decision.recommended_model,
            score=best_score,
            carbon_intensity=context.current_carbon_intensity,
        )
        
        return decision
    
    def _generate_candidates(
        self,
        context: RequestContext,
    ) -> list[tuple[Optional[ModelSize], SchedulingAction]]:
        """Generate candidate (model, action) pairs to evaluate."""
        candidates = []
        
        # Always consider immediate execution with different models
        for size in ModelSize:
            candidates.append((size, SchedulingAction.EXECUTE_IMMEDIATELY))
        
        # Consider deferring if not urgent and carbon is high
        if not context.is_urgent:
            if context.current_carbon_intensity > self.high_carbon_threshold:
                candidates.append((None, SchedulingAction.DEFER))
        
        # Consider forced small model if budget is low
        if context.budget_remaining_gco2 is not None:
            if context.budget_remaining_gco2 < 10.0:  # Very low budget
                candidates.append((ModelSize.SMALL, SchedulingAction.EXECUTE_WITH_SMALL_MODEL))
        
        return candidates
    
    def _calculate_score(
        self,
        model_size: Optional[ModelSize],
        action: SchedulingAction,
        context: RequestContext,
    ) -> float:
        """
        Calculate optimization score: α·C + β·T + γ·Q
        
        Lower score is better.
        """
        if action == SchedulingAction.DEFER:
            # Deferring has zero immediate carbon cost but latency penalty
            carbon_score = 0.0
            latency_score = 1.0  # Max latency penalty
            quality_score = 0.0  # No quality impact
            
        elif action == SchedulingAction.REJECT:
            # Rejection is a last resort
            return float('inf')
            
        else:
            # Estimate carbon for this model
            energy = self.model_energy.get(model_size, 3e-8)
            carbon = energy * context.estimated_tokens * context.current_carbon_intensity
            
            # Normalize carbon score (0-1)
            max_carbon = 8e-8 * 500 * 800  # Large model, 500 tokens, high intensity
            carbon_score = min(1.0, carbon / max_carbon)
            
            # Latency score based on model size
            latency_map = {
                ModelSize.SMALL: 0.2,
                ModelSize.MEDIUM: 0.5,
                ModelSize.LARGE: 1.0,
            }
            latency_score = latency_map.get(model_size, 0.5)
            
            # Quality score (inverse - lower is worse quality)
            quality_map = {
                ModelSize.SMALL: 0.3,   # Lower quality = higher penalty
                ModelSize.MEDIUM: 0.15,
                ModelSize.LARGE: 0.0,   # Best quality = no penalty
            }
            quality_score = quality_map.get(model_size, 0.15)
            
            # Adjust for complexity match
            complexity_model = {
                ComplexityLevel.LOW: ModelSize.SMALL,
                ComplexityLevel.MEDIUM: ModelSize.MEDIUM,
                ComplexityLevel.HIGH: ModelSize.LARGE,
            }
            expected = complexity_model.get(context.complexity_level)
            if model_size == expected:
                quality_score *= 0.5  # Reduce penalty if model matches complexity
        
        # Apply weights
        score = (
            self.weights.alpha_carbon * carbon_score +
            self.weights.beta_latency * latency_score +
            self.weights.gamma_quality * quality_score
        )
        
        return score
    
    def _explain_decision(
        self,
        model_size: Optional[ModelSize],
        action: SchedulingAction,
        context: RequestContext,
    ) -> str:
        """Generate human-readable explanation for the decision."""
        if action == SchedulingAction.DEFER:
            return (
                f"Deferring request due to high carbon intensity "
                f"({context.current_carbon_intensity:.0f} gCO2/kWh)"
            )
        
        if action == SchedulingAction.EXECUTE_WITH_SMALL_MODEL:
            return "Using small model due to low carbon budget remaining"
        
        intensity_level = "low" if context.current_carbon_intensity < self.low_carbon_threshold else \
                         "high" if context.current_carbon_intensity > self.high_carbon_threshold else "medium"
        
        return (
            f"Executing with {model_size.value} model "
            f"(complexity: {context.complexity_level.value}, "
            f"carbon intensity: {intensity_level})"
        )
    
    def _estimate_defer_time(self, context: RequestContext) -> float:
        """Estimate how long to defer the request."""
        # Simple heuristic: defer more when carbon is higher
        base_defer = 60.0  # 1 minute base
        
        if context.current_carbon_intensity > 700:
            return base_defer * 3
        elif context.current_carbon_intensity > 500:
            return base_defer * 2
        else:
            return base_defer
    
    def update_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Update optimization weights dynamically."""
        if alpha is not None:
            self.weights.alpha_carbon = alpha
        if beta is not None:
            self.weights.beta_latency = beta
        if gamma is not None:
            self.weights.gamma_quality = gamma
        
        logger.info(
            "scheduler_weights_updated",
            alpha=self.weights.alpha_carbon,
            beta=self.weights.beta_latency,
            gamma=self.weights.gamma_quality,
        )
