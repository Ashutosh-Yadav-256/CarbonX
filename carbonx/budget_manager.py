"""
Carbon Budget Manager

Implements explicit carbon budget enforcement at the tenant level.
Enforces constraint: Σ Cᵢ ≤ B (sum of emissions ≤ budget)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from collections import defaultdict
import threading
import structlog

from carbonx.carbon_accounting import CarbonMeasurement

logger = structlog.get_logger()


class BudgetStatus(str, Enum):
    """Budget consumption status levels."""
    OK = "ok"                    # < 50% consumed
    WARNING = "warning"          # 50-80% consumed
    CRITICAL = "critical"        # 80-95% consumed
    EXHAUSTED = "exhausted"      # > 95% consumed


@dataclass
class BudgetState:
    """Current state of a tenant's carbon budget."""
    tenant_id: str
    budget_gco2: float
    consumed_gco2: float
    window_start: datetime
    window_hours: int
    
    @property
    def remaining_gco2(self) -> float:
        """Remaining carbon budget."""
        return max(0.0, self.budget_gco2 - self.consumed_gco2)
    
    @property
    def consumption_ratio(self) -> float:
        """Ratio of consumed vs total budget (0-1)."""
        if self.budget_gco2 == 0:
            return 1.0
        return min(1.0, self.consumed_gco2 / self.budget_gco2)
    
    @property
    def status(self) -> BudgetStatus:
        """Current budget status."""
        ratio = self.consumption_ratio
        if ratio >= 0.95:
            return BudgetStatus.EXHAUSTED
        elif ratio >= 0.80:
            return BudgetStatus.CRITICAL
        elif ratio >= 0.50:
            return BudgetStatus.WARNING
        return BudgetStatus.OK
    
    @property
    def window_end(self) -> datetime:
        """End of the current budget window."""
        return self.window_start + timedelta(hours=self.window_hours)
    
    @property
    def window_remaining_hours(self) -> float:
        """Hours remaining in current window."""
        remaining = self.window_end - datetime.utcnow()
        return max(0.0, remaining.total_seconds() / 3600)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "tenant_id": self.tenant_id,
            "budget_gco2": self.budget_gco2,
            "consumed_gco2": self.consumed_gco2,
            "remaining_gco2": self.remaining_gco2,
            "consumption_ratio": self.consumption_ratio,
            "status": self.status.value,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "window_remaining_hours": self.window_remaining_hours,
        }


@dataclass
class BudgetEntry:
    """Internal budget tracking entry."""
    emissions: list[CarbonMeasurement] = field(default_factory=list)
    window_start: datetime = field(default_factory=datetime.utcnow)
    
    def prune_expired(self, window_hours: int) -> None:
        """Remove emissions outside the current window."""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        self.emissions = [e for e in self.emissions if e.timestamp >= cutoff]
        
        # Reset window if all entries pruned
        if not self.emissions:
            self.window_start = datetime.utcnow()
    
    def total_emissions(self) -> float:
        """Total emissions in current window."""
        return sum(e.carbon_gco2 for e in self.emissions)


class CarbonBudgetManager:
    """
    Manages carbon budgets for tenants with sliding window enforcement.
    
    Key features:
    - Per-tenant budget tracking
    - Sliding window for budget periods
    - Real-time enforcement with status levels
    - Thread-safe operations
    """
    
    def __init__(
        self,
        default_budget_gco2: float = 1000.0,
        window_hours: int = 24,
    ):
        """
        Initialize the budget manager.
        
        Args:
            default_budget_gco2: Default carbon budget per tenant (gCO2)
            window_hours: Budget window duration in hours
        """
        self.default_budget_gco2 = default_budget_gco2
        self.window_hours = window_hours
        
        # Tenant-specific budgets (can override default)
        self._tenant_budgets: dict[str, float] = {}
        
        # Current emissions per tenant
        self._entries: dict[str, BudgetEntry] = defaultdict(BudgetEntry)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            "budget_manager_initialized",
            default_budget=default_budget_gco2,
            window_hours=window_hours,
        )
    
    def set_tenant_budget(self, tenant_id: str, budget_gco2: float) -> None:
        """
        Set a custom budget for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            budget_gco2: Carbon budget in gCO2
        """
        with self._lock:
            self._tenant_budgets[tenant_id] = budget_gco2
            logger.info(
                "tenant_budget_set",
                tenant_id=tenant_id,
                budget_gco2=budget_gco2,
            )
    
    def get_tenant_budget(self, tenant_id: str) -> float:
        """Get the budget for a tenant (custom or default)."""
        return self._tenant_budgets.get(tenant_id, self.default_budget_gco2)
    
    def record_emission(
        self,
        measurement: CarbonMeasurement,
        tenant_id: Optional[str] = None,
    ) -> BudgetState:
        """
        Record a carbon emission and update budget.
        
        Args:
            measurement: The carbon measurement to record
            tenant_id: Optional tenant ID (uses measurement's if not provided)
            
        Returns:
            Current BudgetState after recording
        """
        tenant = tenant_id or measurement.tenant_id or "default"
        
        with self._lock:
            entry = self._entries[tenant]
            
            # Prune expired entries first
            entry.prune_expired(self.window_hours)
            
            # Add new measurement
            entry.emissions.append(measurement)
            
            state = self.get_budget_state(tenant)
            
            logger.info(
                "emission_recorded",
                tenant_id=tenant,
                emission_gco2=measurement.carbon_gco2,
                total_consumed=state.consumed_gco2,
                status=state.status.value,
            )
            
            return state
    
    def get_budget_state(self, tenant_id: str) -> BudgetState:
        """
        Get the current budget state for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Current BudgetState
        """
        with self._lock:
            entry = self._entries.get(tenant_id)
            
            if entry is None:
                # New tenant, no emissions yet
                return BudgetState(
                    tenant_id=tenant_id,
                    budget_gco2=self.get_tenant_budget(tenant_id),
                    consumed_gco2=0.0,
                    window_start=datetime.utcnow(),
                    window_hours=self.window_hours,
                )
            
            # Prune expired entries
            entry.prune_expired(self.window_hours)
            
            return BudgetState(
                tenant_id=tenant_id,
                budget_gco2=self.get_tenant_budget(tenant_id),
                consumed_gco2=entry.total_emissions(),
                window_start=entry.window_start,
                window_hours=self.window_hours,
            )
    
    def check_budget_available(
        self,
        tenant_id: str,
        estimated_emission_gco2: float,
    ) -> tuple[bool, BudgetStatus]:
        """
        Check if budget is available for an estimated emission.
        
        Args:
            tenant_id: Tenant identifier
            estimated_emission_gco2: Estimated emission for the request
            
        Returns:
            Tuple of (is_available, current_status)
        """
        state = self.get_budget_state(tenant_id)
        
        would_exceed = (state.consumed_gco2 + estimated_emission_gco2) > state.budget_gco2
        
        logger.debug(
            "budget_check",
            tenant_id=tenant_id,
            estimated=estimated_emission_gco2,
            remaining=state.remaining_gco2,
            would_exceed=would_exceed,
        )
        
        return (not would_exceed, state.status)
    
    def get_recommended_model_size(
        self,
        tenant_id: str,
        estimated_emissions: dict[str, float],
    ) -> Optional[str]:
        """
        Get recommended model size based on budget constraints.
        
        Args:
            tenant_id: Tenant identifier
            estimated_emissions: Dict mapping model size to estimated emission
            
        Returns:
            Recommended model size or None if all exceed budget
        """
        state = self.get_budget_state(tenant_id)
        remaining = state.remaining_gco2
        
        # Sort by emission (lowest first)
        sorted_models = sorted(estimated_emissions.items(), key=lambda x: x[1])
        
        # Find the largest model that fits the budget
        recommended = None
        for model_size, emission in reversed(sorted_models):
            if emission <= remaining:
                recommended = model_size
                break
        
        # If none fit, recommend the smallest
        if recommended is None and sorted_models:
            recommended = sorted_models[0][0]
            
        logger.debug(
            "model_recommendation",
            tenant_id=tenant_id,
            remaining=remaining,
            recommended=recommended,
        )
        
        return recommended
    
    def reset_tenant(self, tenant_id: str) -> None:
        """Reset emissions for a specific tenant."""
        with self._lock:
            if tenant_id in self._entries:
                del self._entries[tenant_id]
            logger.info("tenant_reset", tenant_id=tenant_id)
    
    def reset_all(self) -> None:
        """Reset all tenant emissions."""
        with self._lock:
            self._entries.clear()
            logger.info("all_tenants_reset")
