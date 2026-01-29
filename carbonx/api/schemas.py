"""
Pydantic Schemas for CarbonX API
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request body for inference endpoint."""
    prompt: str = Field(..., description="The input prompt for inference")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    tenant_id: Optional[str] = Field("default", description="Tenant identifier for budget tracking")
    is_urgent: bool = Field(True, description="Whether the request is time-sensitive")
    latency_constraint_ms: Optional[float] = Field(None, description="Optional latency constraint")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is climate change?",
                "max_tokens": 256,
                "temperature": 0.7,
                "tenant_id": "my-app",
                "is_urgent": True
            }
        }


class InferenceResponse(BaseModel):
    """Response from inference endpoint."""
    text: str
    tokens_generated: int
    model_used: str
    carbon_gco2: float
    energy_kwh: float
    latency_ms: float
    from_cache: bool
    request_id: str


class BudgetResponse(BaseModel):
    """Response from budget endpoint."""
    tenant_id: str
    budget_gco2: float
    consumed_gco2: float
    remaining_gco2: float
    consumption_ratio: float
    status: str
    window_start: str
    window_end: str
    window_remaining_hours: float


class SetBudgetRequest(BaseModel):
    """Request to set a tenant's budget."""
    budget_gco2: float = Field(..., gt=0, description="New carbon budget in gCO2")


class EstimateRequest(BaseModel):
    """Request for carbon estimation."""
    prompt: str
    max_tokens: int = 256


class EstimateResponse(BaseModel):
    """Response with carbon estimates per model."""
    estimates: dict[str, float]
    current_carbon_intensity: float
    region: str


class MetricsResponse(BaseModel):
    """Response from metrics endpoint."""
    total_requests: int
    total_tokens: int
    total_energy_kwh: float
    total_carbon_gco2: float
    avg_latency_ms: float
    model_distribution: dict[str, int]
    cache_hits: int
    cache_misses: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: list[str]
    current_carbon_intensity: float


class SimulationRequest(BaseModel):
    """Request for running simulation."""
    num_requests: int = Field(100, ge=1, le=10000)
    complexity_distribution: dict[str, float] = Field(
        default={"low": 0.4, "medium": 0.4, "high": 0.2}
    )
    carbon_intensity_range: tuple[float, float] = Field(default=(200.0, 600.0))


class SimulationResult(BaseModel):
    """Result from simulation run."""
    total_requests: int
    baseline_carbon_gco2: float
    carbonx_carbon_gco2: float
    carbon_reduction_percent: float
    avg_latency_ms: float
    model_distribution: dict[str, int]
