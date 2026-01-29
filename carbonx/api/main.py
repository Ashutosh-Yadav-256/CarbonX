"""CarbonX FastAPI Application"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from carbonx import CarbonX, __version__
from carbonx.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    BudgetResponse,
    SetBudgetRequest,
    EstimateRequest,
    EstimateResponse,
    MetricsResponse,
    HealthResponse,
    SimulationRequest,
    SimulationResult,
)

logger = structlog.get_logger()

# Global CarbonX instance
_carbonx: Optional[CarbonX] = None


def get_carbonx() -> CarbonX:
    """Get or create the CarbonX instance."""
    global _carbonx
    if _carbonx is None:
        _carbonx = CarbonX()
    return _carbonx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("carbonx_api_starting", version=__version__)
    cx = get_carbonx()
    # Optionally preload models
    # cx.load_all_models()
    
    yield
    
    # Shutdown
    logger.info("carbonx_api_stopping")


app = FastAPI(
    title="CarbonX API",
    description="Carbon-First Framework for Sustainable LLM Inference",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health & Info Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API info."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CarbonX API</title>
        <style>
            body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2d7a4e; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>CarbonX API</h1>
        <p>Carbon-First Framework for Sustainable LLM Inference</p>
        
        <h2>Endpoints</h2>
        <div class="endpoint">
            <strong>POST /inference</strong> - Run carbon-aware inference
        </div>
        <div class="endpoint">
            <strong>GET /budget/{tenant_id}</strong> - Check carbon budget
        </div>
        <div class="endpoint">
            <strong>POST /estimate</strong> - Estimate carbon emissions
        </div>
        <div class="endpoint">
            <strong>GET /metrics</strong> - Prometheus metrics
        </div>
        <div class="endpoint">
            <strong>GET /health</strong> - Health check
        </div>
        
        <p><a href="/docs">Interactive API Documentation</a></p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cx = get_carbonx()
    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded=cx.model_pool.get_available_models(),
        current_carbon_intensity=cx.get_current_carbon_intensity(),
    )


# ============================================================
# Inference Endpoints
# ============================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Run carbon-aware LLM inference.
    
    Automatically selects the optimal model based on:
    - Request complexity
    - Carbon budget
    - Current carbon intensity
    - Latency constraints
    """
    try:
        cx = get_carbonx()
        
        response = cx.inference(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            is_urgent=request.is_urgent,
            latency_constraint_ms=request.latency_constraint_ms,
        )
        
        # Update tenant if specified
        if request.tenant_id and request.tenant_id != "default":
            # Re-run with correct tenant (simplified for now)
            pass
        
        return InferenceResponse(
            text=response.text,
            tokens_generated=response.tokens_generated,
            model_used=response.model_used,
            carbon_gco2=response.carbon_gco2,
            energy_kwh=response.energy_kwh,
            latency_ms=response.latency_ms,
            from_cache=response.from_cache,
            request_id=response.request_id,
        )
        
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/estimate", response_model=EstimateResponse)
async def estimate_carbon(request: EstimateRequest):
    """
    Estimate carbon emissions without running inference.
    
    Returns estimated emissions for each model variant.
    """
    cx = get_carbonx()
    
    estimates = cx.estimate_carbon(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
    )
    
    return EstimateResponse(
        estimates=estimates,
        current_carbon_intensity=cx.get_current_carbon_intensity(),
        region=cx.region,
    )


# ============================================================
# Budget Endpoints
# ============================================================

@app.get("/budget/{tenant_id}", response_model=BudgetResponse)
async def get_budget(tenant_id: str):
    """Get carbon budget status for a tenant."""
    cx = get_carbonx()
    state = cx.budget_manager.get_budget_state(tenant_id)
    
    return BudgetResponse(
        tenant_id=state.tenant_id,
        budget_gco2=state.budget_gco2,
        consumed_gco2=state.consumed_gco2,
        remaining_gco2=state.remaining_gco2,
        consumption_ratio=state.consumption_ratio,
        status=state.status.value,
        window_start=state.window_start.isoformat(),
        window_end=state.window_end.isoformat(),
        window_remaining_hours=state.window_remaining_hours,
    )


@app.put("/budget/{tenant_id}")
async def set_budget(tenant_id: str, request: SetBudgetRequest):
    """Set carbon budget for a tenant."""
    cx = get_carbonx()
    cx.budget_manager.set_tenant_budget(tenant_id, request.budget_gco2)
    
    return {"message": f"Budget set to {request.budget_gco2} gCO2 for tenant {tenant_id}"}


@app.delete("/budget/{tenant_id}")
async def reset_budget(tenant_id: str):
    """Reset carbon budget tracking for a tenant."""
    cx = get_carbonx()
    cx.budget_manager.reset_tenant(tenant_id)
    
    return {"message": f"Budget reset for tenant {tenant_id}"}


# ============================================================
# Metrics Endpoints
# ============================================================

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    cx = get_carbonx()
    content = cx.prometheus.get_metrics()
    return Response(
        content=content,
        media_type=cx.prometheus.get_content_type(),
    )


@app.get("/metrics/json", response_model=MetricsResponse)
async def json_metrics():
    """Get metrics in JSON format."""
    cx = get_carbonx()
    totals = cx.get_metrics()
    
    return MetricsResponse(**totals)


# ============================================================
# Carbon Data Endpoints
# ============================================================

@app.get("/carbon-intensity")
async def get_carbon_intensity(
    region: Optional[str] = Query(None, description="Region code (e.g., US, GB, DE)")
):
    """Get current carbon intensity for a region."""
    cx = get_carbonx()
    target_region = region or cx.region
    
    data = cx.carbon_provider.get_current_intensity(target_region)
    
    return {
        "region": data.region,
        "intensity_gco2_kwh": data.intensity_gco2_kwh,
        "source": data.source.value,
        "timestamp": data.timestamp.isoformat(),
    }


@app.get("/regions")
async def list_regions():
    """List available regions with carbon data."""
    cx = get_carbonx()
    regions = cx.carbon_provider.list_available_regions()
    
    return {"regions": regions}


# ============================================================
# Simulation Endpoints
# ============================================================

@app.post("/simulate", response_model=SimulationResult)
async def run_simulation(request: SimulationRequest):
    """
    Run a simulation to compare CarbonX vs baseline.
    
    Simulates the specified number of requests and compares
    carbon emissions between adaptive CarbonX and static baseline.
    """
    from carbonx.simulator.digital_twin import DigitalTwin
    
    try:
        simulator = DigitalTwin()
        result = simulator.run_comparison(
            num_requests=request.num_requests,
            complexity_distribution=request.complexity_distribution,
        )
        
        return SimulationResult(
            total_requests=result["total_requests"],
            baseline_carbon_gco2=result["baseline_carbon_gco2"],
            carbonx_carbon_gco2=result["carbonx_carbon_gco2"],
            carbon_reduction_percent=result["carbon_reduction_percent"],
            avg_latency_ms=result["avg_latency_ms"],
            model_distribution=result["model_distribution"],
        )
        
    except Exception as e:
        logger.error("simulation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Cache Endpoints
# ============================================================

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    cx = get_carbonx()
    
    if cx.cache:
        return cx.cache.stats
    
    return {"enabled": False}


@app.delete("/cache")
async def clear_cache():
    """Clear the response cache."""
    cx = get_carbonx()
    
    if cx.cache:
        cx.cache.clear()
        return {"message": "Cache cleared"}
    
    return {"message": "Cache not enabled"}


# Run with: uvicorn carbonx.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
