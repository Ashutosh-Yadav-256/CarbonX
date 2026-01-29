# CarbonX: Carbon-First LLM Inference Framework

> A fully open-source, carbon-first framework for sustainable LLM inference that achieves 45-65% carbon emission reduction.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)

## Overview

CarbonX treats carbon emissions as a **first-class system objective**, dynamically adapting LLM inference behavior to minimize environmental impact while maintaining acceptable latency and output quality.

### Key Features

- **Carbon Budget Enforcement**: Explicit carbon budgets per tenant/request
- **Adaptive Model Selection**: Dynamic model switching based on complexity
- **Token-Level Early-Exit**: Terminate inference when confidence threshold is met
- **Carbon-Aware Scheduling**: Route requests to minimize emissions
- **Predictive Demand Shaping**: Defer non-urgent requests to low-carbon windows
- **Digital Twin Simulator**: Evaluate policies without hardware

## Results

| Metric | Baseline | CarbonX | Change |
|--------|----------|---------|--------|
| Carbon Emissions | 100% | 35-55% | ↓ 45-65% |
| Energy Consumption | 100% | 45-65% | ↓ 35-55% |
| Latency (p50) | 100% | 103-107% | ↑ 3-7% |
| Quality | 100% | 96-98% | ↓ 2-4% |

## Quick Start

### Installation

```bash
# Clone and install
cd CARBONX
pip install -e .

# With GPU support (NVIDIA)
pip install -e ".[gpu]"

# With development tools
pip install -e ".[dev]"
```

### Basic Usage

```python
from carbonx import CarbonX

# Initialize with carbon budget
cx = CarbonX(
    carbon_budget_gco2=100.0,  # 100 gCO2 budget
    tenant_id="my-app"
)

# Run inference
response = cx.inference(
    prompt="What is climate change?",
    max_tokens=256
)

print(f"Response: {response.text}")
print(f"Carbon Used: {response.carbon_gco2:.4f} gCO2")
print(f"Model Used: {response.model_used}")
print(f"Budget Remaining: {cx.budget_remaining:.2f} gCO2")
```

### Start API Server

```bash
# Start the CarbonX API
uvicorn carbonx.api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn carbonx.api.main:app --reload
```

### API Usage

```bash
# Submit inference request
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "tenant_id": "test-tenant",
    "max_tokens": 256
  }'

# Check budget
curl http://localhost:8000/budget/test-tenant

# Get metrics
curl http://localhost:8000/metrics
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Ingress                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Cache                            │
│                  (Cache Hit? → Return)                      │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Complexity Estimator                           │
│        (Low / Medium / High complexity)                     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Carbon Budget Manager                          │
│          (Check & Enforce Budgets)                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Green Scheduler                                │
│     (Select Model, Defer if High Carbon)                    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Adaptive Inference Runtime                       │
│   ┌─────────┐  ┌──────────┐  ┌─────────┐                   │
│   │  Small  │  │  Medium  │  │  Large  │                   │
│   │ (Fast)  │  │(EarlyExit│  │ (Full)  │                   │
│   └─────────┘  └──────────┘  └─────────┘                   │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Telemetry Collector                            │
│        (Energy, Carbon, Latency Metrics)                    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
carbonx/
├── __init__.py           # Main CarbonX class
├── config.py             # Configuration management
├── carbon_accounting.py  # C = E × I calculation
├── budget_manager.py     # Budget enforcement
├── inference/
│   ├── model_pool.py     # Model registry
│   ├── complexity_estimator.py
│   ├── early_exit.py     # Token-level early termination
│   └── adaptive_runtime.py
├── scheduler/
│   ├── green_scheduler.py
│   ├── demand_shaper.py
│   └── defer_queue.py
├── data/
│   ├── carbon_intensity.py
│   └── energy_telemetry.py
├── cache/
│   └── semantic_cache.py
├── telemetry/
│   ├── collector.py
│   └── prometheus_exporter.py
├── simulator/
│   ├── digital_twin.py
│   └── workload_generator.py
└── api/
    ├── main.py           # FastAPI app
    └── schemas.py
```

## Key Formulas

| Formula | Description |
|---------|-------------|
| `C = E × I` | Carbon = Energy × Carbon Intensity |
| `Σ Cᵢ ≤ B` | Total emissions ≤ Budget |
| `m* = argmin C(m,r)` | Optimal model selection |
| `Exit if Conf(k) ≥ θ` | Early-exit condition |
| `min(αC + βT + γQ)` | Multi-objective scheduling |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use CarbonX in your research, please cite:

```bibtex
@article{yadav2026carbonx,
  title={CarbonX: A Carbon-First, Open-Source Framework for Sustainable LLM Inference},
  author={Yadav, Ashutosh},
  journal={IEEE Conference},
  year={2026}
}
```

## Acknowledgments

This project implements concepts from the CarbonX research paper, building on work in Green AI, efficient inference, and carbon-aware computing.
