"""
CarbonX: A Carbon-First, Open-Source Framework for Sustainable LLM Inference

This package provides tools for carbon-aware LLM inference with explicit
budget enforcement, adaptive model selection, and real-time carbon accounting.
"""

from carbonx.main import CarbonX
from carbonx.config import CarbonXConfig
from carbonx.carbon_accounting import CarbonAccountant
from carbonx.budget_manager import CarbonBudgetManager

__version__ = "0.1.0"
__author__ = "Ashutosh Yadav"

__all__ = [
    "CarbonX",
    "CarbonXConfig",
    "CarbonAccountant",
    "CarbonBudgetManager",
]
