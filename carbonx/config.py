"""
CarbonX Configuration Management

Centralized configuration for all CarbonX components with environment
variable support and sensible defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelSize(str, Enum):
    """Available model size variants."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class CarbonDataSource(str, Enum):
    """Carbon intensity data sources."""
    STATIC = "static"
    ELECTRICITY_MAPS = "electricity_maps"
    WATTTIME = "watttime"


@dataclass
class ModelConfig:
    """Configuration for a single model variant."""
    name: str
    hf_model_id: str
    energy_per_token_kwh: float  # Estimated energy per token in kWh
    quality_score: float  # Relative quality score (0-1)
    max_context_length: int = 2048


@dataclass
class SchedulerWeights:
    """Weights for multi-objective scheduling optimization.
    
    Optimizes: α·Carbon + β·Latency + γ·QualityLoss
    """
    alpha_carbon: float = 0.5
    beta_latency: float = 0.3
    gamma_quality: float = 0.2


@dataclass
class EarlyExitConfig:
    """Configuration for token-level early exit."""
    enabled: bool = True
    confidence_threshold: float = 0.85
    min_layers: int = 2  # Minimum layers before considering exit


@dataclass
class CacheConfig:
    """Configuration for semantic caching."""
    enabled: bool = True
    similarity_threshold: float = 0.92
    max_cache_size: int = 10000
    redis_url: Optional[str] = None


@dataclass
class TelemetryConfig:
    """Configuration for telemetry and monitoring."""
    enabled: bool = True
    prometheus_port: int = 9090
    log_level: str = "INFO"


@dataclass
class CarbonXConfig:
    """Main configuration for CarbonX framework."""
    
    # Carbon budget settings
    default_carbon_budget_gco2: float = 1000.0
    budget_window_hours: int = 24
    
    # Carbon data source
    carbon_data_source: CarbonDataSource = CarbonDataSource.STATIC
    default_region: str = "US"
    electricity_maps_api_key: Optional[str] = None
    
    # Model configuration
    default_model_size: ModelSize = ModelSize.MEDIUM
    models: dict[ModelSize, ModelConfig] = field(default_factory=dict)
    
    # Component configurations
    scheduler_weights: SchedulerWeights = field(default_factory=SchedulerWeights)
    early_exit: EarlyExitConfig = field(default_factory=EarlyExitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    def __post_init__(self):
        """Initialize with defaults if models not provided."""
        if not self.models:
            self.models = self._default_models()
    
    @staticmethod
    def _default_models() -> dict[ModelSize, ModelConfig]:
        """Default model configurations using HuggingFace models."""
        return {
            ModelSize.SMALL: ModelConfig(
                name="small",
                hf_model_id=os.getenv("MODEL_SMALL", "distilgpt2"),
                energy_per_token_kwh=1e-8,  # ~0.01 mWh per token
                quality_score=0.7,
                max_context_length=1024,
            ),
            ModelSize.MEDIUM: ModelConfig(
                name="medium", 
                hf_model_id=os.getenv("MODEL_MEDIUM", "gpt2"),
                energy_per_token_kwh=3e-8,  # ~0.03 mWh per token
                quality_score=0.85,
                max_context_length=1024,
            ),
            ModelSize.LARGE: ModelConfig(
                name="large",
                hf_model_id=os.getenv("MODEL_LARGE", "gpt2-large"),
                energy_per_token_kwh=8e-8,  # ~0.08 mWh per token
                quality_score=1.0,
                max_context_length=1024,
            ),
        }
    
    @classmethod
    def from_env(cls) -> "CarbonXConfig":
        """Create configuration from environment variables."""
        return cls(
            default_carbon_budget_gco2=float(
                os.getenv("DEFAULT_CARBON_BUDGET_GCO2", "1000.0")
            ),
            budget_window_hours=int(os.getenv("BUDGET_WINDOW_HOURS", "24")),
            carbon_data_source=CarbonDataSource(
                os.getenv("CARBON_DATA_SOURCE", "static")
            ),
            default_region=os.getenv("DEFAULT_REGION", "US"),
            electricity_maps_api_key=os.getenv("ELECTRICITY_MAPS_API_KEY"),
            default_model_size=ModelSize(
                os.getenv("DEFAULT_MODEL_SIZE", "medium")
            ),
            scheduler_weights=SchedulerWeights(
                alpha_carbon=float(os.getenv("SCHEDULER_ALPHA_CARBON", "0.5")),
                beta_latency=float(os.getenv("SCHEDULER_BETA_LATENCY", "0.3")),
                gamma_quality=float(os.getenv("SCHEDULER_GAMMA_QUALITY", "0.2")),
            ),
            early_exit=EarlyExitConfig(
                enabled=os.getenv("EARLY_EXIT_ENABLED", "true").lower() == "true",
                confidence_threshold=float(
                    os.getenv("EARLY_EXIT_THRESHOLD", "0.85")
                ),
            ),
            cache=CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                similarity_threshold=float(
                    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")
                ),
                redis_url=os.getenv("REDIS_URL"),
            ),
            telemetry=TelemetryConfig(
                enabled=os.getenv("TELEMETRY_ENABLED", "true").lower() == "true",
                prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            ),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
        )


# Global default configuration
DEFAULT_CONFIG = CarbonXConfig()
