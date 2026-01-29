"""
Telemetry Collector Module

Aggregates runtime metrics for carbon accounting and monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
import threading
import structlog

from carbonx.carbon_accounting import CarbonMeasurement

logger = structlog.get_logger()


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""
    window_start: datetime
    window_end: datetime
    total_requests: int
    total_tokens: int
    total_energy_kwh: float
    total_carbon_gco2: float
    avg_latency_ms: float
    model_distribution: dict[str, int]
    cache_hit_rate: float
    
    def to_dict(self) -> dict:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_energy_kwh": self.total_energy_kwh,
            "total_carbon_gco2": self.total_carbon_gco2,
            "avg_latency_ms": self.avg_latency_ms,
            "model_distribution": self.model_distribution,
            "cache_hit_rate": self.cache_hit_rate,
        }


class TelemetryCollector:
    """
    Collects and aggregates telemetry data for CarbonX.
    
    Tracks:
    - Request counts and latencies
    - Energy consumption
    - Carbon emissions
    - Model usage distribution
    - Cache effectiveness
    """
    
    def __init__(
        self,
        aggregation_window_minutes: int = 5,
        max_history_hours: int = 24,
    ):
        """
        Initialize the telemetry collector.
        
        Args:
            aggregation_window_minutes: Window for metric aggregation
            max_history_hours: How long to keep history
        """
        self.aggregation_window = timedelta(minutes=aggregation_window_minutes)
        self.max_history = timedelta(hours=max_history_hours)
        
        # Measurements storage
        self._measurements: list[CarbonMeasurement] = []
        self._lock = threading.RLock()
        
        # Running totals
        self._total_requests = 0
        self._total_tokens = 0
        self._total_energy_kwh = 0.0
        self._total_carbon_gco2 = 0.0
        self._total_latency_ms = 0.0
        
        # Model usage counts
        self._model_counts: dict[str, int] = defaultdict(int)
        
        # Cache stats
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(
            "telemetry_collector_initialized",
            aggregation_window_min=aggregation_window_minutes,
        )
    
    def record(self, measurement: CarbonMeasurement) -> None:
        """
        Record a measurement.
        
        Args:
            measurement: The carbon measurement to record
        """
        with self._lock:
            self._measurements.append(measurement)
            
            # Update totals
            self._total_requests += 1
            self._total_tokens += measurement.tokens_generated
            self._total_energy_kwh += measurement.energy_kwh
            self._total_carbon_gco2 += measurement.carbon_gco2
            self._total_latency_ms += measurement.latency_ms
            self._model_counts[measurement.model_used] += 1
            
            # Prune old measurements
            self._prune_old()
        
        logger.debug(
            "measurement_recorded",
            model=measurement.model_used,
            carbon=measurement.carbon_gco2,
        )
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1
    
    def _prune_old(self) -> None:
        """Remove measurements older than max_history."""
        cutoff = datetime.utcnow() - self.max_history
        self._measurements = [
            m for m in self._measurements
            if m.timestamp >= cutoff
        ]
    
    def get_current_metrics(self) -> AggregatedMetrics:
        """Get aggregated metrics for the current window."""
        now = datetime.utcnow()
        window_start = now - self.aggregation_window
        
        with self._lock:
            window_measurements = [
                m for m in self._measurements
                if m.timestamp >= window_start
            ]
            
            if not window_measurements:
                return AggregatedMetrics(
                    window_start=window_start,
                    window_end=now,
                    total_requests=0,
                    total_tokens=0,
                    total_energy_kwh=0.0,
                    total_carbon_gco2=0.0,
                    avg_latency_ms=0.0,
                    model_distribution={},
                    cache_hit_rate=0.0,
                )
            
            total_requests = len(window_measurements)
            total_tokens = sum(m.tokens_generated for m in window_measurements)
            total_energy = sum(m.energy_kwh for m in window_measurements)
            total_carbon = sum(m.carbon_gco2 for m in window_measurements)
            avg_latency = sum(m.latency_ms for m in window_measurements) / total_requests
            
            model_dist = defaultdict(int)
            for m in window_measurements:
                model_dist[m.model_used] += 1
            
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0
        
        return AggregatedMetrics(
            window_start=window_start,
            window_end=now,
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_energy_kwh=total_energy,
            total_carbon_gco2=total_carbon,
            avg_latency_ms=avg_latency,
            model_distribution=dict(model_dist),
            cache_hit_rate=cache_hit_rate,
        )
    
    def get_totals(self) -> dict:
        """Get all-time totals."""
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0 else 0.0
            )
            
            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_energy_kwh": self._total_energy_kwh,
                "total_carbon_gco2": self._total_carbon_gco2,
                "avg_latency_ms": avg_latency,
                "model_distribution": dict(self._model_counts),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }
    
    def get_carbon_rate(self) -> float:
        """Get current carbon emission rate (gCO2/minute)."""
        metrics = self.get_current_metrics()
        window_minutes = self.aggregation_window.total_seconds() / 60
        
        if window_minutes <= 0:
            return 0.0
        
        return metrics.total_carbon_gco2 / window_minutes
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._measurements.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._total_energy_kwh = 0.0
            self._total_carbon_gco2 = 0.0
            self._total_latency_ms = 0.0
            self._model_counts.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        
        logger.info("telemetry_reset")
