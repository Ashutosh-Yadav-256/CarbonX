"""
Prometheus Exporter Module

Exports CarbonX metrics in Prometheus format for monitoring.
"""

from typing import Optional
import structlog

logger = structlog.get_logger()

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics export disabled")


class PrometheusExporter:
    """
    Exports CarbonX metrics for Prometheus scraping.
    
    Metrics exported:
    - carbonx_requests_total: Total inference requests
    - carbonx_tokens_total: Total tokens generated
    - carbonx_carbon_gco2_total: Total carbon emissions
    - carbonx_energy_kwh_total: Total energy consumed
    - carbonx_latency_seconds: Request latency histogram
    - carbonx_cache_hits_total: Cache hits
    - carbonx_budget_remaining_gco2: Current budget remaining
    """
    
    def __init__(self, prefix: str = "carbonx"):
        """
        Initialize the Prometheus exporter.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self._initialized = False
        
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
        
        logger.info(
            "prometheus_exporter_initialized",
            available=PROMETHEUS_AVAILABLE,
        )
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Counters
        self.requests_total = Counter(
            f'{self.prefix}_requests_total',
            'Total number of inference requests',
            ['model', 'tenant']
        )
        
        self.tokens_total = Counter(
            f'{self.prefix}_tokens_total',
            'Total tokens generated',
            ['model']
        )
        
        self.carbon_total = Counter(
            f'{self.prefix}_carbon_gco2_total',
            'Total carbon emissions in gCO2',
            ['model']
        )
        
        self.energy_total = Counter(
            f'{self.prefix}_energy_kwh_total',
            'Total energy consumption in kWh',
            ['model']
        )
        
        self.cache_hits = Counter(
            f'{self.prefix}_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            f'{self.prefix}_cache_misses_total',
            'Total cache misses'
        )
        
        # Gauges
        self.budget_remaining = Gauge(
            f'{self.prefix}_budget_remaining_gco2',
            'Remaining carbon budget in gCO2',
            ['tenant']
        )
        
        self.carbon_intensity = Gauge(
            f'{self.prefix}_carbon_intensity_gco2_kwh',
            'Current carbon intensity',
            ['region']
        )
        
        self.queue_size = Gauge(
            f'{self.prefix}_defer_queue_size',
            'Number of deferred requests'
        )
        
        # Histograms
        self.latency = Histogram(
            f'{self.prefix}_latency_seconds',
            'Request latency in seconds',
            ['model'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self._initialized = True
    
    def record_request(
        self,
        model: str,
        tenant: str = "default",
        tokens: int = 0,
        carbon_gco2: float = 0.0,
        energy_kwh: float = 0.0,
        latency_seconds: float = 0.0,
    ) -> None:
        """
        Record metrics for a completed request.
        
        Args:
            model: Model used
            tenant: Tenant ID
            tokens: Tokens generated
            carbon_gco2: Carbon emitted
            energy_kwh: Energy consumed
            latency_seconds: Request latency
        """
        if not self._initialized:
            return
        
        self.requests_total.labels(model=model, tenant=tenant).inc()
        self.tokens_total.labels(model=model).inc(tokens)
        self.carbon_total.labels(model=model).inc(carbon_gco2)
        self.energy_total.labels(model=model).inc(energy_kwh)
        self.latency.labels(model=model).observe(latency_seconds)
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        if self._initialized:
            self.cache_hits.inc()
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        if self._initialized:
            self.cache_misses.inc()
    
    def set_budget_remaining(self, tenant: str, remaining_gco2: float) -> None:
        """Update budget remaining gauge."""
        if self._initialized:
            self.budget_remaining.labels(tenant=tenant).set(remaining_gco2)
    
    def set_carbon_intensity(self, region: str, intensity: float) -> None:
        """Update carbon intensity gauge."""
        if self._initialized:
            self.carbon_intensity.labels(region=region).set(intensity)
    
    def set_queue_size(self, size: int) -> None:
        """Update defer queue size."""
        if self._initialized:
            self.queue_size.set(size)
    
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics as bytes for HTTP response
        """
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus client not available\n"
        
        return generate_latest()
    
    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        if PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain"
