"""
Demand Shaper Module

Implements predictive demand shaping to defer non-urgent requests
to low-carbon windows.

Minimizes: ∫ E(t) · I(t) dt
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import structlog

logger = structlog.get_logger()


@dataclass
class CarbonForecast:
    """Forecasted carbon intensity for a time window."""
    timestamp: datetime
    intensity_gco2_kwh: float
    confidence: float = 0.8


@dataclass
class DemandShapingDecision:
    """Result of demand shaping analysis."""
    should_defer: bool
    optimal_time: Optional[datetime] = None
    current_intensity: float = 0.0
    forecasted_intensity: float = 0.0
    expected_savings_percent: float = 0.0
    reason: str = ""


class DemandShaper:
    """
    Shapes demand by identifying optimal execution windows.
    
    Uses carbon intensity forecasts to recommend deferring
    non-urgent requests to greener time windows.
    """
    
    def __init__(
        self,
        defer_threshold_ratio: float = 0.7,
        max_defer_hours: float = 4.0,
        forecast_hours: int = 24,
    ):
        """
        Initialize the demand shaper.
        
        Args:
            defer_threshold_ratio: Defer if forecast intensity is this ratio of current
            max_defer_hours: Maximum hours to defer a request
            forecast_hours: Hours of forecast to consider
        """
        self.defer_threshold_ratio = defer_threshold_ratio
        self.max_defer_hours = max_defer_hours
        self.forecast_hours = forecast_hours
        
        # Simple in-memory forecast (in production, fetch from API)
        self._forecasts: list[CarbonForecast] = []
        
        logger.info(
            "demand_shaper_initialized",
            defer_threshold=defer_threshold_ratio,
            max_defer_hours=max_defer_hours,
        )
    
    def update_forecasts(self, forecasts: list[CarbonForecast]) -> None:
        """Update the carbon intensity forecasts."""
        self._forecasts = sorted(forecasts, key=lambda f: f.timestamp)
        logger.info("forecasts_updated", count=len(forecasts))
    
    def generate_synthetic_forecasts(
        self,
        current_intensity: float,
        hours: int = 24,
    ) -> list[CarbonForecast]:
        """
        Generate synthetic forecasts for testing.
        
        Creates a realistic daily pattern with:
        - Lower intensity during night/early morning
        - Higher intensity during peak hours
        - Some random variation
        """
        forecasts = []
        now = datetime.utcnow()
        
        for h in range(hours):
            timestamp = now + timedelta(hours=h)
            hour_of_day = timestamp.hour
            
            # Daily pattern: lower at night, higher during day
            if 6 <= hour_of_day <= 9:  # Morning ramp-up
                factor = 1.0 + (hour_of_day - 6) * 0.1
            elif 10 <= hour_of_day <= 17:  # Peak hours
                factor = 1.3
            elif 18 <= hour_of_day <= 21:  # Evening decline
                factor = 1.2 - (hour_of_day - 18) * 0.1
            else:  # Night - lowest
                factor = 0.7
            
            intensity = current_intensity * factor
            confidence = max(0.5, 0.95 - (h * 0.02))  # Confidence decreases with time
            
            forecasts.append(CarbonForecast(
                timestamp=timestamp,
                intensity_gco2_kwh=intensity,
                confidence=confidence,
            ))
        
        return forecasts
    
    def analyze_deferral(
        self,
        current_intensity: float,
        is_urgent: bool = True,
        max_defer_hours: Optional[float] = None,
    ) -> DemandShapingDecision:
        """
        Analyze whether to defer a request based on carbon forecasts.
        
        Args:
            current_intensity: Current carbon intensity (gCO2/kWh)
            is_urgent: Whether the request is time-sensitive
            max_defer_hours: Override max deferral time
            
        Returns:
            DemandShapingDecision with recommendation
        """
        # Urgent requests should not be deferred
        if is_urgent:
            return DemandShapingDecision(
                should_defer=False,
                current_intensity=current_intensity,
                reason="Request marked as urgent",
            )
        
        max_defer = max_defer_hours or self.max_defer_hours
        
        # Generate forecasts if none available
        if not self._forecasts:
            self._forecasts = self.generate_synthetic_forecasts(
                current_intensity,
                hours=self.forecast_hours,
            )
        
        # Find the optimal window
        now = datetime.utcnow()
        defer_deadline = now + timedelta(hours=max_defer)
        
        best_time = None
        best_intensity = current_intensity
        
        for forecast in self._forecasts:
            if forecast.timestamp > defer_deadline:
                break
            
            if forecast.intensity_gco2_kwh < best_intensity:
                best_intensity = forecast.intensity_gco2_kwh
                best_time = forecast.timestamp
        
        # Calculate if deferral is worthwhile
        if best_time and best_intensity < current_intensity * self.defer_threshold_ratio:
            savings = (current_intensity - best_intensity) / current_intensity * 100
            
            decision = DemandShapingDecision(
                should_defer=True,
                optimal_time=best_time,
                current_intensity=current_intensity,
                forecasted_intensity=best_intensity,
                expected_savings_percent=savings,
                reason=f"Lower carbon intensity ({best_intensity:.0f} gCO2/kWh) "
                       f"forecasted at {best_time.strftime('%H:%M')} UTC",
            )
            
            logger.info(
                "deferral_recommended",
                current=current_intensity,
                forecasted=best_intensity,
                optimal_time=best_time.isoformat(),
                savings_percent=savings,
            )
            
            return decision
        
        return DemandShapingDecision(
            should_defer=False,
            current_intensity=current_intensity,
            forecasted_intensity=best_intensity,
            reason="No significantly better window found within deferral limit",
        )
    
    def find_greenest_window(
        self,
        duration_hours: float = 1.0,
    ) -> Optional[tuple[datetime, float]]:
        """
        Find the greenest time window for batch processing.
        
        Args:
            duration_hours: Required processing duration
            
        Returns:
            Tuple of (start_time, average_intensity) or None
        """
        if not self._forecasts:
            return None
        
        # Sliding window to find lowest average intensity
        window_size = int(duration_hours)
        best_start = None
        best_avg = float('inf')
        
        for i in range(len(self._forecasts) - window_size + 1):
            window = self._forecasts[i:i + window_size]
            avg_intensity = sum(f.intensity_gco2_kwh for f in window) / len(window)
            
            if avg_intensity < best_avg:
                best_avg = avg_intensity
                best_start = window[0].timestamp
        
        if best_start:
            logger.info(
                "greenest_window_found",
                start_time=best_start.isoformat(),
                avg_intensity=best_avg,
                duration_hours=duration_hours,
            )
        
        return (best_start, best_avg) if best_start else None
