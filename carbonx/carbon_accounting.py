"""
Carbon Accounting Module

Implements the core carbon emission calculation: C = E × I
Where:
    C = Carbon emissions (gCO2eq)
    E = Energy consumption (kWh)
    I = Carbon intensity (gCO2eq/kWh)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import structlog

logger = structlog.get_logger()


@dataclass
class CarbonMeasurement:
    """A single carbon emission measurement."""
    timestamp: datetime
    energy_kwh: float
    carbon_intensity_gco2_kwh: float
    carbon_gco2: float
    model_used: str
    tokens_generated: int
    latency_ms: float
    tenant_id: Optional[str] = None
    request_id: Optional[str] = None
    
    @property
    def carbon_per_token(self) -> float:
        """Carbon emissions per token generated."""
        if self.tokens_generated == 0:
            return 0.0
        return self.carbon_gco2 / self.tokens_generated


class CarbonAccountant:
    """
    Calculates and tracks carbon emissions for LLM inference.
    
    Implements the formula: C = E × I
    
    Supports multiple energy measurement strategies:
    - Direct measurement via NVML (GPU)
    - Estimated measurement based on execution time
    """
    
    def __init__(
        self,
        default_carbon_intensity: float = 400.0,  # gCO2/kWh (global average)
    ):
        """
        Initialize the carbon accountant.
        
        Args:
            default_carbon_intensity: Default carbon intensity in gCO2/kWh
        """
        self.default_carbon_intensity = default_carbon_intensity
        self._total_emissions = 0.0
        self._measurement_count = 0
        
    def calculate_carbon(
        self,
        energy_kwh: float,
        carbon_intensity_gco2_kwh: Optional[float] = None,
    ) -> float:
        """
        Calculate carbon emissions using C = E × I.
        
        Args:
            energy_kwh: Energy consumption in kWh
            carbon_intensity_gco2_kwh: Carbon intensity in gCO2/kWh
            
        Returns:
            Carbon emissions in gCO2
        """
        intensity = carbon_intensity_gco2_kwh or self.default_carbon_intensity
        carbon = energy_kwh * intensity
        
        logger.debug(
            "carbon_calculated",
            energy_kwh=energy_kwh,
            carbon_intensity=intensity,
            carbon_gco2=carbon,
        )
        
        return carbon
    
    def estimate_energy_from_time(
        self,
        execution_time_seconds: float,
        power_watts: float = 250.0,  # Typical GPU power draw
    ) -> float:
        """
        Estimate energy consumption from execution time.
        
        Uses a calibrated power model based on average GPU power draw.
        
        Args:
            execution_time_seconds: Inference execution time in seconds
            power_watts: Estimated power consumption in watts
            
        Returns:
            Estimated energy consumption in kWh
        """
        # Convert watts-seconds to kWh
        # kWh = (W × s) / (1000 × 3600)
        energy_kwh = (power_watts * execution_time_seconds) / 3_600_000
        
        logger.debug(
            "energy_estimated",
            execution_time_s=execution_time_seconds,
            power_watts=power_watts,
            energy_kwh=energy_kwh,
        )
        
        return energy_kwh
    
    def estimate_energy_per_token(
        self,
        model_energy_per_token: float,
        num_tokens: int,
    ) -> float:
        """
        Estimate energy based on per-token energy consumption.
        
        Args:
            model_energy_per_token: Energy per token for the model (kWh)
            num_tokens: Number of tokens generated
            
        Returns:
            Total energy consumption in kWh
        """
        return model_energy_per_token * num_tokens
    
    def record_measurement(
        self,
        energy_kwh: float,
        carbon_intensity_gco2_kwh: float,
        model_used: str,
        tokens_generated: int,
        latency_ms: float,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> CarbonMeasurement:
        """
        Record a carbon emission measurement.
        
        Args:
            energy_kwh: Energy consumed
            carbon_intensity_gco2_kwh: Carbon intensity at time of inference
            model_used: Name of the model used
            tokens_generated: Number of output tokens
            latency_ms: Total latency in milliseconds
            tenant_id: Optional tenant identifier
            request_id: Optional request identifier
            
        Returns:
            CarbonMeasurement object with all details
        """
        carbon = self.calculate_carbon(energy_kwh, carbon_intensity_gco2_kwh)
        
        measurement = CarbonMeasurement(
            timestamp=datetime.utcnow(),
            energy_kwh=energy_kwh,
            carbon_intensity_gco2_kwh=carbon_intensity_gco2_kwh,
            carbon_gco2=carbon,
            model_used=model_used,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            tenant_id=tenant_id,
            request_id=request_id,
        )
        
        self._total_emissions += carbon
        self._measurement_count += 1
        
        logger.info(
            "measurement_recorded",
            carbon_gco2=carbon,
            model=model_used,
            tokens=tokens_generated,
            tenant_id=tenant_id,
        )
        
        return measurement
    
    @property
    def total_emissions(self) -> float:
        """Total carbon emissions recorded (gCO2)."""
        return self._total_emissions
    
    @property
    def measurement_count(self) -> int:
        """Number of measurements recorded."""
        return self._measurement_count
    
    @property
    def average_emission_per_request(self) -> float:
        """Average carbon emission per request (gCO2)."""
        if self._measurement_count == 0:
            return 0.0
        return self._total_emissions / self._measurement_count
    
    def reset(self):
        """Reset all accumulated measurements."""
        self._total_emissions = 0.0
        self._measurement_count = 0
        logger.info("accountant_reset")
