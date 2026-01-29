"""
Energy Telemetry Module

Measures energy consumption during inference using:
- NVIDIA NVML (GPU)
- Intel RAPL (CPU on Linux)
- Time-based estimation (fallback)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import time
import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class EnergyMeasurement:
    """An energy consumption measurement."""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    energy_kwh: float
    power_watts: float
    measurement_method: str
    gpu_available: bool = False


class EnergyTelemetry:
    """
    Measures energy consumption for inference workloads.
    
    Supports:
    - NVIDIA GPU via pynvml
    - CPU estimation via psutil
    - Time-based estimation (fallback)
    """
    
    def __init__(
        self,
        default_gpu_power_watts: float = 250.0,
        default_cpu_power_watts: float = 65.0,
        use_gpu: bool = True,
    ):
        """
        Initialize energy telemetry.
        
        Args:
            default_gpu_power_watts: Default GPU power draw
            default_cpu_power_watts: Default CPU power draw
            use_gpu: Whether to attempt GPU measurement
        """
        self.default_gpu_power = default_gpu_power_watts
        self.default_cpu_power = default_cpu_power_watts
        self.use_gpu = use_gpu
        
        self._nvml_available = False
        self._nvml_handle = None
        
        # Try to initialize NVML
        if use_gpu:
            self._init_nvml()
        
        logger.info(
            "energy_telemetry_initialized",
            nvml_available=self._nvml_available,
            default_gpu_power=default_gpu_power_watts,
        )
    
    def _init_nvml(self) -> None:
        """Initialize NVIDIA NVML for GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            # Get handle for first GPU
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            logger.info("nvml_initialized", gpu_name=name)
            
        except ImportError:
            logger.debug("pynvml_not_installed")
        except Exception as e:
            logger.debug("nvml_init_failed", error=str(e))
    
    def get_gpu_power(self) -> Optional[float]:
        """Get current GPU power consumption in watts."""
        if not self._nvml_available:
            return None
        
        try:
            import pynvml
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            return power_mw / 1000.0  # Convert mW to W
        except Exception as e:
            logger.debug("gpu_power_read_failed", error=str(e))
            return None
    
    def get_cpu_power_estimate(self) -> float:
        """Estimate CPU power based on utilization."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Scale power linearly with utilization
        return self.default_cpu_power * (0.3 + 0.7 * cpu_percent / 100)
    
    def measure_energy(
        self,
        duration_seconds: float,
        include_cpu: bool = True,
    ) -> EnergyMeasurement:
        """
        Measure energy consumption for a completed operation.
        
        Args:
            duration_seconds: Duration of the operation
            include_cpu: Whether to include CPU power estimate
            
        Returns:
            EnergyMeasurement with consumption data
        """
        end_time = datetime.utcnow()
        start_time = datetime.utcnow()  # Approximation
        
        gpu_power = None
        cpu_power = 0.0
        method = "estimated"
        gpu_available = False
        
        # Try GPU measurement
        if self.use_gpu and self._nvml_available:
            gpu_power = self.get_gpu_power()
            if gpu_power is not None:
                method = "nvml"
                gpu_available = True
        
        # Use default GPU power if not measured
        if gpu_power is None:
            gpu_power = self.default_gpu_power
        
        # Add CPU estimate
        if include_cpu:
            cpu_power = self.get_cpu_power_estimate()
        
        total_power = gpu_power + cpu_power
        
        # Calculate energy: E = P × t
        # Convert W × s to kWh: divide by 3,600,000
        energy_kwh = (total_power * duration_seconds) / 3_600_000
        
        measurement = EnergyMeasurement(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            energy_kwh=energy_kwh,
            power_watts=total_power,
            measurement_method=method,
            gpu_available=gpu_available,
        )
        
        logger.debug(
            "energy_measured",
            duration_s=duration_seconds,
            power_w=total_power,
            energy_kwh=energy_kwh,
            method=method,
        )
        
        return measurement
    
    def estimate_inference_energy(
        self,
        tokens: int,
        model_size: str = "medium",
        time_per_token_ms: float = 50.0,
    ) -> float:
        """
        Estimate energy for an inference operation.
        
        Args:
            tokens: Number of tokens to generate
            model_size: Size of model (small, medium, large)
            time_per_token_ms: Estimated time per token
            
        Returns:
            Estimated energy in kWh
        """
        # Adjust power based on model size
        power_multiplier = {
            "small": 0.5,
            "medium": 1.0,
            "large": 1.5,
        }.get(model_size, 1.0)
        
        power = self.default_gpu_power * power_multiplier
        duration_seconds = (tokens * time_per_token_ms) / 1000
        
        energy_kwh = (power * duration_seconds) / 3_600_000
        
        return energy_kwh
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                logger.info("nvml_shutdown")
            except Exception:
                pass


class TimedEnergyContext:
    """Context manager for timing and measuring energy."""
    
    def __init__(
        self,
        telemetry: EnergyTelemetry,
        include_cpu: bool = True,
    ):
        self.telemetry = telemetry
        self.include_cpu = include_cpu
        self.start_time = 0.0
        self.measurement: Optional[EnergyMeasurement] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.measurement = self.telemetry.measure_energy(
            duration_seconds=duration,
            include_cpu=self.include_cpu,
        )
        return False
