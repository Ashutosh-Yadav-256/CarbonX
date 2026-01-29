"""
Carbon Intensity Provider Module

Provides real-time and historical carbon intensity data from
various sources including Electricity Maps, WattTime, and static datasets.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import os
import httpx
import structlog

logger = structlog.get_logger()


class CarbonDataSource(str, Enum):
    """Available carbon intensity data sources."""
    STATIC = "static"
    ELECTRICITY_MAPS = "electricity_maps"
    WATTTIME = "watttime"


@dataclass
class CarbonIntensityData:
    """Carbon intensity measurement."""
    timestamp: datetime
    intensity_gco2_kwh: float
    region: str
    source: CarbonDataSource
    is_forecast: bool = False
    confidence: float = 1.0


# Static carbon intensity data by region (gCO2/kWh)
# Source: Various public electricity grid data
STATIC_INTENSITIES = {
    # Low carbon regions
    "NO": 20,    # Norway (hydro)
    "SE": 45,    # Sweden (hydro + nuclear)
    "FR": 60,    # France (nuclear)
    "CA-QC": 30, # Quebec (hydro)
    "IS": 15,    # Iceland (geothermal + hydro)
    
    # Medium carbon regions
    "GB": 250,   # UK
    "DE": 350,   # Germany
    "ES": 200,   # Spain
    "IT": 300,   # Italy
    "JP": 450,   # Japan
    
    # Higher carbon regions
    "US": 400,   # US average
    "US-CA": 250,# California
    "US-TX": 380,# Texas
    "CN": 550,   # China
    "IN": 650,   # India
    "AU": 500,   # Australia
    "PL": 700,   # Poland (coal)
    
    # Default fallback
    "DEFAULT": 400,
}


class CarbonIntensityProvider:
    """
    Provides carbon intensity data from multiple sources.
    
    Supports:
    - Static data (fallback, always available)
    - Electricity Maps API (free tier)
    - WattTime API (free tier)
    """
    
    def __init__(
        self,
        source: Optional[CarbonDataSource] = None,
        region: str = "US",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the carbon intensity provider.
        
        Args:
            source: Data source to use (auto-detected if not specified)
            region: Default region code
            api_key: API key for external services
        """
        self.api_key = api_key or os.getenv("ELECTRICITY_MAPS_API_KEY")
        self.region = region
        
        # Auto-detect source based on API key availability
        if source is not None:
            self.source = source
        elif self.api_key:
            self.source = CarbonDataSource.ELECTRICITY_MAPS
        else:
            self.source = CarbonDataSource.STATIC
        
        # Cache for current intensity
        self._cache: Optional[CarbonIntensityData] = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
        logger.info(
            "carbon_intensity_provider_initialized",
            source=self.source.value,
            region=region,
            has_api_key=bool(self.api_key),
        )
    
    def get_current_intensity(
        self,
        region: Optional[str] = None,
    ) -> CarbonIntensityData:
        """
        Get current carbon intensity for a region.
        
        Args:
            region: Optional region override
            
        Returns:
            CarbonIntensityData with current intensity
        """
        target_region = region or self.region
        
        # Check cache
        if self._is_cache_valid(target_region):
            return self._cache
        
        # Fetch based on source
        try:
            if self.source == CarbonDataSource.ELECTRICITY_MAPS and self.api_key:
                data = self._fetch_electricity_maps(target_region)
            elif self.source == CarbonDataSource.WATTTIME and self.api_key:
                data = self._fetch_watttime(target_region)
            else:
                data = self._get_static(target_region)
        except Exception as e:
            logger.warning(
                "carbon_data_fetch_failed",
                source=self.source.value,
                error=str(e),
            )
            # Fallback to static
            data = self._get_static(target_region)
        
        self._cache = data
        return data
    
    def _is_cache_valid(self, region: str) -> bool:
        """Check if cache is valid."""
        if self._cache is None:
            return False
        if self._cache.region != region:
            return False
        
        age = (datetime.utcnow() - self._cache.timestamp).total_seconds()
        return age < self._cache_ttl_seconds
    
    def _get_static(self, region: str) -> CarbonIntensityData:
        """Get static carbon intensity data."""
        intensity = STATIC_INTENSITIES.get(
            region,
            STATIC_INTENSITIES["DEFAULT"]
        )
        
        return CarbonIntensityData(
            timestamp=datetime.utcnow(),
            intensity_gco2_kwh=float(intensity),
            region=region,
            source=CarbonDataSource.STATIC,
        )
    
    def _fetch_electricity_maps(self, region: str) -> CarbonIntensityData:
        """Fetch from Electricity Maps API."""
        # Note: API URL is electricitymaps.com (not electricitymap.org)
        url = "https://api.electricitymaps.com/v3/carbon-intensity/latest"
        headers = {"auth-token": self.api_key}
        params = {"zone": region}
        
        logger.info("fetching_electricity_maps", region=region)
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        
        intensity = float(data.get("carbonIntensity", 400))
        logger.info("electricity_maps_response", intensity=intensity, region=region)
        
        return CarbonIntensityData(
            timestamp=datetime.utcnow(),
            intensity_gco2_kwh=intensity,
            region=region,
            source=CarbonDataSource.ELECTRICITY_MAPS,
        )
    
    def _fetch_watttime(self, region: str) -> CarbonIntensityData:
        """Fetch from WattTime API."""
        # WattTime requires registration and uses different region codes
        # This is a simplified implementation
        url = "https://api.watttime.org/v3/signal-index"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"region": region}
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        
        # WattTime returns a percentage, convert to approximate gCO2/kWh
        percent = float(data.get("percent", 50))
        intensity = percent * 8  # Rough conversion
        
        return CarbonIntensityData(
            timestamp=datetime.utcnow(),
            intensity_gco2_kwh=intensity,
            region=region,
            source=CarbonDataSource.WATTTIME,
        )
    
    def get_intensity_value(self, region: Optional[str] = None) -> float:
        """Convenience method to get just the intensity value."""
        return self.get_current_intensity(region).intensity_gco2_kwh
    
    def list_available_regions(self) -> list[str]:
        """List regions with static data available."""
        return list(STATIC_INTENSITIES.keys())
    
    def get_greenest_region(self, regions: list[str]) -> tuple[str, float]:
        """
        Find the greenest region from a list.
        
        Args:
            regions: List of region codes to compare
            
        Returns:
            Tuple of (region_code, intensity)
        """
        best_region = None
        best_intensity = float('inf')
        
        for region in regions:
            intensity = self.get_intensity_value(region)
            if intensity < best_intensity:
                best_intensity = intensity
                best_region = region
        
        return (best_region or regions[0], best_intensity)
