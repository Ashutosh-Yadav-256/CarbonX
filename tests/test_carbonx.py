"""
Basic tests for CarbonX framework.
"""

import pytest


class TestCarbonAccounting:
    """Tests for carbon accounting module."""
    
    def test_carbon_calculation(self):
        """Test C = E × I formula."""
        from carbonx.carbon_accounting import CarbonAccountant
        
        accountant = CarbonAccountant(default_carbon_intensity=400.0)
        
        # Test basic calculation
        carbon = accountant.calculate_carbon(
            energy_kwh=0.001,  # 1 Wh
            carbon_intensity_gco2_kwh=400.0,
        )
        
        assert carbon == pytest.approx(0.4)  # 0.001 kWh × 400 gCO2/kWh = 0.4 gCO2
    
    def test_energy_estimation(self):
        """Test energy estimation from time."""
        from carbonx.carbon_accounting import CarbonAccountant
        
        accountant = CarbonAccountant()
        
        # 1 second at 250W
        energy = accountant.estimate_energy_from_time(
            execution_time_seconds=1.0,
            power_watts=250.0,
        )
        
        # Expected: (250W × 1s) / 3600000 = ~6.94e-5 kWh
        assert energy == pytest.approx(250 / 3_600_000, rel=1e-6)


class TestBudgetManager:
    """Tests for carbon budget manager."""
    
    def test_budget_tracking(self):
        """Test budget consumption tracking."""
        from carbonx.budget_manager import CarbonBudgetManager
        from carbonx.carbon_accounting import CarbonMeasurement
        from datetime import datetime
        
        manager = CarbonBudgetManager(default_budget_gco2=100.0)
        
        # Initial state
        state = manager.get_budget_state("test-tenant")
        assert state.remaining_gco2 == 100.0
        assert state.consumption_ratio == 0.0
        
        # Record emission
        measurement = CarbonMeasurement(
            timestamp=datetime.utcnow(),
            energy_kwh=0.001,
            carbon_intensity_gco2_kwh=400,
            carbon_gco2=10.0,
            model_used="test",
            tokens_generated=100,
            latency_ms=500,
        )
        
        manager.record_emission(measurement, "test-tenant")
        
        # Check updated state
        state = manager.get_budget_state("test-tenant")
        assert state.consumed_gco2 == 10.0
        assert state.remaining_gco2 == 90.0


class TestComplexityEstimator:
    """Tests for complexity estimation."""
    
    def test_simple_query(self):
        """Test simple factual query detection."""
        from carbonx.inference.complexity_estimator import (
            ComplexityEstimator,
            ComplexityLevel,
        )
        
        estimator = ComplexityEstimator()
        
        # Simple factual query
        result = estimator.estimate("What is the capital of France?")
        
        # Should be low complexity
        assert result.level == ComplexityLevel.LOW
        assert result.confidence > 0.5
    
    def test_complex_query(self):
        """Test complex reasoning query detection."""
        from carbonx.inference.complexity_estimator import (
            ComplexityEstimator,
            ComplexityLevel,
        )
        
        estimator = ComplexityEstimator()
        
        # Complex reasoning query
        result = estimator.estimate(
            "Explain how machine learning algorithms work and compare "
            "supervised vs unsupervised learning approaches in detail."
        )
        
        # Should be high complexity
        assert result.level in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]


class TestModelPool:
    """Tests for model pool."""
    
    def test_model_registration(self):
        """Test model registration."""
        from carbonx.inference.model_pool import ModelPool
        
        pool = ModelPool()
        
        # Default models should be registered
        models = pool.list_models()
        assert "small" in models
        assert "medium" in models
        assert "large" in models
    
    def test_energy_estimation(self):
        """Test per-model energy estimation."""
        from carbonx.inference.model_pool import ModelPool
        
        pool = ModelPool()
        
        # Larger models should use more energy
        energy_small = pool.estimate_energy("small", 100)
        energy_large = pool.estimate_energy("large", 100)
        
        assert energy_large > energy_small


class TestGreenScheduler:
    """Tests for green scheduler."""
    
    def test_scheduling_decision(self):
        """Test basic scheduling decision."""
        from carbonx.scheduler.green_scheduler import (
            GreenScheduler,
            RequestContext,
            SchedulingAction,
        )
        from carbonx.inference.complexity_estimator import ComplexityLevel
        
        scheduler = GreenScheduler()
        
        context = RequestContext(
            complexity_level=ComplexityLevel.LOW,
            estimated_tokens=50,
            is_urgent=True,
            current_carbon_intensity=400.0,
        )
        
        decision = scheduler.schedule(context)
        
        # Urgent request should execute immediately
        assert decision.action == SchedulingAction.EXECUTE_IMMEDIATELY
        assert decision.recommended_model is not None


class TestDigitalTwin:
    """Tests for digital twin simulator."""
    
    def test_simulation_comparison(self):
        """Test simulation comparison."""
        from carbonx.simulator.digital_twin import DigitalTwin
        
        simulator = DigitalTwin(seed=42)
        
        result = simulator.run_comparison(num_requests=100)
        
        # CarbonX should use less carbon than baseline
        assert result["carbonx_carbon_gco2"] < result["baseline_carbon_gco2"]
        assert result["carbon_reduction_percent"] > 0
        
        # Should have variety in model distribution
        assert len(result["model_distribution"]) > 1
