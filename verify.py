"""Quick verification script for CarbonX."""
print("Testing CarbonX imports...")

try:
    from carbonx.config import CarbonXConfig
    print("[OK] config module")
except Exception as e:
    print(f"[FAIL] config module: {e}")

try:
    from carbonx.carbon_accounting import CarbonAccountant
    print("[OK] carbon_accounting module")
except Exception as e:
    print(f"[FAIL] carbon_accounting module: {e}")

try:
    from carbonx.budget_manager import CarbonBudgetManager
    print("[OK] budget_manager module")
except Exception as e:
    print(f"[FAIL] budget_manager module: {e}")

try:
    from carbonx.inference.complexity_estimator import ComplexityEstimator
    print("[OK] complexity_estimator module")
except Exception as e:
    print(f"[FAIL] complexity_estimator module: {e}")

try:
    from carbonx.inference.model_pool import ModelPool
    print("[OK] model_pool module")
except Exception as e:
    print(f"[FAIL] model_pool module: {e}")

try:
    from carbonx.scheduler.green_scheduler import GreenScheduler
    print("[OK] green_scheduler module")
except Exception as e:
    print(f"[FAIL] green_scheduler module: {e}")

try:
    from carbonx.simulator.digital_twin import DigitalTwin
    print("[OK] digital_twin module")
except Exception as e:
    print(f"[FAIL] digital_twin module: {e}")

print("\n--- Running Simulation ---")
try:
    sim = DigitalTwin(seed=42)
    result = sim.run_comparison(num_requests=50)
    print(f"Requests: {result['total_requests']}")
    print(f"Baseline: {result['baseline_carbon_gco2']:.4f} gCO2")
    print(f"CarbonX:  {result['carbonx_carbon_gco2']:.4f} gCO2")
    print(f"Reduction: {result['carbon_reduction_percent']:.1f}%")
    print("\n[OK] Simulation successful!")
except Exception as e:
    print(f"[FAIL] Simulation failed: {e}")

print("\n--- All tests complete! ---")
