"""CarbonX CLI Module"""

import argparse
import json
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CarbonX: Carbon-First LLM Inference Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  carbonx serve                    Start the API server
  carbonx simulate --requests 100  Run simulation
  carbonx inference "Hello"        Run inference
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--requests", type=int, default=1000, help="Number of requests")
    sim_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Inference command
    inf_parser = subparsers.add_parser("inference", help="Run inference")
    inf_parser.add_argument("prompt", help="Input prompt")
    inf_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    inf_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Budget command
    budget_parser = subparsers.add_parser("budget", help="Check budget")
    budget_parser.add_argument("--tenant", default="default", help="Tenant ID")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "carbonx.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    
    elif args.command == "simulate":
        from carbonx.simulator import DigitalTwin
        
        simulator = DigitalTwin()
        result = simulator.run_comparison(num_requests=args.requests)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nCarbonX Simulation Results")
            print(f"{'=' * 50}")
            print(f"  Requests:        {result['total_requests']}")
            print(f"  Baseline Carbon: {result['baseline_carbon_gco2']:.4f} gCO2")
            print(f"  CarbonX Carbon:  {result['carbonx_carbon_gco2']:.4f} gCO2")
            print(f"  Reduction:       {result['carbon_reduction_percent']:.1f}%")
            print(f"  Model Dist:      {result['model_distribution']}")
    
    elif args.command == "inference":
        from carbonx import CarbonX
        
        cx = CarbonX()
        response = cx.inference(args.prompt, max_tokens=args.max_tokens)
        
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(f"\nCarbonX Inference")
            print(f"{'=' * 50}")
            print(f"Response: {response.text}")
            print(f"\nModel: {response.model_used}")
            print(f"Tokens: {response.tokens_generated}")
            print(f"Carbon: {response.carbon_gco2:.6f} gCO2")
            print(f"Latency: {response.latency_ms:.1f} ms")
    
    elif args.command == "budget":
        from carbonx import CarbonX
        
        cx = CarbonX(tenant_id=args.tenant)
        state = cx.budget_state
        
        print(f"\nCarbon Budget Status")
        print(f"{'=' * 50}")
        print(f"  Tenant:    {state.tenant_id}")
        print(f"  Budget:    {state.budget_gco2:.2f} gCO2")
        print(f"  Consumed:  {state.consumed_gco2:.2f} gCO2")
        print(f"  Remaining: {state.remaining_gco2:.2f} gCO2")
        print(f"  Status:    {state.status.value}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
