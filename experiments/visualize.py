"""CarbonX Visualization Scripts"""

import json
from pathlib import Path
from datetime import datetime

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Run: pip install matplotlib")


def set_paper_style():
    """Set publication-quality plot style."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_carbon_comparison(results: dict, output_path: str = "figures/carbon_comparison.png"):
    """Plot carbon comparison bar chart."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    set_paper_style()
    
    strategies = list(results.keys())
    carbons = [results[s]["carbon_mean"] for s in strategies]
    errors = [results[s]["carbon_std"] for s in strategies]
    
    # Colors
    colors = {
        "carbonx": "#22c55e",  # Green
        "always_large": "#ef4444",  # Red
        "always_small": "#3b82f6",  # Blue
        "random": "#f59e0b",  # Yellow
    }
    
    bar_colors = [colors.get(s, "#888888") for s in strategies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(strategies, carbons, yerr=errors, color=bar_colors, 
                  capsize=5, edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Carbon Emissions (gCO₂)")
    ax.set_title("Carbon Emissions by Inference Strategy")
    
    # Add value labels
    for bar, carbon in zip(bars, carbons):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{carbon:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Highlight CarbonX
    if "carbonx" in strategies:
        idx = strategies.index("carbonx")
        bars[idx].set_hatch('//')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_model_distribution(distribution: dict, output_path: str = "figures/model_distribution.png"):
    """Plot model distribution pie chart."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    set_paper_style()
    
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    
    colors = {
        "small": "#22c55e",
        "medium": "#3b82f6",
        "large": "#f59e0b",
        "cache": "#a855f7",
    }
    
    pie_colors = [colors.get(l, "#888888") for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05 if l == "small" else 0 for l in labels],
    )
    
    ax.set_title("CarbonX Model Distribution")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_latency_vs_carbon(results: dict, output_path: str = "figures/latency_carbon_tradeoff.png"):
    """Plot latency vs carbon tradeoff."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        "carbonx": "#22c55e",
        "always_large": "#ef4444",
        "always_small": "#3b82f6",
        "random": "#f59e0b",
    }
    
    for strategy, data in results.items():
        ax.scatter(
            data["latency_mean"],
            data["carbon_mean"],
            s=200,
            c=colors.get(strategy, "#888888"),
            label=strategy,
            edgecolors='black',
            linewidth=1.5,
            zorder=5,
        )
        
        # Add error bars
        ax.errorbar(
            data["latency_mean"],
            data["carbon_mean"],
            xerr=data["latency_std"],
            yerr=data["carbon_std"],
            fmt='none',
            c=colors.get(strategy, "#888888"),
            capsize=5,
            zorder=4,
        )
    
    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Carbon Emissions (gCO₂)")
    ax.set_title("Latency vs Carbon Tradeoff")
    ax.legend(loc='upper right')
    
    # Add Pareto frontier annotation
    ax.annotate(
        'Lower is better →',
        xy=(0.95, 0.05),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        style='italic',
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_accuracy_comparison(results: dict, output_path: str = "figures/accuracy_comparison.png"):
    """Plot accuracy comparison."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    set_paper_style()
    
    strategies = list(results.keys())
    accuracies = [results[s]["accuracy_mean"] * 100 for s in strategies]
    errors = [results[s]["accuracy_std"] * 100 for s in strategies]
    
    colors = {
        "carbonx": "#22c55e",
        "always_large": "#ef4444",
        "always_small": "#3b82f6",
        "random": "#f59e0b",
    }
    
    bar_colors = [colors.get(s, "#888888") for s in strategies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(strategies, accuracies, yerr=errors, color=bar_colors,
                  capsize=5, edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task Accuracy by Strategy")
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_all_figures(results_path: str = None):
    """Generate all publication figures."""
    print("Generating Publication Figures")
    print("=" * 50)
    
    # Load results or use sample data
    if results_path and Path(results_path).exists():
        with open(results_path) as f:
            data = json.load(f)
    else:
        # Sample data for demonstration
        data = {
            "carbonx": {
                "carbon_mean": 0.0120,
                "carbon_std": 0.002,
                "latency_mean": 450,
                "latency_std": 50,
                "accuracy_mean": 0.82,
                "accuracy_std": 0.03,
                "model_distribution": {"small": 35, "medium": 40, "large": 15, "cache": 10},
            },
            "always_large": {
                "carbon_mean": 0.0480,
                "carbon_std": 0.005,
                "latency_mean": 800,
                "latency_std": 80,
                "accuracy_mean": 0.95,
                "accuracy_std": 0.01,
                "model_distribution": {"large": 100},
            },
            "always_small": {
                "carbon_mean": 0.0080,
                "carbon_std": 0.001,
                "latency_mean": 200,
                "latency_std": 20,
                "accuracy_mean": 0.60,
                "accuracy_std": 0.05,
                "model_distribution": {"small": 100},
            },
            "random": {
                "carbon_mean": 0.0280,
                "carbon_std": 0.008,
                "latency_mean": 500,
                "latency_std": 100,
                "accuracy_mean": 0.75,
                "accuracy_std": 0.08,
                "model_distribution": {"small": 33, "medium": 34, "large": 33},
            },
        }
    
    output_dir = Path("experiments/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    plot_carbon_comparison(data, str(output_dir / "carbon_comparison.png"))
    plot_model_distribution(data["carbonx"]["model_distribution"], str(output_dir / "model_distribution.png"))
    plot_latency_vs_carbon(data, str(output_dir / "latency_carbon_tradeoff.png"))
    plot_accuracy_comparison(data, str(output_dir / "accuracy_comparison.png"))
    
    print("\nAll figures generated!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    generate_all_figures()
