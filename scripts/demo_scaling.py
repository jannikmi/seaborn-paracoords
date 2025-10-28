"""
Demo: Scaling Modes - Independent vs Shared Axis Scaling
Run with: uv run python scripts/demo_scaling.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate different scaling modes for parallel coordinates."""
    print("📊 Demo: Scaling Modes")

    # Create data with vastly different ranges to show scaling difference
    df_mixed = pd.DataFrame(
        {
            "small": [0.1, 0.2, 0.3, 0.4, 0.5],
            "medium": [10, 20, 30, 40, 50],
            "large": [1000, 2000, 3000, 4000, 5000],
            "category": ["A", "B", "C", "D", "E"],
        }
    )

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Demo 1: Independent scaling (default) - synthetic data
    print("  • Creating plot with independent scaling (synthetic data)...")
    snp.parallelplot(
        data=df_mixed,
        vars=["small", "medium", "large"],
        hue="category",
        alpha=0.8,
        linewidth=2,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "Independent Scaling (default)\nEach variable uses its own range",
        fontsize=11,
        fontweight="bold",
    )

    # Demo 2: Shared y-axis - synthetic data
    print("  • Creating plot with shared y-axis (synthetic data)...")
    snp.parallelplot(
        data=df_mixed,
        vars=["small", "medium", "large"],
        hue="category",
        sharey=True,
        alpha=0.8,
        linewidth=2,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(
        "Shared Y-Axis (sharey=True)\nAll variables share 0-5000 range",
        fontsize=11,
        fontweight="bold",
    )

    # Demo 3: Independent scaling - iris dataset
    print("  • Creating plot with independent scaling (iris data)...")
    df_iris = sns.load_dataset("iris")
    snp.parallelplot(
        data=df_iris,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        alpha=0.6,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(
        "Iris Dataset - Independent Scaling\nEach feature scaled independently",
        fontsize=11,
        fontweight="bold",
    )

    # Demo 4: Shared y-axis - iris dataset
    print("  • Creating plot with shared y-axis (iris data)...")
    snp.parallelplot(
        data=df_iris,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        sharey=True,
        alpha=0.6,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Iris Dataset - Shared Y-Axis\nAll features share common scale",
        fontsize=11,
        fontweight="bold",
    )

    plt.suptitle(
        "Scaling Modes Comparison\n(Independent vs Shared Axis Scaling)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")
    print("\n📝 Key differences:")
    print(
        "  • Independent: Each variable shows its true data range (preserves original values)"
    )
    print(
        "  • Shared: All variables share the same scale (useful for comparing magnitudes)"
    )


if __name__ == "__main__":
    main()
