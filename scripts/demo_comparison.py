"""
Demo script for comparing different plot configurations.
Run with: uv run python scripts/demo_comparison.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demo comparing different axis display modes."""
    print("📊 Demo: Comparing axis display modes")

    df = sns.load_dataset("iris")

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Original axis values (default)
    print("\n1. Creating plot with original axis values (default)...")
    snp.parallelplot(data=df, hue="species", orient="v", alpha=0.6, ax=axes[0])
    axes[0].set_title(
        "Default: Original Axis Values\n(Each axis shows its own range)", fontsize=12
    )

    # Plot 2: Shared y-axis
    print("2. Creating plot with shared y-axis...")
    snp.parallelplot(
        data=df,
        hue="species",
        orient="v",
        sharey=True,
        alpha=0.6,
        ax=axes[1],
    )
    axes[1].set_title(
        "Shared Y-Axis (sharey=True)\n(All axes share 0-8 range)", fontsize=12
    )

    # Plot 3: Using mixed scale data to show the difference
    print("3. Creating plot with high-contrast data...")
    df_mixed = df.copy()
    df_mixed["large_scale"] = df_mixed["sepal_length"] * 100  # Much larger scale

    snp.parallelplot(
        data=df_mixed,
        vars=["sepal_length", "sepal_width", "petal_length", "large_scale"],
        hue="species",
        orient="v",
        alpha=0.6,
        ax=axes[2],
    )
    axes[2].set_title(
        "Mixed Scales (default)\n(Shows utility of original values)", fontsize=12
    )

    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Comparison plot saved to: {output_path}")
    print("\n📝 Key differences:")
    print("   • Default: Each variable shows its true data range")
    print("   • sharey=True: All variables share the same scale")
    print("   • Mixed scales: Original values prevent distortion")


if __name__ == "__main__":
    main()
