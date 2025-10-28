"""
Demo: Multiple Datasets - Tips, Flights, and Auto Variable Selection
Run with: uv run python scripts/demo_datasets.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate parallel coordinates with multiple datasets."""
    print("📊 Demo: Multiple Datasets")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Demo 1: Tips dataset
    print("  • Loading tips dataset...")
    df_tips = sns.load_dataset("tips")
    snp.parallelplot(
        data=df_tips,
        vars=["total_bill", "tip", "size"],
        hue="time",
        alpha=0.6,
        ax=axes[0],
    )
    axes[0].set_title("Tips Dataset\n(with hue='time')", fontsize=11, fontweight="bold")

    # Demo 2: Flights dataset sample
    print("  • Loading flights dataset...")
    df_flights = sns.load_dataset("flights")
    df_sample = df_flights.sample(n=100, random_state=42)
    snp.parallelplot(
        data=df_sample,
        vars=["year", "passengers"],
        alpha=0.7,
        linewidth=1.2,
        ax=axes[1],
    )
    axes[1].set_title(
        "Flights Dataset\n(100 random samples)", fontsize=11, fontweight="bold"
    )

    # Demo 3: Auto variable selection (mixed data types)
    print("  • Creating mixed dataset for auto-selection...")
    df_mixed = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [10, 20, 30, 40, 50],
            "numeric3": [100, 200, 300, 400, 500],
            "text_col": ["A", "B", "C", "D", "E"],
            "bool_col": [True, False, True, False, True],
        }
    )
    snp.parallelplot(data=df_mixed, alpha=0.8, linewidth=2, ax=axes[2])
    axes[2].set_title(
        "Auto Variable Selection\n(only numeric columns)",
        fontsize=11,
        fontweight="bold",
    )

    plt.suptitle("Multiple Datasets Demo", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_datasets.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
