"""
Demo: Orientations - Vertical and Horizontal Parallel Coordinates
Run with: uv run python scripts/demo_orientations.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate both vertical and horizontal orientations with iris dataset."""
    print("🌸 Demo: Parallel Coordinates Orientations")

    df = sns.load_dataset("iris")

    # Create figure with both orientations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Vertical orientation
    print("  • Creating vertical orientation plot...")
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="v",
        alpha=0.7,
        ax=ax1,
    )
    ax1.set_title("Vertical Orientation (orient='v')", fontsize=12, fontweight="bold")

    # Horizontal orientation
    print("  • Creating horizontal orientation plot...")
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="h",
        alpha=0.7,
        ax=ax2,
    )
    ax2.set_title("Horizontal Orientation (orient='h')", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Iris Dataset - Orientation Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_orientations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
