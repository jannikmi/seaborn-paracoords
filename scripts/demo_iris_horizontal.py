"""
Demo script for iris dataset with horizontal orientation.
Run with: uv run python scripts/demo_iris_horizontal.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demo with iris dataset - horizontal orientation."""
    print("🌸 Demo: Iris dataset - Horizontal orientation")

    df = sns.load_dataset("iris")

    fig, ax = plt.subplots(figsize=(8, 8))
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="h",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Iris Dataset - Horizontal Parallel Coordinates")
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_iris_horizontal.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
