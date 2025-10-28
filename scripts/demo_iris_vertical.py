"""
Demo script for iris dataset with vertical orientation.
Run with: uv run python scripts/demo_iris_vertical.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demo with iris dataset - vertical orientation."""
    print("🌸 Demo: Iris dataset - Vertical orientation")

    df = sns.load_dataset("iris")

    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="v",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Iris Dataset - Vertical Parallel Coordinates")
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/iris_vertical_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
