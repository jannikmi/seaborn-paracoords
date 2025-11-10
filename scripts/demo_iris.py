"""
Demo: Iris Dataset - Classic Parallel Coordinates Example
Run with: uv run python scripts/demo_iris.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate parallel coordinates with the classic iris dataset."""
    print("🌸 Demo: Iris Dataset")

    # Load the iris dataset
    print("  • Loading iris dataset...")
    iris = sns.load_dataset("iris")

    # Create minimal horizontal plot including species column
    print("  • Creating horizontal iris plot...")
    snp.parallelplot(
        data=iris,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        # flip=["sepal_width"],
        hue="species",
        orient="h",
    )

    plt.title("Iris Dataset - Horizontal Orientation", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path = "demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"🌸 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
