"""
Combined demo script showing both orientations of iris dataset.
Run with: uv run python scripts/parallel_demo.py (or demo_iris_combined.py)
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn_parallel as snp


def demo_iris_vertical():
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
    plt.show()


def demo_iris_horizontal():
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
    plt.show()


def demo_tips():
    """Demo with tips dataset."""
    print("🍽️  Demo: Tips dataset")

    df = sns.load_dataset("tips")

    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(
        data=df, vars=["total_bill", "tip", "size"], hue="time", alpha=0.6, ax=ax
    )
    ax.set_title("Restaurant Tips - Parallel Coordinates")
    plt.tight_layout()
    plt.show()


def demo_custom_styling():
    """Demo with custom Seaborn styling."""
    print("🎨 Demo: Custom styling integration")

    # Apply custom seaborn theme
    sns.set_theme(style="darkgrid", palette="husl")

    df = sns.load_dataset("flights")
    df_sample = df.sample(n=100, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(
        data=df_sample,
        vars=["year", "month", "passengers"],
        alpha=0.8,
        linewidth=1.5,
        ax=ax,
    )
    ax.set_title("Flight Passengers - Custom Seaborn Theme")
    plt.tight_layout()
    plt.show()

    # Reset to default
    sns.reset_defaults()


def demo_auto_selection():
    """Demo with automatic variable selection."""
    print("🤖 Demo: Automatic variable selection")

    # Create a mixed dataset
    df = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [10, 20, 30, 40, 50],
            "numeric3": [100, 200, 300, 400, 500],
            "text_col": ["A", "B", "C", "D", "E"],
            "bool_col": [True, False, True, False, True],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    # Should automatically select only numeric columns
    snp.parallelplot(data=df, alpha=0.8, ax=ax)
    ax.set_title("Auto Variable Selection - Only Numeric Columns")
    plt.tight_layout()
    plt.show()


def demo_normalization_comparison():
    """Demo comparing independent vs shared scaling."""
    print("📊 Demo: Scaling comparison")

    # Create data with different scales
    df = pd.DataFrame(
        {
            "small": [0.1, 0.2, 0.3, 0.4, 0.5],
            "medium": [10, 20, 30, 40, 50],
            "large": [1000, 2000, 3000, 4000, 5000],
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Independent scaling (default)
    snp.parallelplot(data=df, alpha=0.8, ax=ax1)
    ax1.set_title("Independent Scaling (default)\nEach variable uses full range")

    # Shared y-axis
    snp.parallelplot(data=df, sharey=True, alpha=0.8, ax=ax2)
    ax2.set_title("Shared Y-Axis (sharey=True)\nAll variables share 0-5000 range")

    plt.tight_layout()
    plt.show()


def create_combined_iris_demo():
    """Create a single PNG with both vertical and horizontal iris plots."""
    print("🌸 Creating combined iris dataset demo - both orientations")

    df = sns.load_dataset("iris")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Vertical plot
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="v",
        alpha=0.7,
        ax=ax1,
    )
    ax1.set_title("Iris Dataset - Vertical Parallel Coordinates")

    # Horizontal plot
    snp.parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="h",
        alpha=0.7,
        ax=ax2,
    )
    ax2.set_title("Iris Dataset - Horizontal Parallel Coordinates")

    plt.tight_layout()

    # Create tmp directory if it doesn't exist
    os.makedirs("./tmp", exist_ok=True)

    # Save to relative tmp folder
    output_path = "./tmp/seaborn-paracoords_demo_parallel.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Combined demo plot saved to: {output_path}")
    return output_path


def main():
    """Run the combined iris demonstration and save to PNG."""
    print("🚀 Parallel Coordinates Plot Demo - Creating Single Output File\n")

    try:
        output_path = create_combined_iris_demo()
        print("\n✅ Demo completed successfully!")
        print(f"🎨 Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
