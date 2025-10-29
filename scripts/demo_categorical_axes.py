"""
Demo: Categorical Axes Support in Parallel Coordinates Plot

This script demonstrates the automatic detection and visualization of
categorical variables in parallel coordinates plots.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp

# Create output directory
os.makedirs("./tmp", exist_ok=True)


def demo_auto_categorical_detection():
    """Demo 1: Automatic detection of categorical variables."""
    print("\n" + "=" * 70)
    print("Demo 1: Automatic Categorical Detection with Mixed Data Types")
    print("=" * 70)

    # Create mixed-type dataset with both numeric and categorical columns
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 28, 32, 29, 31, 27, 33, 26],
            "Category": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
            "Score": [85, 92, 78, 88, 95, 82, 90, 87, 91, 84],
            "Grade": [
                "Good",
                "Excellent",
                "Fair",
                "Good",
                "Excellent",
                "Good",
                "Excellent",
                "Good",
                "Excellent",
                "Good",
            ],
            "Value": [100, 200, 150, 180, 220, 160, 210, 170, 205, 155],
        }
    )

    print("\nDataset:")
    print(df.head())
    print("\nCategorical columns automatically detected: 'Category', 'Grade'")
    print("Numeric columns: 'Age', 'Score', 'Value'")

    ax = snp.parallelplot(data=df, hue="Category", orient="v", alpha=0.7, linewidth=1.5)
    ax.set_title(
        "Mixed Data Types: Automatic Detection\n(Categorical + Numeric Columns)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = "./tmp/seaborn-paracoords_demo_categorical_auto_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def demo_explicit_categorical():
    """Demo 2: Explicitly specify categorical axes."""
    print("\n" + "=" * 70)
    print("Demo 2: Explicit Categorical Axes Specification")
    print("=" * 70)

    df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "A", "B", "C"],
            "value1": [10, 20, 15, 12, 18, 16],
            "value2": [100, 200, 150, 120, 180, 160],
            "group": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )

    print("\nDataset:")
    print(df)
    print("\nExplicitly setting 'category' and 'group' as categorical")

    ax = snp.parallelplot(
        data=df,
        vars=["category", "value1", "value2", "group"],
        categorical_axes=["category", "group"],
        hue="group",
        orient="v",
        alpha=0.7,
        linewidth=2.5,
    )
    ax.set_title(
        "Explicit Categorical Axes\n(Multiple Categorical Variables)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = "./tmp/seaborn-paracoords_demo_categorical_explicit.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def demo_custom_category_order():
    """Demo 3: Custom ordering of categories."""
    print("\n" + "=" * 70)
    print("Demo 3: Custom Category Ordering")
    print("=" * 70)

    df = pd.DataFrame(
        {
            "size": ["small", "large", "medium", "small", "large", "medium"],
            "score": [85, 95, 90, 88, 92, 87],
            "rating": ["good", "excellent", "average", "good", "excellent", "average"],
        }
    )

    print("\nDataset:")
    print(df)
    print("\nCustom ordering: size=['small', 'medium', 'large']")
    print("                 rating=['average', 'good', 'excellent']")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Default ordering (alphabetical)
    snp.parallelplot(
        data=df,
        categorical_axes=["size", "rating"],
        orient="v",
        alpha=0.7,
        linewidth=2.0,
        ax=ax1,
    )
    ax1.set_title("Default Ordering\n(Alphabetical)", fontsize=12, fontweight="bold")

    # Custom ordering
    snp.parallelplot(
        data=df,
        categorical_axes=["size", "rating"],
        category_orders={
            "size": ["small", "medium", "large"],
            "rating": ["average", "good", "excellent"],
        },
        orient="v",
        alpha=0.7,
        linewidth=2.0,
        ax=ax2,
    )
    ax2.set_title("Custom Ordering\n(Logical Order)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    output_path = "./tmp/seaborn-paracoords_demo_categorical_custom_order.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def demo_horizontal_categorical():
    """Demo 4: Horizontal orientation with categorical axes."""
    print("\n" + "=" * 70)
    print("Demo 4: Horizontal Orientation with Categorical Axes")
    print("=" * 70)

    # Load iris dataset
    iris = sns.load_dataset("iris")
    iris_sample = iris.sample(n=30, random_state=42)

    print("\nUsing Iris dataset sample (30 rows)")
    print("Categorical axis: 'species'")

    ax = snp.parallelplot(
        data=iris_sample,
        vars=["species", "sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orient="h",
        alpha=0.6,
        linewidth=1.5,
        palette="Set2",
    )
    ax.set_title(
        "Iris Dataset - Horizontal Orientation\n(Categorical + Numeric Axes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = "./tmp/seaborn-paracoords_demo_categorical_horizontal.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def demo_multiple_categories_per_axis():
    """Demo 5: Axes with many categories."""
    print("\n" + "=" * 70)
    print("Demo 5: Multiple Categories per Axis")
    print("=" * 70)

    df = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "East",
                "West",
                "North",
                "South",
                "East",
                "West",
            ],
            "product": ["A", "B", "C", "D", "B", "C", "D", "A"],
            "sales": [100, 150, 200, 250, 120, 180, 220, 110],
            "profit": [20, 30, 40, 50, 25, 35, 45, 22],
        }
    )

    print("\nDataset:")
    print(df)

    ax = snp.parallelplot(
        data=df,
        vars=["region", "product", "sales", "profit"],
        categorical_axes=["region", "product"],
        hue="region",
        orient="v",
        alpha=0.7,
        linewidth=2.0,
        palette="tab10",
    )
    ax.set_title(
        "Sales Data: Multiple Categorical Axes\n(Region & Product)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = "./tmp/seaborn-paracoords_demo_categorical_multiple.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CATEGORICAL AXES DEMO - Parallel Coordinates Plot")
    print("=" * 70)

    demo_auto_categorical_detection()
    demo_explicit_categorical()
    demo_custom_category_order()
    demo_horizontal_categorical()
    demo_multiple_categories_per_axis()

    print("\n" + "=" * 70)
    print("✅ All demos completed successfully!")
    print("=" * 70)
    print("\nOutput files saved to: ./tmp/")
    print("  - seaborn-paracoords_demo_categorical_auto_detection.png")
    print("  - seaborn-paracoords_demo_categorical_explicit.png")
    print("  - seaborn-paracoords_demo_categorical_custom_order.png")
    print("  - seaborn-paracoords_demo_categorical_horizontal.png")
    print("  - seaborn-paracoords_demo_categorical_multiple.png")
    print()
