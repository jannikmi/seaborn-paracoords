"""
Demo: Seaborn Plotting Contexts - Paper, Notebook, Talk, Poster
Run with: uv run python scripts/demo_seaborn_contexts.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate parallel coordinates with different seaborn plotting contexts."""
    print("🎨 Demo: Seaborn Plotting Contexts")

    df = sns.load_dataset("iris")

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    contexts = ["paper", "notebook", "talk", "poster"]

    for idx, context in enumerate(contexts):
        print(f"  • Creating plot with '{context}' context...")
        with sns.plotting_context(context):
            snp.parallelplot(
                data=df,
                vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                hue="species",
                orient="v",
                alpha=0.7,
                ax=axes_flat[idx],
            )
            axes_flat[idx].set_title(
                f"Context: '{context}'",
                fontsize=12,
                fontweight="bold",
            )

    plt.suptitle(
        "Seaborn Plotting Contexts Demo\n(Paper, Notebook, Talk, Poster)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_seaborn_contexts.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")
    print("\n📝 Contexts explained:")
    print("  • paper: Smallest fonts and elements (for publications)")
    print("  • notebook: Default size (for Jupyter notebooks)")
    print("  • talk: Larger fonts (for presentations)")
    print("  • poster: Largest fonts (for posters)")


if __name__ == "__main__":
    main()
