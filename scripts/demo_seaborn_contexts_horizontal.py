"""
Demo: Seaborn Plotting Contexts - Horizontal Orientation
Run with: uv run python scripts/demo_seaborn_contexts_horizontal.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demonstrate horizontal parallel coordinates with different seaborn contexts."""
    print("🎨 Demo: Seaborn Plotting Contexts (Horizontal)")

    df = sns.load_dataset("iris")
    fig = plt.figure(figsize=(16, 12))
    contexts = ["paper", "notebook", "talk", "poster"]

    for idx, context in enumerate(contexts):
        print(f"  • Creating plot with '{context}' context...")

        with sns.plotting_context(context):
            ax = fig.add_subplot(2, 2, idx + 1)

            snp.parallelplot(
                data=df,
                vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                hue="species",
                orient="h",
                alpha=0.7,
                ax=ax,
            )

            ax.set_title(
                f"Context: '{context}'",
                fontsize=plt.rcParams["axes.titlesize"],
                fontweight="bold",
            )

    plt.suptitle(
        "Seaborn Plotting Contexts Demo - Horizontal Orientation\n(Paper, Notebook, Talk, Poster)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_seaborn_contexts_horizontal.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
