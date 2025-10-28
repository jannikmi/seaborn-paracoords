"""
Test Seaborn Integration - All Contexts and Styles

This script tests the parallel coordinates plot with all Seaborn contexts
and styles to ensure proper integration.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import seaborn_parallel as snp


def main():
    # Load sample data
    df = sns.load_dataset("iris")

    print("Testing Seaborn Integration")
    print("=" * 60)

    # Test all contexts
    contexts = ["paper", "notebook", "talk", "poster"]
    for context in contexts:
        print(f"\nTesting context: {context}")
        with sns.plotting_context(context):
            fig, ax = plt.subplots(figsize=(10, 6))
            snp.parallelplot(
                df,
                vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                hue="species",
                ax=ax,
            )
            ax.set_title(f"Context: {context}")
            output_path = f"./tmp/context_{context}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved: {output_path}")

    # Test all styles
    styles = ["white", "whitegrid", "dark", "darkgrid", "ticks"]
    for style in styles:
        print(f"\nTesting style: {style}")
        with sns.axes_style(style):
            fig, ax = plt.subplots(figsize=(10, 6))
            snp.parallelplot(
                df,
                vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                hue="species",
                ax=ax,
            )
            ax.set_title(f"Style: {style}")
            output_path = f"./tmp/style_{style}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved: {output_path}")

    # Test combined: poster context with whitegrid style
    print("\nTesting combined: poster + whitegrid")
    with sns.plotting_context("poster"):
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            snp.parallelplot(
                df,
                vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                hue="species",
                ax=ax,
            )
            ax.set_title("Combined: Poster Context + Whitegrid Style")
            output_path = "./tmp/combined_poster_whitegrid.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved: {output_path}")

    # Test with set_theme (most common usage)
    print("\nTesting set_theme approach")
    sns.set_theme(context="talk", style="darkgrid", palette="muted")
    fig, ax = plt.subplots(figsize=(11, 7))
    snp.parallelplot(
        df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        ax=ax,
    )
    ax.set_title("Using set_theme(context='talk', style='darkgrid', palette='muted')")
    output_path = "./tmp/set_theme_example.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_path}")

    # Reset to default
    sns.set_theme()

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("Check the ./tmp/ directory for generated plots")


if __name__ == "__main__":
    main()
