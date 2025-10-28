"""
Demo script for custom styling options.
Run with: uv run python scripts/demo_custom_styling.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demo with custom Seaborn styling."""
    print("🎨 Demo: Custom styling integration")

    # Apply custom seaborn theme
    sns.set_theme(style="darkgrid", palette="husl")

    df = sns.load_dataset("flights")
    df_sample = df.sample(n=100, random_state=42)

    # Use only numeric columns to avoid categorical issues
    numeric_vars = ["year", "passengers"]  # month is categorical, skip it

    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(data=df_sample, vars=numeric_vars, alpha=0.8, linewidth=1.5, ax=ax)
    ax.set_title("Flight Passengers - Custom Seaborn Theme")
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_custom_styling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Reset to default
    sns.reset_defaults()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
