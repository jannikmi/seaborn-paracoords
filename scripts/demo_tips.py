"""
Demo script for tips dataset.
Run with: uv run python scripts/demo_tips.py
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Demo with tips dataset."""
    print("🍽️  Demo: Tips dataset")

    df = sns.load_dataset("tips")

    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(
        data=df, vars=["total_bill", "tip", "size"], hue="time", alpha=0.6, ax=ax
    )
    ax.set_title("Restaurant Tips - Parallel Coordinates")
    plt.tight_layout()

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_tips.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
