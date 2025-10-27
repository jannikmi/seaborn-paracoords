"""
Demo script for automatic variable selection.
Run with: uv run python scripts/demo_auto_selection.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn_parallel as snp


def main():
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

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/auto_selection_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
