"""
Demo script for normalization modes.
Run with: uv run python scripts/demo_normalization.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn_parallel as snp


def main():
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

    # Save to tmp folder
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_normalization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
