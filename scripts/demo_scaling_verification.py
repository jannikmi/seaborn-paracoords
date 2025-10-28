"""
Visual verification that scaling works correctly.
This creates a simple dataset where the difference should be obvious.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn_parallel as snp


def main():
    """Demo with simple data to verify scaling."""
    print("🔍 Scaling Verification Demo")

    # Create simple data with very different ranges
    # Variable A: 0-10
    # Variable B: 0-100
    # Variable C: 0-1000
    df = pd.DataFrame(
        {
            "A_0to10": [0, 5, 10, 2, 8],
            "B_0to100": [0, 50, 100, 20, 80],
            "C_0to1000": [0, 500, 1000, 200, 800],
            "category": ["low", "mid", "high", "low", "high"],
        }
    )

    print("\n📊 Data ranges:")
    print(f"  A: {df['A_0to10'].min():.0f} - {df['A_0to10'].max():.0f}")
    print(f"  B: {df['B_0to100'].min():.0f} - {df['B_0to100'].max():.0f}")
    print(f"  C: {df['C_0to1000'].min():.0f} - {df['C_0to1000'].max():.0f}")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Default (each variable scaled independently)
    print("\n1. Default (independent scaling)...")
    snp.parallelplot(
        data=df,
        vars=["A_0to10", "B_0to100", "C_0to1000"],
        hue="category",
        orient="v",
        alpha=0.8,
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set_title(
        "DEFAULT (sharey=False)\nEach variable scaled to [0,1] independently\n"
        + "Lines should show different patterns",
        fontsize=11,
        fontweight="bold",
    )

    # Right: Shared y-axis
    print("2. Shared y-axis...")
    snp.parallelplot(
        data=df,
        vars=["A_0to10", "B_0to100", "C_0to1000"],
        hue="category",
        orient="v",
        sharey=True,
        alpha=0.8,
        linewidth=2,
        ax=axes[1],
    )
    axes[1].set_title(
        "SHARED Y-AXIS (sharey=True)\nAll variables scaled to [0,1000]\n"
        + "Lines compressed on left axes",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/seaborn-paracoords_demo_scaling_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Verification plot saved to: {output_path}")
    print("\n📝 What to look for:")
    print("  LEFT (default): Each line should use full vertical range on each axis")
    print("  RIGHT (sharey): Lines should be compressed on A and B axes (small values)")
    print("                  Lines should use full range only on C axis (large values)")
    print("\n🔍 If both plots look the same, the scaling is broken!")


if __name__ == "__main__":
    main()
