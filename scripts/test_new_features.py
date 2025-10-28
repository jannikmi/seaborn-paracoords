"""
Test script for new sharex/sharey features.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp


def main():
    """Test the new sharex and sharey features."""
    print("🧪 Testing new sharex/sharey features")

    df = sns.load_dataset("iris")

    # Test 1: Default behavior (original axis values)
    print("\n1. Default behavior (original axis values per variable)")
    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(data=df, hue="species", orient="v", alpha=0.7, ax=ax)
    ax.set_title("Default: Original Axis Values")
    plt.tight_layout()
    os.makedirs("./tmp", exist_ok=True)
    plt.savefig("./tmp/test_default.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved to tmp/test_default.png")

    # Test 2: Shared y-axis (vertical orientation)
    print("\n2. Shared y-axis (vertical orientation)")
    fig, ax = plt.subplots(figsize=(10, 6))
    snp.parallelplot(data=df, hue="species", orient="v", sharey=True, alpha=0.7, ax=ax)
    ax.set_title("Shared Y-Axis Range (sharey=True)")
    plt.tight_layout()
    plt.savefig("./tmp/test_sharey.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved to tmp/test_sharey.png")

    # Test 3: Horizontal orientation with shared x-axis
    print("\n3. Horizontal orientation with shared x-axis")
    fig, ax = plt.subplots(figsize=(10, 8))
    snp.parallelplot(data=df, hue="species", orient="h", sharex=True, alpha=0.7, ax=ax)
    ax.set_title("Horizontal with Shared X-Axis (sharex=True)")
    plt.tight_layout()
    plt.savefig("./tmp/test_horizontal_sharex.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved to tmp/test_horizontal_sharex.png")

    # Test 4: Horizontal orientation without shared axes
    print("\n4. Horizontal orientation without shared axes")
    fig, ax = plt.subplots(figsize=(10, 8))
    snp.parallelplot(data=df, hue="species", orient="h", alpha=0.7, ax=ax)
    ax.set_title("Horizontal with Original Axis Values")
    plt.tight_layout()
    plt.savefig("./tmp/test_horizontal_default.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved to tmp/test_horizontal_default.png")

    print("\n✅ All tests completed successfully!")
    print("📁 Check the ./tmp folder for output images")


if __name__ == "__main__":
    main()
