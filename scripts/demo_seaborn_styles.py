"""
Demo: Seaborn Styles Comparison for Parallel Coordinates

This script demonstrates how parallel coordinates plots look with different
seaborn styles applied: white, dark, whitegrid, darkgrid, and ticks.

Note: Each style is saved as a separate file because matplotlib's style context
doesn't apply retroactively to existing axes. View the individual files to see
the visual differences between styles.

Run with: uv run python scripts/demo_seaborn_styles.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_parallel as snp

# Load example dataset
iris = sns.load_dataset("iris")

# Define all available seaborn styles
styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

print("🎨 Generating parallel coordinates plots with different seaborn styles...")
print("   (Each style saved as a separate file for accurate comparison)\n")

# Create individual plots for each style
print("📊 Creating individual styled plots...")

for style in styles:
    print(f"  - Creating plot with '{style}' style...")

    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=(12, 6))

        snp.parallelplot(
            data=iris,
            hue="species",
            ax=ax,
            alpha=0.6,
            linewidth=1.5,
        )

        ax.set_title(
            f"Parallel Coordinates - Seaborn Style: '{style}'",
            fontsize=14,
            fontweight="bold",
        )

        # Save both PNG and SVG for inspection
        png_path = f"tmp/seaborn_style_{style}.png"
        svg_path = f"tmp/seaborn_style_{style}.svg"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close()
        print(f"    Saved PNG: {png_path}")
        print(f"    Saved SVG: {svg_path}")

print("\n✨ All seaborn style demos completed!")
print("\n📁 Generated files:")
for style in styles:
    print(f"  - tmp/seaborn_style_{style}.png")
    print(f"  - tmp/seaborn_style_{style}.svg (for detailed inspection)")
print("\n💡 Tip: Compare the different style files to see variations in:")
print("   • Background colors (white vs. dark gray)")
print("   • Grid lines (whitegrid, darkgrid vs. none)")
print("   • Spine/border styles (ticks vs. others)")
