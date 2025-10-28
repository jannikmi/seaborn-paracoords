"""
Aggressive workaround: Modify Seaborn Objects output to approximate
parallel coordinates with per-variable axes.

This pushes the workaround as far as possible by:
1. Removing the shared [0,1] y-axis ticks
2. Adding custom tick labels for each variable's x-position
3. Using vertical text to simulate separate y-axes
"""

import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import numpy as np


# RIGHT: seaborn-parallel for comparison
from seaborn_parallel import parallelplot


# Load data
df = sns.load_dataset("iris")

# Setup
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Normalize each column to [0, 1]
normalized_df = df.copy()
original_ranges = {}

print("=" * 80)
print("AGGRESSIVE WORKAROUND: Modify Seaborn Objects output")
print("=" * 80)

for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    original_ranges[col] = (min_val, max_val)
    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
    print(f"{col}: [{min_val:.2f}, {max_val:.2f}] → [0, 1]")

# Melt for Seaborn Objects
melted = (
    normalized_df.rename_axis("example")
    .reset_index()
    .melt(["example", "species"], var_name="variable", value_name="normalized_value")
)

# ============================================================================
# Create modified Seaborn Objects plot
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT: Aggressive workaround
print("\nCreating aggressively modified plot...")
plot = (
    so.Plot(melted, x="variable", y="normalized_value", color="species")
    .add(so.Lines(alpha=0.3), group="example")
    .on(ax1)
    .plot()
)

# Step 1: Remove the shared y-axis ticks (they're meaningless)
ax1.set_yticks([])
ax1.set_ylabel("")

# Step 2: Add custom y-tick labels at each x-position
variable_positions = {var: i for i, var in enumerate(numeric_cols)}

# Create custom tick marks and labels for each variable
tick_positions = np.linspace(0, 1, 5)  # 5 ticks from 0 to 1

for var, x_pos in variable_positions.items():
    min_val, max_val = original_ranges[var]

    # Draw custom tick marks and labels on the LEFT side of each variable
    for norm_y in tick_positions:
        # Calculate original value
        orig_value = min_val + (max_val - min_val) * norm_y

        # Draw tick mark
        ax1.plot(
            [x_pos - 0.02, x_pos],
            [norm_y, norm_y],
            color="black",
            linewidth=1,
            clip_on=False,
            zorder=100,
        )

        # Add tick label
        ax1.text(
            x_pos - 0.05,
            norm_y,
            f"{orig_value:.1f}",
            ha="right",
            va="center",
            fontsize=8,
            clip_on=False,
        )

# Step 3: Add vertical axis lines for each variable
for var, x_pos in variable_positions.items():
    ax1.axvline(x=x_pos, color="gray", linewidth=1.5, alpha=0.5, zorder=1)

# Step 4: Adjust layout to prevent clipping
ax1.set_xlim(-0.5, len(numeric_cols) - 0.5)
ax1.set_ylim(-0.05, 1.05)

ax1.set_title(
    "Seaborn Objects + Aggressive Modifications\n(Custom tick labels per variable)",
    fontsize=12,
    fontweight="bold",
)

# Add explanatory note
ax1.text(
    0.5,
    -0.15,
    "✓ Removed shared [0,1] axis\n"
    + "✓ Added per-variable tick labels\n"
    + "⚠️  But still: all variables use same normalized [0,1] scale internally",
    ha="center",
    transform=ax1.transAxes,
    fontsize=9,
    color="orange",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)


parallelplot(data=df, hue="species", alpha=0.3, ax=ax2)

ax2.set_title(
    "seaborn-parallel (Native Support)\n(True independent axes)",
    fontsize=12,
    fontweight="bold",
)

# Add note
ax2.text(
    0.5,
    -0.15,
    "✓ True per-variable axes (not normalized internally)\n"
    + "✓ Native matplotlib tick positioning\n"
    + "✓ Each variable uses its actual data range",
    ha="center",
    transform=ax2.transAxes,
    fontsize=9,
    color="green",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)

plt.tight_layout()
plt.savefig("tmp/aggressive_workaround_comparison.png", dpi=150, bbox_inches="tight")
print("\n📊 Saved: tmp/aggressive_workaround_comparison.png")

# ============================================================================
# Create another version with even more modifications
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

print("\nCreating ultra-aggressive version...")
plot = (
    so.Plot(melted, x="variable", y="normalized_value", color="species")
    .add(so.Lines(alpha=0.3), group="example")
    .on(ax)
    .plot()
)

# Remove all default axis formatting
ax.set_yticks([])
ax.set_ylabel("")
ax.spines["left"].set_visible(False)

# Add custom axes for each variable with grid lines
for var, x_pos in variable_positions.items():
    min_val, max_val = original_ranges[var]

    # Draw vertical axis line (thicker)
    ax.plot(
        [x_pos, x_pos], [0, 1], color="black", linewidth=2, clip_on=False, zorder=100
    )

    # Add grid lines and tick labels
    tick_positions = np.linspace(0, 1, 6)  # 6 ticks
    for norm_y in tick_positions:
        orig_value = min_val + (max_val - min_val) * norm_y

        # Horizontal grid line (light)
        ax.axhline(y=norm_y, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

        # Tick mark (extending both sides)
        ax.plot(
            [x_pos - 0.03, x_pos + 0.03],
            [norm_y, norm_y],
            color="black",
            linewidth=1.5,
            clip_on=False,
            zorder=100,
        )

        # Tick label (left side)
        ax.text(
            x_pos - 0.06,
            norm_y,
            f"{orig_value:.1f}",
            ha="right",
            va="center",
            fontsize=9,
            clip_on=False,
            fontweight="bold",
        )

# Adjust layout
ax.set_xlim(-0.5, len(numeric_cols) - 0.5)
ax.set_ylim(-0.08, 1.08)

ax.set_title(
    "Ultra-Aggressive Workaround\n(Maximum aesthetic modifications)",
    fontsize=14,
    fontweight="bold",
)

# Add legend in better position
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

plt.tight_layout()
plt.savefig("tmp/ultra_aggressive_workaround.png", dpi=150, bbox_inches="tight")
print("📊 Saved: tmp/ultra_aggressive_workaround.png")

# ============================================================================
# Analysis: Does this solve the problem?
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS: Does the aggressive workaround solve the problem?")
print("=" * 80)

print("\n✓ Improvements achieved:")
print("  1. Removed misleading shared [0,1] y-axis")
print("  2. Added per-variable tick labels showing original ranges")
print("  3. Visual appearance is closer to true parallel coordinates")
print("  4. Added vertical axis lines for each variable")
print("  5. Added custom grid lines")

print("\n❌ Fundamental limitations that CANNOT be fixed:")
print("  1. Data is still normalized to [0,1] internally")
print("     - All variables forced to same visual scale")
print("     - A value of 0.5 looks the same height for all variables")
print("     - But 0.5 means different things: ")
print(f"       - sepal_length: 0.5 = {4.3 + (7.9 - 4.3) * 0.5:.2f} cm")
print(f"       - petal_width:  0.5 = {0.1 + (2.5 - 0.1) * 0.5:.2f} cm")
print("  2. Cannot show TRUE independent scaling")
print("     - sepal_length (range 3.6) and petal_width (range 2.4)")
print("       look equally tall in the plot")
print("     - With true independent axes, they would have different heights")
print("  3. Tick labels are 'fake' - decorative text, not true axis ticks")
print("     - Matplotlib doesn't know about them")
print("     - Interactive features (zoom, pan) won't update them")
print("     - Cannot be styled with rcParams")

print("\n⚠️  Code complexity:")
print("  - Simple version: ~30 lines of manual annotation code")
print("  - Ultra version: ~50+ lines of custom drawing code")
print("  - vs. parallelplot(): 1 line")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print("""
The aggressive workaround improves AESTHETICS but cannot fix the FUNDAMENTAL
limitation: Seaborn Objects forces internal normalization to a shared scale.

You can make it LOOK like parallel coordinates with custom annotations,
but it's still using normalized [0,1] data underneath, which means:
- All variables appear equally tall (misleading visual)
- Cannot show true proportional differences between variable ranges
- Tick labels are decorative, not functional matplotlib axis ticks

It's like putting a sports car body kit on a sedan - looks better,
but it's still not a sports car underneath.

For production use, native independent axis support is essential.
""")

print("=" * 80)
