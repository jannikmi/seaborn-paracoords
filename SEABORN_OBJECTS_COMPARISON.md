# Seaborn Objects vs seaborn-parallel: Feature Comparison

## Summary

**Can pure Seaborn Objects replace this library?**

**No.** While Seaborn Objects can produce basic parallel coordinates visualizations, it **cannot replicate the core functionality** that makes this library useful for data exploration.

## The Example

The Seaborn Objects approach referenced:

```python
(
    sns.load_dataset("iris")
    .rename_axis("example")
    .reset_index()
    .melt(["example", "species"])
    .pipe(so.Plot, x="variable", y="value", color="species")
    .add(so.Lines(alpha=.5), group="example")
)
```

vs. our library:

```python
parallelplot(
    data=sns.load_dataset("iris"),
    hue='species',
    alpha=0.5
)
```

## Critical Missing Features in Seaborn Objects

### 1. **Independent Axis Scaling** ❌ (DEALBREAKER)

**Seaborn Objects:**
- Forces ALL variables to share a single y-axis scale
- In the iris dataset, petal_width (range 0-2.5) and sepal_length (range 4-8) both use 0-8 scale
- Small variations in petal_width become invisible
- **This defeats the purpose of parallel coordinates for comparing variables with different ranges**

**seaborn-parallel:**
- Each variable gets its own y-axis showing actual data range (default behavior)
- petal_width axis shows 0.0-2.5
- sepal_length axis shows 4.5-8.0
- All variations visible regardless of scale
- Can ALSO force shared scaling with `sharey=True` if needed

### 2. **Per-Variable Axis Labels** ❌

**Seaborn Objects:**
- Single y-axis label applies to all variables
- Cannot show what values 0, 2, 4, 6, 8 mean for each specific variable

**seaborn-parallel:**
- Each variable displays its own tick labels showing actual min/max values
- User can immediately see the range and distribution for each variable

### 3. **Categorical Variable Handling** ⚠️

**Seaborn Objects:**
- Treats categorical values as text labels on the axis
- No proper spacing or numerical mapping
- Difficult to trace lines through categorical axes

**seaborn-parallel:**
- Maps categorical values to [0, 1] with proper spacing
- Displays category labels at correct positions
- Built-in `category_orders` parameter for custom ordering
- Smooth line interpolation through categorical axes

### 4. **API Simplicity** ⚠️

**Seaborn Objects:**
- Requires data reshaping with `.melt()`
- Need to understand "tidy data" format
- More verbose syntax

**seaborn-parallel:**
- Direct DataFrame input
- Intuitive parameters matching Seaborn conventions
- No preprocessing required

## Feature Comparison Table

| Feature | Seaborn Objects | seaborn-parallel |
|---------|----------------|------------------|
| Basic parallel coordinates | ✓ Yes | ✓ Yes |
| Color by category (hue) | ✓ Yes | ✓ Yes |
| Horizontal orientation | ⚠️ Manual reshape | ✓ `orient='h'` |
| **Independent axis scaling** | ❌ **No - forced shared y-axis** | ✓ **Default behavior** |
| Shared axis scaling | ✓ Default behavior | ✓ `sharex`/`sharey` params |
| **Per-variable axis labels** | ❌ **Single y-axis only** | ✓ **Each axis shows range** |
| Categorical variables | ⚠️ Shows as text | ✓ Proper [0,1] mapping |
| Custom category order | ⚠️ Manual pre-sort | ✓ `category_orders` param |
| Line width | ✓ `linewidth` param | ✓ `linewidth` param |
| Transparency | ✓ `alpha` param | ✓ `alpha` param |
| API simplicity | ⚠️ Requires `.melt()` | ✓ Direct DataFrame |

## What Seaborn Objects CAN Do

Seaborn Objects is excellent for:
- Basic exploratory parallel coordinates when all variables have similar scales
- Quick prototyping with the new `seaborn.objects` interface
- Integration with other Seaborn Objects plots in a pipeline

## What seaborn-parallel Adds

This library extends Seaborn with **parallel coordinates-specific functionality**:

1. **Independent axis scaling** - preserve original data ranges (core value)
2. **Per-variable axis labeling** - show actual min/max for each variable
3. **Categorical axis handling** - proper spacing and ordering
4. **Orientation support** - true vertical/horizontal layouts
5. **Scaling flexibility** - choose independent OR shared axes
6. **Simple API** - no data reshaping required

## Conclusion

**seaborn-parallel is NOT redundant** with Seaborn Objects. The libraries serve different purposes:

- **Seaborn Objects**: General-purpose plotting with a unified grammar
- **seaborn-parallel**: Specialized parallel coordinates with essential features for multi-variable data exploration

The independent axis scaling feature alone (showing each variable's actual range) makes this library irreplaceable for real-world parallel coordinates analysis.

## Visual Proof

Run this comparison script to see the difference:

```bash
uv run python tmp/compare_seaborn_objects_vs_library.py
```

The output clearly shows how Seaborn Objects forces all variables to 0-8 scale, while seaborn-parallel preserves each variable's natural range.

---

## Can We Work Around the Limitations?

### Attempted Workarounds

After normalizing data to [0,1], we can modify the Seaborn Objects plot aesthetically:

```python
# Normalize data
for col in numeric_cols:
    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Plot with Seaborn Objects
plot = so.Plot(melted, x="variable", y="value", color="species")...

# Remove shared y-axis
ax.set_yticks([])
ax.set_ylabel("")

# Add custom tick labels for each variable
for var, x_pos in variable_positions.items():
    for norm_y in [0, 0.25, 0.5, 0.75, 1.0]:
        orig_value = min_val + (max_val - min_val) * norm_y
        ax.text(x_pos - 0.05, norm_y, f'{orig_value:.1f}', ...)
```

**Result:** ⚠️ Looks better, but fundamentally flawed

### Why Workarounds Fail

The aggressive modifications improve **aesthetics** but cannot fix the **fundamental problem**:

**The data is still normalized to [0,1] internally**, which means:

1. **All variables appear equally tall** (misleading visual representation)
   - `tiny_range` (0.4 range) → 100% height
   - `huge_range` (400 range) → 100% height
   - A 0.1 change looks identical to a 100 change!

2. **Cannot show true proportional scaling**
   - Variables with 1000x different ranges appear visually identical
   - Hides the relative importance of each variable
   - Makes it impossible to see which variables vary more

3. **Tick labels are decorative, not functional**
   - Not real matplotlib axis ticks
   - Don't respond to zoom/pan interactions
   - Cannot be styled with rcParams
   - Manual position calculations required

### Real-World Impact

Imagine patient medical data:
- Age: 20-80 (range: 60 years)
- Heart rate: 60-100 (range: 40 bpm)
- Temperature: 36.5-38.5°C (range: 2°C)

**With normalization:** A 1°C fever appears as visually dramatic as aging 30 years!

**With independent axes:** You see the true proportional differences - the fever is a small change, age is a large change.

This is not an aesthetic preference - **it's a correctness issue**. Parallel coordinates must show true independent scaling to be scientifically valid.

### Code Complexity

- **Seaborn Objects + workaround:** 50+ lines of manual drawing code
- **seaborn-parallel:** 1 line

See demonstration scripts in `tmp/`:
- `aggressive_workaround_demo.py` - Modified Seaborn Objects plots
- `demonstration_why_normalization_matters.py` - Visual proof of the problem

---

**Last updated:** October 28, 2025
