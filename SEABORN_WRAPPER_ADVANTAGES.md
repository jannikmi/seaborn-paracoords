### # Seaborn Objects Integration: Advantages for a Wrapper Library

## Executive Summary

**Key Insight:** Normalized [0,1] parallel coordinates are scientifically valid for pattern detection and ML visualization. As a **Seaborn wrapper project**, this library should leverage Seaborn Objects for the normalized mode while keeping the current implementation for absolute value comparison.

## Two Valid Use Cases

### 1. **Absolute Value Comparison** (Current Implementation)
**Question:** "How do these measurements compare in their actual units?"

```python
parallelplot(data=iris, hue='species')  # Independent axes, original ranges
```

**Use cases:**
- Comparing measurements in their original units
- Understanding actual magnitudes
- Scientific reporting with raw values

### 2. **Pattern & Relationship Detection** (Seaborn Objects)
**Question:** "What patterns/clusters exist across variables?"

```python
parallelplot(data=iris, hue='species', normalize=True)  # Use Seaborn Objects
```

**Use cases:**
- Identifying clusters and patterns
- ML feature visualization before training
- Variables with very different scales (e.g., age vs. temperature)
- Standard data science EDA

**Why normalization is valid here:**
- Matches scikit-learn preprocessing (StandardScaler, MinMaxScaler)
- Makes all variables contribute equally to pattern visibility
- Standard practice in ML visualization
- Required for fair comparison across different units

## Advantages of Seaborn Objects (for normalized mode)

### 1. **Code Reuse** ✓

**Current approach:** Custom implementation
```python
# In parallelplot.py - we manually handle everything
colors = _handle_colors(...)  # Custom color logic
ax.plot(x, y, color=color, ...)  # Manual plotting
ax.legend(...)  # Custom legend formatting
```

**Seaborn Objects approach:** Leverage existing functionality
```python
# Seaborn handles it all
(so.Plot(melted_data, x="variable", y="value", color="species")
   .add(so.Lines(alpha=.3), group="index")
   .plot())
```

**Benefits:**
- ✓ Automatic palette integration (no custom `_handle_colors()`)
- ✓ Legend formatting built-in
- ✓ Missing data handling
- ✓ Theme/context integration (paper/talk/poster)
- ✓ Color cycle management

### 2. **Advanced Features** ✓

Features that would be hard to implement manually:

**Faceting (small multiples):**
```python
(so.Plot(data, x="variable", y="value", color="species")
   .add(so.Lines(alpha=.3), group="index")
   .facet("species", wrap=3))  # ← Automatic grid of subplots!
```

**Statistical overlays:**
```python
(so.Plot(data, x="variable", y="value", color="species")
   .add(so.Lines(alpha=.1), group="index")   # Individual lines
   .add(so.Line(linewidth=3), so.Agg()))     # Mean line overlay
```

**Confidence bands:**
```python
.add(so.Band(), so.Est(errorbar="sd"))  # Add standard deviation bands
```

**Benefits:**
- ✓ Faceting with `.facet()` - would need ~100 lines of custom code
- ✓ Statistical transformations - would need scipy integration
- ✓ Multiple mark types - would need custom composition logic
- ✓ Extensibility - users can add their own layers

### 3. **Simpler Implementation** ✓

**Current approach:** ~687 lines managing matplotlib details

**Normalized mode with Seaborn Objects:**
```python
def parallelplot(..., normalize=True):
    if normalize:
        # Normalize data
        normalized_df = data.copy()
        for col in numeric_cols:
            normalized_df[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        # Melt to tidy format
        melted = normalized_df.reset_index().melt(id_vars=['index', hue], ...)

        # Use Seaborn Objects
        plot = (so.Plot(melted, x="variable", y="value", color=hue)
                  .add(so.Lines(alpha=alpha, linewidth=linewidth), group="index"))

        if ax:
            plot.on(ax).plot()
        else:
            plot.show()

        return plot  # Return Plot object for further customization!
    else:
        # Use current implementation
        ...
```

**Benefits:**
- ~50 lines instead of ~300 for normalized mode
- Matplotlib complexity hidden by Seaborn
- User gets Plot object to add more layers
- Easier to maintain

### 4. **Categorical Variables** ⚠️

**Test results:** Seaborn Objects handles categoricals with encoding

```python
# Encode categoricals as numeric codes
df['category'] = df['category'].cat.codes

# Plot works fine
(so.Plot(melted, x="variable", y="value", color="group")
   .add(so.Lines(alpha=.3), group="index")
   .plot())

# Add categorical labels manually (same as current library)
for code, label in enumerate(['Low', 'Medium', 'High']):
    ax.text(x_pos, code, label, ...)
```

**Conclusion:** Categorical handling is comparable to current implementation - both need manual label placement. No advantage or disadvantage.

### 5. **Integration with ML Workflows** ✓

```python
from sklearn.preprocessing import MinMaxScaler
from seaborn_parallel import parallelplot

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Visualize normalized features (matches what ML model sees)
parallelplot(pd.DataFrame(X_normalized, columns=feature_names),
             hue=y, normalize=True)  # Already normalized, but consistent API
```

**Benefits:**
- ✓ Visual representation matches ML preprocessing
- ✓ Standard workflow in data science
- ✓ Users expect this for feature engineering
- ✓ Matches pandas-profiling, yellowbrick, etc.

### 6. **Palette Ecosystem** ✓

```python
# Seamless integration with Seaborn palettes
sns.set_palette('husl')
parallelplot(data, hue='species', normalize=True)  # Uses active palette

# Or explicit
parallelplot(data, hue='species', normalize=True, palette='rocket')
```

Current implementation already does this via `_handle_colors()`, but Seaborn Objects makes it more automatic.

## Proposed API

```python
def parallelplot(
    data,
    hue=None,
    normalize=False,  # ← NEW parameter
    alpha=0.5,
    linewidth=1.0,
    palette=None,
    orient='v',
    ax=None,
    **kwargs
):
    """
    Create parallel coordinates plot.

    Parameters
    ----------
    normalize : bool, default False
        If True, normalize all variables to [0, 1] and use Seaborn Objects
        for rendering. This enables pattern detection and advanced features
        (faceting, statistical overlays). If False, use independent axes
        with original data ranges.

    Returns
    -------
    If normalize=True: seaborn.objects.Plot (can add more layers)
    If normalize=False: matplotlib.axes.Axes
    """

    if normalize:
        return _parallelplot_normalized(data, hue, alpha, linewidth, palette, ax, **kwargs)
    else:
        return _parallelplot_independent(data, hue, alpha, linewidth, palette, orientation, ax, **kwargs)
```

## Implementation Strategy

### Phase 1: Add normalized mode
1. Create `_parallelplot_normalized()` function using Seaborn Objects
2. Add `normalize` parameter (default `False` for backward compatibility)
3. Return `Plot` object for user extensibility

### Phase 2: Enhance normalized mode
1. Add optional tick label customization for categoricals
2. Support `**plot_kws` to pass to Seaborn Objects
3. Add convenience methods for common overlays (mean, median, etc.)

### Phase 3: Documentation
1. Add examples showing when to use `normalize=True` vs `False`
2. Show advanced Seaborn Objects features (faceting, stats)
3. ML workflow integration examples

## Code Comparison

**Current (independent axes):** ~687 lines
**New normalized mode:** ~50 lines + reuse Seaborn Objects

**Total LOC:** Similar or less (code reuse!)
**Features gained:** Faceting, stats, easier palette integration
**Features kept:** Independent axes, categorical support, original ranges

## Conclusion

As a **Seaborn wrapper project**, using Seaborn Objects for normalized mode is the right choice:

✓ **Dramatic code reduction** for normalized mode
✓ **Advanced features** (faceting, stats) come free
✓ **Scientific validity** - normalization IS correct for pattern detection
✓ **Best of both worlds** - keep independent axes for absolute values
✓ **True to mission** - actually wrapping Seaborn, not reimplementing it
✓ **Extensibility** - users can add Seaborn Objects layers

The current implementation remains valuable for absolute value comparison, while Seaborn Objects handles the normalized/pattern-detection use case elegantly.

**Recommendation:** Implement `normalize=True` mode using Seaborn Objects.

---

**Demo outputs:**
- `tmp/seaborn_objects_faceted.png` - Small multiples
- `tmp/seaborn_objects_with_stats.png` - Statistical overlays
- `tmp/seaborn_objects_palettes.png` - Palette integration
- `tmp/categorical_comparison.png` - Categorical handling
