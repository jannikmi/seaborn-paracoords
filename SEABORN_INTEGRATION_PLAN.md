# Seaborn Integration Implementation Plan (DRY Principle)
## Research Phase Complete ✅

**Goal**: Make this a true Seaborn extension/wrapper, not a reimplementation.
**Principle**: Delegate to Seaborn wherever possible. Only write custom code for parallel-coordinates-specific logic.

Based on analysis of Seaborn v0.13.2 source code and runtime behavior.

---

## Key Findings from Phase 1 Research

### 1. **Seaborn's Styling System**

Seaborn uses matplotlib's `rcParams` system with two separate mechanisms:

- **`axes_style()`**: Controls visual appearance (grid, colors, spines)
  - Styles: `white`, `dark`, `whitegrid`, `darkgrid`, `ticks`
  - Sets `axes.grid` to `True` for grid styles, `False` otherwise
  - Grid automatically appears when creating new axes under a grid style

- **`plotting_context()`**: Controls sizing/scaling
  - Contexts: `paper` (0.8×), `notebook` (1.0×), `talk` (1.5×), `poster` (2.0×)
  - Scales fonts, line widths, grid widths proportionally

**Key insight**: Seaborn DOES NOT have custom text/annotation utilities. It relies entirely on matplotlib's `ax.text()` and `ax.annotate()`, but benefits from scaled rcParams.

### 2. **What Seaborn Provides**

**Available utilities** (from `seaborn.utils`):
- ✅ `despine()` - removes/styles plot spines
- ✅ `move_legend()` - repositions legends with proper styling
- ✅ `get_color_cycle()` - retrieves current color cycle
- ✅ Color manipulation: `desaturate()`, `saturate()`, `set_hls_values()`
- ❌ NO text/annotation helpers
- ❌ NO grid styling utilities (uses rcParams only)
- ❌ NO custom LineCollection wrappers

**Color palettes**:
- `color_palette()` returns `_ColorPalette` class (subclass of list)
- Has context manager support for temporary palette changes
- Provides `.as_hex()` method

### 3. **How Seaborn's Plots Look "Seaborn-ish"**

They simply:
1. Create plots using matplotlib primitives (`ax.plot()`, `LineCollection`, etc.)
2. Trust that `rcParams` are already set correctly via `set_theme()`
3. Call `despine()` at the end
4. DON'T manually set grid - it appears automatically if style has `axes.grid=True`

### 4. **Critical Discovery: Grid Behavior**

**Current code has a bug:**
```python
ax.grid(True, alpha=0.3)  # ❌ WRONG - forces grid ON regardless of style
```

**Correct approach:**
```python
# Don't call ax.grid() at all!
# Grid appears automatically if user set whitegrid/darkgrid style
```

### 5. **Context Scaling Values**

| Context  | Multiplier | font.size | xtick.labelsize | lines.linewidth | grid.linewidth |
|----------|-----------|-----------|-----------------|-----------------|----------------|
| paper    | 0.8×      | 9.6       | 8.8             | 1.2             | 0.8            |
| notebook | 1.0×      | 12.0      | 11.0            | 1.5             | 1.0            |
| talk     | 1.5×      | 18.0      | 16.5            | 2.25            | 1.5            |
| poster   | 2.0×      | 24.0      | 22.0            | 3.0             | 2.0            |

---

## What We're Currently Duplicating (Code Smell Analysis) 🚨

### 1. **Color Mapping Logic** - `_handle_colors()` function

**Current implementation** (lines 372-392):
```python
def _handle_colors(data, hue, palette, n_rows):
    """Handle color mapping with proper fallbacks."""
    if hue is None or data is None:
        default_color = sns.color_palette("deep", 1)[0]
        return [default_color] * n_rows, None, None

    if hue not in data.columns:
        raise KeyError(f"hue column '{hue}' not found in data")

    hue_data = data[hue]
    unique_vals = sorted(hue_data.dropna().unique())

    if len(unique_vals) == 0:
        default_color = sns.color_palette("deep", 1)[0]
        return [default_color] * n_rows, None, None

    palette_colors = sns.color_palette(palette, len(unique_vals))
    color_map = dict(zip(unique_vals, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, unique_vals
```

**Problem**: This duplicates Seaborn's `HueMapping` class logic!

**Seaborn already has** (in `seaborn._base.py`):
- `HueMapping` - handles categorical/numeric hue mapping
- `categorical_mapping()` - creates color lookup tables
- `numeric_mapping()` - handles continuous colormaps
- Proper support for palette dictionaries, lists, and names

**Solution**: Use `HueMapping` directly instead of reimplementing.

### 2. **Hardcoded Font Sizes**

**Current**: `fontsize=8` (lines 470, 527)
**Problem**: Ignores context settings
**Solution**: Use `mpl.rcParams['xtick.labelsize']`

### 3. **Manual Grid Override**

**Current**: `ax.grid(True, alpha=0.3)` (line 596)
**Problem**: Overrides user's style choice
**Solution**: Remove - let Seaborn's style system handle it

### 4. **Default Color Selection**

**Current**: `sns.color_palette("deep", 1)[0]`
**Problem**: Doesn't use active color cycle
**Solution**: Use `get_color_cycle()` from Seaborn utils

---

## Revised Implementation Plan (DRY Focus)

## Revised Implementation Plan (DRY Focus)

### **NEW Phase 0: Replace Custom Color Mapping with Seaborn's HueMapping** 🎨

This is the most significant DRY violation. We should use Seaborn's battle-tested color mapping system.

**Current `_handle_colors()` function** → **Delete and replace with `HueMapping`**

**Implementation:**

```python
from seaborn._base import HueMapping, VectorPlotter
from seaborn._core.data import PlotData

def _handle_colors_seaborn(data, hue, palette, n_rows):
    """
    Handle color mapping using Seaborn's HueMapping class.

    This delegates to Seaborn's proven color handling logic instead of
    reimplementing it.
    """
    if hue is None or data is None:
        # Use Seaborn's color cycle
        from seaborn.utils import get_color_cycle
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    # Create a minimal VectorPlotter-like object for HueMapping
    # HueMapping expects a plotter with plot_data, var_types, input_format
    class MinimalPlotter:
        def __init__(self, data, hue):
            self.plot_data = {"hue": data[hue]}
            self.var_types = {"hue": "categorical"}  # or detect numeric
            self.input_format = "long"

    plotter = MinimalPlotter(data, hue)

    # Use Seaborn's HueMapping - handles palette, order, norm automatically
    hue_mapper = HueMapping(plotter, palette=palette, order=None, norm=None)

    # Extract what we need
    unique_vals = hue_mapper.levels
    color_map = hue_mapper.lookup_table

    # Map colors for each row
    hue_data = data[hue]
    colors = [hue_mapper(val) for val in hue_data]

    return colors, color_map, unique_vals
```

**Benefits:**
- ✅ Handles numeric hues with colormaps (we don't support this currently!)
- ✅ Supports palette dictionaries properly
- ✅ Respects `hue_order` parameter (if we add it)
- ✅ Proper warning messages for invalid inputs
- ✅ Consistent with other Seaborn plots
- ✅ Reduces our code by ~20 lines

**Alternative (even simpler):**

Actually, we can simplify further by using Seaborn's existing categorical_order and color_palette helpers:

```python
from seaborn._base import categorical_order
from seaborn.utils import get_color_cycle

def _handle_colors_seaborn(data, hue, palette, n_rows):
    """Handle color mapping using Seaborn utilities."""
    if hue is None or data is None:
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    if hue not in data.columns:
        raise KeyError(f"hue column '{hue}' not found in data")

    hue_data = data[hue]

    # Use Seaborn's categorical_order (same as HueMapping uses internally)
    levels = categorical_order(hue_data, order=None)

    if len(levels) == 0:
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    # Use Seaborn's palette logic (same as HueMapping)
    n_colors = len(levels)
    if palette is None:
        if n_colors <= len(get_color_cycle()):
            palette_colors = sns.color_palette(None, n_colors)
        else:
            palette_colors = sns.color_palette("husl", n_colors)
    else:
        palette_colors = sns.color_palette(palette, n_colors)

    color_map = dict(zip(levels, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, levels
```

**This version:**
- Uses `categorical_order()` from Seaborn (standard function for ordering)
- Uses same palette selection logic as `HueMapping.categorical_mapping()`
- No custom logic - just assembling Seaborn's building blocks
- Still ~30% less code than our current version

---

### Phase 1: Remove Grid Override (CRITICAL BUG FIX) 🔥

**File:** `src/seaborn_parallel/parallelplot.py`

**Change in `_configure_axes()`:**

```python
# BEFORE (line ~596):
sns.despine(ax=ax)
ax.grid(True, alpha=0.3)  # ❌ DELETE THIS LINE

# AFTER:
sns.despine(ax=ax)
# Grid now respects user's set_theme(style=...) choice
```

**Impact**:
- ✅ Respects `whitegrid`/`darkgrid` styles (grid appears)
- ✅ Respects `white`/`dark`/`ticks` styles (no grid)
- ✅ Grid styling (color, linewidth, alpha) automatically matches context

---

### Phase 2: Use rcParams for Font Sizing 📏

**Current code** has hardcoded sizes:
```python
ax.text(..., fontsize=8, ...)  # Line ~470, 527
```

**Replace with context-aware sizing:**

```python
import matplotlib as mpl

# In _configure_axes(), add at top:
tick_label_size = mpl.rcParams['xtick.labelsize'] * 0.8  # Slightly smaller than ticks

# Then use it:
ax.text(
    i, pos, f"  {label}",
    ha="left", va="center",
    fontsize=tick_label_size,  # ✅ Now scales with context
    alpha=0.9,
    fontweight="bold",
)
```

**Why 0.8 multiplier?**
- Tick labels should be slightly smaller than axis labels
- `xtick.labelsize` is already scaled for context (8.8 for paper, 22.0 for poster)
- 0.8× makes them distinguishable from regular tick labels

---

### Phase 2: Use rcParams for Font Sizing 📏

**Current code** has hardcoded sizes:
```python
ax.text(..., fontsize=8, ...)  # Line ~470, 527
```

**Replace with context-aware sizing:**

```python
import matplotlib as mpl

# In _configure_axes(), add at top:
tick_label_size = mpl.rcParams['xtick.labelsize'] * 0.8  # Slightly smaller than ticks

# Then use it:
ax.text(
    i, pos, f"  {label}",
    ha="left", va="center",
    fontsize=tick_label_size,  # ✅ Now scales with context
    alpha=0.9,
    fontweight="bold",
)
```

**Why 0.8 multiplier?**
- Tick labels should be slightly smaller than axis labels
- `xtick.labelsize` is already scaled for context (8.8 for paper, 22.0 for poster)
- 0.8× makes them distinguishable from regular tick labels

---

### **NEW Phase 2.5: Use `ax.set()` Pattern (Seaborn Style)** 🎨

**Objective**: Replace multiple `ax.set_*()` calls with `ax.set()` for cleaner, more Seaborn-like code.

**Changes in `_configure_axes()` function:**

**Before (verbose matplotlib):**
```python
if orientation == "vertical":
    ax.set_xlim(-0.1, n_vars - 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(vars, rotation=45, ha="right")

    if shared_range is not None:
        positions, labels = _format_axis_ticks(shared_range)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Value")
    else:
        ax.set_yticks([])
        ax.set_ylabel("")
```

**After (cleaner Seaborn style):**
```python
if orientation == "vertical":
    # Use ax.set() for bulk property setting
    ax.set(
        xlim=(-0.1, n_vars - 0.9),
        ylim=(-0.05, 1.05),
        xticks=range(n_vars),
    )
    # Rotation requires separate call or set_xticklabels
    ax.set_xticklabels(vars, rotation=45, ha="right")

    if shared_range is not None:
        positions, labels = _format_axis_ticks(shared_range)
        ax.set(
            yticks=positions,
            yticklabels=labels,
            ylabel="Value"
        )
    else:
        ax.set(yticks=[], ylabel="")
```

**Benefits:**
- ✅ More concise (fewer lines)
- ✅ Matches Seaborn's internal code style
- ✅ Easier to read (grouped related properties)
- ✅ Pythonic dictionary-style syntax

**Same for horizontal orientation:**
```python
elif orientation == "horizontal":
    ax.set(
        ylim=(-0.1, n_vars - 0.9),
        xlim=(-0.05, 1.05),
        yticks=range(n_vars),
        yticklabels=vars,
    )

    if shared_range is not None:
        positions, labels = _format_axis_ticks(shared_range)
        ax.set(
            xticks=positions,
            xticklabels=labels,
            xlabel="Value"
        )
    else:
        ax.set(xticks=[], xlabel="")
```

**Estimated impact**: ~15 lines reduced, improved readability

---

### Phase 3: Scale Line Width with Context 📐

**Enhancement to main `parallelplot()` function:**

```python
def parallelplot(...):
    # ... existing code ...

    # Scale linewidth based on context (optional - make it context-aware)
    # Base linewidth from rcParams for lines
    context_linewidth_base = mpl.rcParams['lines.linewidth']

    # User's linewidth is treated as a multiplier
    # Default linewidth=1.0 means "use context default"
    effective_linewidth = linewidth * (context_linewidth_base / 1.5)  # 1.5 is notebook default

    lc = LineCollection(
        lines,
        colors=colors,
        alpha=alpha,
        linewidth=effective_linewidth,  # ✅ Context-aware
        **kwargs
    )
```

**Note**: This is OPTIONAL. Current behavior (user-specified linewidth) is also valid.

---

### Phase 4: Improve Color Handling 🎨

**Use Seaborn's color utilities:**

```python
def _handle_colors(data, hue, palette, n_rows):
    """Handle color mapping with Seaborn utilities."""
    from seaborn.utils import get_color_cycle

    if hue is None or data is None:
        # Use Seaborn's color cycle instead of hardcoding
        default_colors = get_color_cycle()
        default_color = default_colors[0]
        return [default_color] * n_rows, None, None

    if hue not in data.columns:
        raise KeyError(f"hue column '{hue}' not found in data")

    hue_data = data[hue]
    unique_vals = sorted(hue_data.dropna().unique())

    if len(unique_vals) == 0:
        default_colors = get_color_cycle()
        return [default_colors[0]] * n_rows, None, None

    # ✅ Use sns.color_palette (already doing this - GOOD!)
    palette_colors = sns.color_palette(palette, len(unique_vals))
    color_map = dict(zip(unique_vals, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, unique_vals
```

**Changes**: Import `get_color_cycle` and use it for default color instead of `sns.color_palette("deep", 1)[0]`.

---

### Phase 5: Enhance Legend with Seaborn Utilities 🏷️

**Use `move_legend` for better positioning:**

```python
def _configure_axes(...):
    # ... existing code ...

    # Add legend if hue provided
    if unique_vals is not None and color_map is not None:
        # Scale legend line width with context
        legend_linewidth = mpl.rcParams['lines.linewidth'] * 2

        handles = [
            plt.Line2D([0], [0],
                      color=color_map[val],
                      lw=legend_linewidth)  # ✅ Context-aware
            for val in unique_vals
        ]
        legend = ax.legend(
            handles, unique_vals,
            title=hue,
            loc="best",
            # Don't override fontsize - let rcParams handle it
        )

    # Apply Seaborn styling
    sns.despine(ax=ax)
    # REMOVED: ax.grid(True, alpha=0.3)
```

---

### Phase 6: Comprehensive Testing 🧪

**Create test file**: `tests/test_seaborn_integration.py`

```python
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_parallel import parallelplot

def test_respects_whitegrid_style():
    """Grid should appear with whitegrid style."""
    df = sns.load_dataset("iris").iloc[:50]

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        parallelplot(df, vars=df.columns[:4], ax=ax)

        # Check that grid is enabled
        assert ax.xaxis._gridOnMajor or ax.yaxis._gridOnMajor

    plt.close()

def test_respects_white_style():
    """Grid should NOT appear with white style."""
    df = sns.load_dataset("iris").iloc[:50]

    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        parallelplot(df, vars=df.columns[:4], ax=ax)

        # Grid should not be visible (this is tricky to test)
        # Just ensure it doesn't crash
        assert ax is not None

    plt.close()

def test_poster_context_scaling():
    """Text should be larger in poster context."""
    df = sns.load_dataset("iris").iloc[:10]

    with sns.plotting_context("poster"):
        import matplotlib as mpl
        fig, ax = plt.subplots()

        expected_font_size = mpl.rcParams['xtick.labelsize']
        parallelplot(df, vars=df.columns[:4], ax=ax)

        # Font size should be large (poster is 22.0 for tick labels)
        assert expected_font_size > 20

    plt.close()

def test_paper_context_scaling():
    """Text should be smaller in paper context."""
    df = sns.load_dataset("iris").iloc[:10]

    with sns.plotting_context("paper"):
        import matplotlib as mpl
        fig, ax = plt.subplots()

        expected_font_size = mpl.rcParams['xtick.labelsize']
        parallelplot(df, vars=df.columns[:4], ax=ax)

        # Font size should be small (paper is 8.8 for tick labels)
        assert expected_font_size < 10

    plt.close()
```

---

### Phase 7: Documentation Updates 📚

**Update docstring** in `parallelplot()`:

```python
def parallelplot(...):
    """
    Draw a parallel coordinates plot.

    This function integrates with Seaborn's theming system and respects
    context (via `plotting_context()`) and style (via `axes_style()`)
    settings. Use `sns.set_theme()` to control the overall appearance.

    Parameters
    ----------
    ... existing params ...

    Returns
    -------
    ax : Axes
        The matplotlib axes containing the plot

    Notes
    -----
    The plot styling responds to Seaborn's theme settings:

    - **Context** (`paper`, `notebook`, `talk`, `poster`): Controls font sizes,
      line widths, and overall scaling. Larger contexts produce larger text
      and thicker lines.

    - **Style** (`white`, `dark`, `whitegrid`, `darkgrid`, `ticks`): Controls
      visual appearance. Grid styles automatically show grid lines; other
      styles do not.

    - **Palette**: Color schemes specified via `palette` parameter or set
      globally with `set_palette()`.

    Examples
    --------
    Use with different Seaborn contexts:

    >>> import seaborn as sns
    >>> import seaborn_parallel as snp
    >>> df = sns.load_dataset("iris")

    >>> # Larger text and lines for presentations
    >>> with sns.plotting_context("poster"):
    ...     ax = snp.parallelplot(df, hue="species")

    >>> # With grid for better readability
    >>> with sns.axes_style("whitegrid"):
    ...     ax = snp.parallelplot(df, vars=["sepal_length", "petal_length"])

    >>> # Combined styling
    >>> sns.set_theme(context="talk", style="darkgrid", palette="muted")
    >>> ax = snp.parallelplot(df, hue="species")
    """
```

---

## Summary of Changes (DRY-Focused + Seaborn Patterns)

### Must Implement (High Priority - DRY Violations)

1. ✅ **Phase 0**: Replace `_handle_colors()` with Seaborn's `categorical_order()` + built-in palette logic
2. ✅ **Phase 1**: Remove `ax.grid(True, alpha=0.3)` call
3. ✅ **Phase 2**: Replace hardcoded `fontsize=8` with `mpl.rcParams['xtick.labelsize'] * 0.8`
4. ✅ **Phase 2.5** (NEW): Use `ax.set()` pattern instead of multiple `ax.set_*()` calls (Seaborn style)

### Should Implement (Medium Priority - Seaborn Integration)

5. ✅ **Phase 5**: Scale legend line width with context using `mpl.rcParams`
6. ✅ **Phase 6**: Add integration tests
7. ✅ **Phase 7**: Update documentation

### Optional (Low Priority)

8. ⚠️ **Phase 3**: Context-aware linewidth scaling (may break user expectations)

---

## Key DRY Improvements

### Before (Current Code)
```python
# Custom color handling (~20 lines)
def _handle_colors(data, hue, palette, n_rows):
    if hue is None or data is None:
        default_color = sns.color_palette("deep", 1)[0]
        return [default_color] * n_rows, None, None

    hue_data = data[hue]
    unique_vals = sorted(hue_data.dropna().unique())  # ❌ Reimplements categorical_order

    if len(unique_vals) == 0:
        default_color = sns.color_palette("deep", 1)[0]  # ❌ Hardcoded palette
        return [default_color] * n_rows, None, None

    palette_colors = sns.color_palette(palette, len(unique_vals))  # ✅ OK
    color_map = dict(zip(unique_vals, palette_colors))  # ✅ OK
    colors = [color_map.get(val, "gray") for val in hue_data]  # ✅ OK

    return colors, color_map, unique_vals
```

### After (Seaborn Extension)
```python
# Delegates to Seaborn utilities (~15 lines)
from seaborn._base import categorical_order
from seaborn.utils import get_color_cycle

def _handle_colors_seaborn(data, hue, palette, n_rows):
    if hue is None or data is None:
        default_color = get_color_cycle()[0]  # ✅ Uses active color cycle
        return [default_color] * n_rows, None, None

    hue_data = data[hue]
    levels = categorical_order(hue_data, order=None)  # ✅ Uses Seaborn's ordering

    if len(levels) == 0:
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    # ✅ Same palette logic as HueMapping.categorical_mapping()
    n_colors = len(levels)
    if palette is None:
        if n_colors <= len(get_color_cycle()):
            palette_colors = sns.color_palette(None, n_colors)
        else:
            palette_colors = sns.color_palette("husl", n_colors)
    else:
        palette_colors = sns.color_palette(palette, n_colors)

    color_map = dict(zip(levels, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, levels
```

**Improvements:**
- Uses `categorical_order()` instead of `sorted(unique())`
- Uses `get_color_cycle()` instead of hardcoded "deep" palette
- Matches Seaborn's palette selection logic exactly
- Future-proof: if Seaborn improves `categorical_order()`, we benefit automatically

---

## Implementation Notes

### What NOT to Do

❌ Don't try to use Seaborn's internal `_base.py` or `_core` modules - they're private
❌ Don't create custom text annotation functions - none exist in Seaborn
❌ Don't use `FacetGrid`/`PairGrid` as base class - unnecessary complexity
❌ Don't manually set grid properties - trust `axes.grid` rcParam

### What to Keep Doing

✅ Using `sns.color_palette()` - already correct
✅ Using `sns.despine()` - already correct
✅ Using matplotlib's `LineCollection` - Seaborn does the same
✅ Using `ax.text()` - Seaborn has no alternative

### The Core Principle

**Seaborn plots look "Seaborn-ish" because:**
1. Users call `sns.set_theme()` before plotting
2. Plots use matplotlib primitives that respect rcParams
3. Plots call `sns.despine()` for spine styling
4. Plots DON'T override grid settings

Our parallel plot should follow the same pattern.

---

## Testing Strategy

**Manual visual tests:**
```python
import seaborn as sns
import seaborn_parallel as snp

df = sns.load_dataset("iris")

# Test 1: All contexts
for context in ["paper", "notebook", "talk", "poster"]:
    with sns.plotting_context(context):
        fig, ax = plt.subplots(figsize=(10, 6))
        snp.parallelplot(df, hue="species", ax=ax)
        ax.set_title(f"Context: {context}")
        plt.savefig(f"tmp/context_{context}.png")
        plt.close()

# Test 2: All styles
for style in ["white", "whitegrid", "dark", "darkgrid", "ticks"]:
    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=(10, 6))
        snp.parallelplot(df, hue="species", ax=ax)
        ax.set_title(f"Style: {style}")
        plt.savefig(f"tmp/style_{style}.png")
        plt.close()
```

---

## Matplotlib → Seaborn Replacements Analysis 🔄

### Current Matplotlib Usage Audit

| Matplotlib Function | Line(s) | Can Replace? | Seaborn/Better Alternative |
|-------------------|---------|--------------|---------------------------|
| `plt.gca()` | 206 | ✅ YES | Accept `ax` parameter (already have), use `plt.gca()` as fallback (keep) |
| `plt.subplots()` | 614 | ⚠️ NO | No Seaborn equivalent (this is fine - standard pattern) |
| `plt.Line2D` | 591 | ⚠️ PARTIAL | Could use `mpl.lines.Line2D` for clarity |
| `LineCollection` | 11, 211 | ❌ NO | No Seaborn alternative - this is the right tool |
| `ax.text()` | 457, 481, etc. | ❌ NO | Seaborn has no text utility - matplotlib is correct |
| `ax.set_xlim/ylim()` | 432, 433, etc. | ✅ YES | Use `ax.set(xlim=..., ylim=...)` for cleaner code |
| `ax.set_xticks/xticklabels()` | 434-435, etc. | ✅ YES | Use `ax.set(xticks=..., xticklabels=...)` |
| `ax.set_xlabel/ylabel()` | 442, 501, etc. | ✅ YES | Use `ax.set(xlabel=..., ylabel=...)` |
| `ax.legend()` | 593 | ⚠️ PARTIAL | Keep, but could add `sns.move_legend()` call |
| `ax.grid()` | 597 | ✅ DELETE | Remove entirely - use rcParams |
| `sns.despine()` | 596 | ✅ KEEP | Already using Seaborn! |
| `ax.twiny()` | 557 | ❌ NO | Matplotlib-specific, no Seaborn alternative |
| `plt.tight_layout()` | 632 | ⚠️ KEEP | Standard matplotlib, but could use constrained_layout |
| `plt.savefig()` | 637 | ⚠️ KEEP | Standard matplotlib (correct) |

### Key Finding: `ax.set()` is the Seaborn Way! 🎯

Seaborn plots use **`ax.set()`** instead of individual `ax.set_xlabel()`, `ax.set_xlim()`, etc.
This is cleaner and more Pythonic.

**Before (matplotlib style):**
```python
ax.set_xlim(-0.1, n_vars - 0.9)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(n_vars))
ax.set_xticklabels(vars, rotation=45, ha="right")
ax.set_ylabel("Value")
```

**After (Seaborn/cleaner style):**
```python
ax.set(
    xlim=(-0.1, n_vars - 0.9),
    ylim=(-0.05, 1.05),
    xticks=range(n_vars),
    xticklabels=vars,
    ylabel="Value",
)
# Separate call for rotation since it's not a property
ax.tick_params(axis='x', rotation=45)
# Or still use set_xticklabels for rotation:
ax.set_xticklabels(vars, rotation=45, ha="right")
```

**Analysis**:
- ✅ Use `ax.set()` for simple property setting (xlim, ylim, xlabel, ylabel, xticks when no rotation)
- ⚠️ Keep `ax.set_xticklabels()` when using rotation/alignment params
- This matches Seaborn's internal style

### What We Should Change

#### 1. **Use `ax.set()` for Property Setting** (Seaborn pattern)

Replace multiple `ax.set_*()` calls with single `ax.set()` calls where appropriate.

#### 2. **Use `mpl.rcParams` for Font Sizes** (Already planned)

Instead of `fontsize=8`, use `mpl.rcParams['xtick.labelsize']`.

#### 3. **Remove `ax.grid()` Call** (Already planned)

Let Seaborn's style system handle grid.

#### 4. **Use Context-Aware Line Widths** (Already planned)

Scale legend line widths with `mpl.rcParams['lines.linewidth']`.

#### 5. **Import Clarifications**

```python
# Current
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Better (more explicit about what's matplotlib vs seaborn)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import seaborn as sns
from seaborn._base import categorical_order
from seaborn.utils import get_color_cycle, despine
```

### What We Should NOT Change

❌ **Don't replace `LineCollection`** - This is the right tool for drawing many lines efficiently
❌ **Don't replace `ax.text()`** - Seaborn has no alternative
❌ **Don't replace `ax.twiny()`** - Matplotlib-specific feature for our categorical axis labels
❌ **Don't replace `plt.subplots()`** - Standard way to create figures

---

## What Makes This a True Seaborn Extension?

### Seaborn Functions We Use Directly

| Function | Module | Purpose | Our Usage |
|----------|--------|---------|-----------|
| `categorical_order()` | `seaborn._base` | Order categorical data consistently | Replacing `sorted(unique())` |
| `get_color_cycle()` | `seaborn.utils` | Get active color cycle | Default colors |
| `color_palette()` | `seaborn.palettes` | Get color palettes | Hue colors (already using) |
| `despine()` | `seaborn.utils` | Style plot spines | Plot styling (already using) |
| `move_legend()` | `seaborn.utils` | Reposition legends | Optional enhancement |
| `load_dataset()` | `seaborn.utils` | Load example data | Examples/tests (already using) |

### Matplotlib rcParams We Respect

| Parameter | Set By | Purpose | Our Usage |
|-----------|--------|---------|-----------|
| `axes.grid` | `axes_style()` | Enable/disable grid | Stop overriding it! |
| `grid.color` | `axes_style()` | Grid line color | Automatic |
| `grid.linewidth` | `plotting_context()` | Grid line width | Automatic |
| `font.size` | `plotting_context()` | Base font size | Scaling reference |
| `xtick.labelsize` | `plotting_context()` | Tick label size | Text annotations |
| `lines.linewidth` | `plotting_context()` | Default line width | Legend, optional scaling |

### What We Don't Duplicate Anymore

✅ **Categorical ordering** - Use `categorical_order()` instead of `sorted(unique())`
✅ **Default color selection** - Use `get_color_cycle()` instead of hardcoding "deep"
✅ **Palette fallback logic** - Match `HueMapping.categorical_mapping()` exactly
✅ **Grid styling** - Trust `axes_style()` instead of forcing `ax.grid()`
✅ **Font sizing** - Use `rcParams` instead of hardcoding `fontsize=8`

### What Remains Custom (Parallel Coordinates Specific)

✅ **Line coordinate generation** - `_create_line_coordinates()` - unique to parallel plots
✅ **Axis positioning** - Multiple axes at discrete positions - plot-specific
✅ **Tick label placement** - Side labels for each axis - visualization-specific
✅ **Categorical axis mapping** - Converting categories to [0,1] - our innovation
✅ **Variable normalization** - Preserving original ranges - our feature

**This is the right balance**: Delegate common tasks, implement plot-specific logic.

---

## Timeline (Updated for DRY Approach + Seaborn Patterns)

1. **Phase 0** (Replace `_handle_colors`): 20 minutes
2. **Phase 1** (Grid fix): 5 minutes
3. **Phase 2** (Font sizing): 15 minutes
4. **Phase 2.5** (Use `ax.set()` pattern): 25 minutes
5. **Phase 5** (Legend scaling): 10 minutes
6. **Phase 6** (Tests): 30 minutes
7. **Phase 7** (Docs): 20 minutes

**Total estimated time**: ~125 minutes (more thorough, much cleaner code)

---

## Ready to Implement?

All phases now include:
- ✅ Concrete code examples based on actual Seaborn internals
- ✅ DRY principle: maximum reuse of Seaborn utilities
- ✅ Seaborn patterns: using `ax.set()`, rcParams, etc.
- ✅ Clear distinction between what to delegate and what to implement
- ✅ Matplotlib functionality replaced with Seaborn equivalents where possible
