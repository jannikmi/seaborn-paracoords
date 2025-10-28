# Implementation Plan: Seaborn Objects with Post-Plot Axis Adjustment ("Aggressive Workaround")

## Executive Summary

**Goal**: Use Seaborn Objects for core plotting functionality, then post-process matplotlib axes to add independent tick labels when `sharex=False` or `sharey=False`.

**Strategy**: Normalize data → Plot with Seaborn Objects → Adjust axes afterward

**Status**: Prototype phase - **FULL REPLACEMENT** of current implementation

**Result**: Native Seaborn integration with parallel coordinates' independent axis behavior

**Implementation Approach**: Direct replacement with single code path using Seaborn Objects
- No backward compatibility required (prototype phase)
- No performance optimization needed yet
- Breaking changes acceptable

---

## Design Decisions (Resolved)

### ✅ Implementation Approach
**Decision**: Fully replace current implementation
- Single code path using Seaborn Objects
- No backward compatibility constraints (prototype phase)
- Performance optimization deferred to later

### ✅ sharex/sharey Behavior
**Decision**: Control normalization vs independent axes
- `sharex=False` / `sharey=False`: Each numeric variable gets own axis range (e.g., 4.3-7.9, 2.0-4.4)
  - Data normalized to [0,1] internally
  - Custom tick labels show original ranges per variable
  - **Visual heights differ based on actual data ranges** (not all normalized to same height)

- `sharex=True` / `sharey=True`: All numeric variables share same axis range (e.g., 0-8)
  - Find global min/max across all variables
  - Normalize all to same range
  - Show single shared axis with common scale

### ✅ Orientation API
**Decision**: Follow Seaborn convention with `orient` parameter
- `orient="v"` or `orient="y"`: Vertical (default) - variables on x-axis, values on y-axis
- `orient="h"` or `orient="x"`: Horizontal - values on x-axis, variables on y-axis
- Matches `seaborn.catplot()`, `seaborn.boxplot()`, etc.

### ✅ Other Design Choices
- **Constant columns**: Normalize to 0.5, show warning
- **Categorical spacing**: Keep current uniform spacing approach
- **Tick formatting**: Match Seaborn's matplotlib ticker behavior
- **Axis spines**: Remove when using independent ticks (cleaner look)

---

## Core Concept

```python
# User calls:
parallelplot(data, hue='species', sharey=False)

# Internally:
# 1. Each numeric column normalized to [0, 1] independently
# 2. Plot with Seaborn Objects (gets themes, palettes, faceting, etc.)
# 3. Since sharey=False, remove shared y-axis spine
# 4. Add custom tick labels at each x-position showing that variable's original range
# 5. Result: Each variable shows its own scale (4.3-7.9, 2.0-4.4, etc.)
```

---

## Design Decisions (FINALIZED)

### ✅ Q1: Function Return Type

**DECISION**: Always return `plt.Axes`

**Rationale**:
- Matches Seaborn's axis-level functions (`sns.boxplot`, `sns.violinplot`)
- Consistent, predictable API
- Users can still access the underlying Seaborn Objects Plot if needed via direct `so.Plot()` usage
- Simpler implementation (single return type)

---

### ✅ Q2: Faceting + Independent Ticks

**DECISION**: Independent tick labels on **outer facets only**

**Rationale**:
- Matches Seaborn's `FacetGrid` behavior (tick labels on outer facets only)
- Reduces visual clutter while maintaining informativeness
- Inner facets still have tick marks, just no labels
- Familiar pattern for Seaborn users

---

### ✅ Q3: kwargs Handling

**DECISION**: Pass `**kwargs` to `so.Lines()`

**Rationale**:
- Follows Seaborn Objects pattern where styling kwargs go to the Mark
- Allows users to customize line appearance (`linestyle`, `marker`, etc.)
- Seaborn Objects handles validation and error messages
- Reduces code duplication (don't re-implement validation)

**Supported kwargs**:
- `color`, `alpha`, `linewidth` (explicitly handled)
- `linestyle`, `marker`, `markersize` (passed through)
- Any other `so.Lines()` compatible parameters

---

### ✅ Q4: Tick Label Formatting

**DECISION**: Reuse Seaborn/matplotlib tick formatting code

**Rationale**:
- Avoid code duplication
- Leverage matplotlib's smart `MaxNLocator` and `ScalarFormatter`
- Consistent formatting across Seaborn ecosystem
- Automatically handles integers vs floats, scientific notation, etc.

**Implementation approach**:
1. Use `matplotlib.ticker.MaxNLocator(nbins=5)` for tick placement
2. Use `matplotlib.ticker.ScalarFormatter` for tick label formatting
3. Let matplotlib determine optimal precision based on data range
4. This matches what Seaborn does internally for other plot types

---

### Q5: **Should we support both `orientation` and `orient` parameters?**

**Current library**: Uses `orientation='vertical'` or `'horizontal'`

**Seaborn convention**: Uses `orient` parameter
- Takes values: `'v'`, `'h'`, `'x'`, `'y'`
- More concise
- Standard across Seaborn API

**Options**:
1. **Support both** (backwards compatibility)
   - Accept `orientation` AND `orient`
   - Deprecation warning for `orientation`
   - More code complexity

2. **Only `orient`** (clean break)
   - Simpler code
   - Consistent with Seaborn
   - Breaking change

**👉 DECISION**: Only support `orient` parameter (clean break). Since this is a prototype/replacement phase, use this opportunity for full Seaborn API alignment.

---

### Q6: **Palette parameter - follow Seaborn conventions?**

**Seaborn patterns**:
- `palette=None`: Use current default palette
- `palette="deep"`: Named palette
- `palette=["red", "blue"]`: List of colors
- `palette=sns.color_palette()`: Palette object

**Current implementation**: Accepts palette name as string

**👉 RECOMMENDATION**: Keep current behavior - `palette` parameter matches Seaborn convention. No changes needed.

---

### ✅ Q7: Faceting Parameters

**DECISION**: Faceting is **out of scope** for this implementation

**Rationale**:
- Keeps `parallelplot()` as a simple axis-level function
- Avoids complexity of wrapping Seaborn Objects faceting API
- Reduces maintenance burden (faceting API changes don't affect us)
- Post-processing for independent ticks would add significant complexity with facets
- Future enhancement if there's user demand

**Implementation approach**:
- Document in README/docstring that faceting is not currently supported
- Direct users to open an issue or submit a PR if they need this functionality
- No usage examples required (out of scope)

**Documentation note**:
```
Faceting is not currently supported. If you need this feature, please open an issue
or submit a pull request at https://github.com/jannikmi/seaborn-paracoords
```

---

### ✅ Q8: Grid Lines with Independent Axes

**DECISION**: Align with default Seaborn axis labeling behavior; reuse internal library code

**Rationale**:
- Consistency with Seaborn ecosystem (familiar to users)
- Avoid code duplication (leverage existing Seaborn/matplotlib code)
- Benefit from Seaborn's smart defaults for tick placement and formatting
- Easier maintenance (Seaborn handles edge cases)
- Grid behavior respects current Seaborn style settings

**Implementation approach**:

1. **Respect Seaborn style for grid lines**:
   - If style is `whitegrid` or `darkgrid`: Keep grid lines
   - If style is `white`, `dark`, or `ticks`: No grid lines
   - Let Seaborn's existing grid rendering handle this automatically

2. **Reuse Seaborn's internal axis code**:
   ```python
   # Use Seaborn's axis utilities where possible
   from seaborn import utils
   from matplotlib import ticker

   # Let matplotlib/seaborn determine tick positions
   locator = ticker.MaxNLocator(nbins='auto')  # Seaborn default
   formatter = ticker.ScalarFormatter()  # Seaborn default
   ```

3. **For independent axes, draw per-variable axis lines**:
   - Add vertical axis line at each variable position (for vertical orientation)
   - Tick marks extend from the axis line
   - Tick labels positioned using Seaborn's spacing defaults
   - Grid lines (if enabled by style) remain from Seaborn Objects rendering

4. **Avoid custom grid drawing unless necessary**:
   - First attempt: Let Seaborn Objects handle grid naturally
   - Only add custom axis lines/ticks for independent scaling
   - Minimize custom matplotlib drawing code

**Key difference from ultra_aggressive_workaround.png**:
- Don't manually draw grid lines with `axhline()`
- Instead, rely on Seaborn's grid rendering from the style
- Focus custom code on: per-variable axis lines, tick marks, and tick labels only

---

### ✅ Q9: Legend Placement

**DECISION**: Let Seaborn Objects handle legend placement automatically

**Rationale**:
- Seaborn Objects already handles legend positioning intelligently
- Consistent with Seaborn ecosystem behavior
- No additional code needed

---

## Implementation Phases

### Phase 1: Core Foundation (PRIORITY)

**Goal**: Basic SO integration with post-processing for vertical orientation

**Tasks**:

1. Data normalization function
   ```python
   def _normalize_data(data, vars, hue, sharex, sharey)
   → returns: normalized_df, original_ranges, categorical_info
   ```
   - Normalize numeric columns to [0,1] based on sharex/sharey
   - Store original ranges for tick label generation
   - Handle constant columns (normalize to 0.5, emit warning)

2. Seaborn Objects plot creation
   ```python
   def _create_seaborn_plot(melted_data, orient, hue, **kwargs)
   → returns: so.Plot object
   ```
   - Melt data to tidy format for SO
   - Create basic `so.Plot()` with `so.Lines()`
   - Pass kwargs to `so.Lines()`

3. Post-processing for vertical orientation with independent axes
   ```python
   def _add_independent_tick_labels_vertical(ax, vars, original_ranges)
   ```
   - Remove shared y-axis ticks and spine
   - Draw vertical axis line at each variable position
   - Add custom tick marks using matplotlib's tick locators
   - Add tick labels using matplotlib's formatters (reuse Seaborn code)
   - Grid lines handled automatically by Seaborn style (no custom grid drawing)
   - Reuse Seaborn/matplotlib tick formatting code (MaxNLocator, ScalarFormatter)

**Deliverable**: Working vertical numeric plots with independent ticks, aligned with Seaborn defaults

**Questions resolved**: Q1 (return Axes), Q3 (kwargs), Q4 (tick formatting), Q8 (grid)

---

### Phase 2: Full Orientation Support

**Goal**: Both orientations + shared axis support

**Tasks**:

1. Horizontal orientation tick labels
   ```python
   def _add_independent_tick_labels_horizontal(ax, vars, original_ranges)
   ```
   - Mirror vertical implementation for horizontal orientation
   - Vertical grid lines, horizontal axis lines

2. Shared axis implementation
   ```python
   def _apply_shared_scaling(data, vars, sharex, sharey, orientation)
   ```
   - Find global min/max across variables
   - Normalize all variables to same range
   - Let Seaborn Objects handle shared axis naturally

3. Clean up shared axis when not using independent ticks
   - No custom post-processing needed
   - Standard Seaborn Objects output

**Deliverable**: Both orientations work with shared and independent scaling

**Questions resolved**: None (implementation details only)

---

### Phase 3: Categorical Support

**Goal**: Handle categorical variables alongside numeric

**Tasks**:

1. Categorical variable encoding
   ```python
   def _encode_categoricals(data, vars, category_orders)
   ```
   - Encode categorical variables as integer codes
   - Respect category_orders if provided
   - Fall back to Seaborn's categorical_order()

2. Categorical tick labels
   ```python
   def _add_categorical_tick_labels(ax, var_pos, categories, orientation)
   ```
   - Add category names as tick labels
   - Position appropriately for orientation

3. Mixed numeric/categorical handling
   - Normalize numeric to [0,1]
   - Encode categorical to [0,1] with uniform spacing
   - Post-process both types appropriately

**Deliverable**: Numeric + categorical variables work together

**Questions resolved**: Q2 (faceting behavior - though faceting not in function signature)

---

### Phase 4: Polish & Testing

**Goal**: Production-ready code

**Tasks**:

1. Documentation
   - Update docstrings with all parameters
   - Add examples for common use cases
   - Document how to use `so.Plot()` directly for faceting
   - Add example code for faceting workflow

2. Testing
   - Unit tests for data normalization
   - Unit tests for tick label generation
   - Visual regression tests for key scenarios
   - Test both orientations
   - Test with/without hue
   - Test shared vs independent scaling
   - Test categorical variables

3. Code cleanup
   - Remove old implementation code
   - Add type hints
   - Add input validation
   - Handle edge cases (empty data, single column, etc.)

**Deliverable**: Tested, documented, production-ready implementation

**Questions resolved**: Q10 (interactive), Q11 (testing)

---

## Technical Implementation Details

### Data Flow

```
Input: DataFrame
    ↓
1. Separate numeric/categorical columns
    ↓
2. Normalize numeric to [0,1]
    ↓
3. Encode categorical as codes
    ↓
4. Melt to tidy format
    ↓
5. Create Seaborn Objects plot
    ↓
6. Render to matplotlib axes
    ↓
7. If sharex=False or sharey=False:
   a. Remove shared axis ticks
   b. Add custom tick labels per variable
   c. Draw custom tick marks
    ↓
Output: matplotlib Axes (or so.Plot for faceting?)
```

### Key Functions

```python
def parallelplot(data, vars=None, hue=None, orientation='vertical',
                 sharex=False, sharey=False,
                 col=None, row=None,  # Faceting
                 show_mean=False,     # Stats
                 categorical_axes=None, category_orders=None,
                 **kwargs):

    # 1. Normalize data
    normalized_df, original_ranges, categorical_info = _normalize_data(...)

    # 2. Melt for SO
    melted = _melt_for_seaborn(normalized_df, vars, hue)

    # 3. Create SO plot
    plot = _create_seaborn_plot(melted, orientation, hue, ...)

    # 4. Add faceting if requested
    if col or row:
        plot = plot.facet(col=col, row=row)

    # 5. Add stats if requested
    if show_mean:
        plot = plot.add(so.Line(), so.Agg())

    # 6. Render
    rendered = plot.plot()
    ax = _extract_axes(rendered)

    # 7. Post-process for independent ticks
    if (orientation == 'vertical' and not sharey) or \
       (orientation == 'horizontal' and not sharex):
        _remove_shared_axis_ticks(ax, orientation, sharex, sharey)
        _add_independent_tick_labels(ax, vars, original_ranges,
                                     categorical_info, orientation)

    return ax  # or plot? or figure?
```

---

## Final Open Questions

### Q10: **Interactive features - are they important?**

**Issue**: Custom tick labels are static text annotations
- Zooming won't update custom labels
- Panning won't update custom labels
- Not true matplotlib axis ticks

**Seaborn typically**:
- Focuses on publication-quality static plots
- Interactive features are secondary
- Users who need interactivity use Plotly/Bokeh

**👉 RECOMMENDATION**: Accept limitation. Document that custom tick labels are static. Most Seaborn users expect static plots anyway.

---

### Q11: **Testing strategy?**

**Seaborn testing patterns**:
- Visual regression tests for key plot types
- Unit tests for data transformations
- Integration tests for parameter combinations
- Rely on matplotlib/numpy for numerical correctness

**For parallel coordinates**:
- Test data normalization (unit)
- Test tick label placement (unit)
- Test categorical encoding (unit)
- Visual tests for key cases (integration)
- Skip performance benchmarks for now (prototype phase)

**👉 RECOMMENDATION**: Focus on unit tests for data transformations and visual regression tests for main use cases. Performance can be addressed later.

---

---

## Final Decision Summary

### ✅ ALL DECISIONS FINALIZED - Ready for Implementation

**Critical Decisions**:

1. **Q1 - Return Type**: Always `plt.Axes` ✓
2. **Q2 - Faceting + Independent Ticks**: Outer facets only ✓
3. **Q3 - kwargs**: Pass to `so.Lines()` ✓
4. **Q4 - Tick Formatting**: Reuse matplotlib MaxNLocator/ScalarFormatter ✓
5. **Q7 - Faceting Parameters**: NOT in function signature; document `so.Plot()` usage ✓
6. **Q8 - Grid Lines**: KEEP grid with strong axis lines (ultra_aggressive style) ✓

**Design Choices Locked In**:

- ✅ Implementation: Full replacement, no backward compatibility
- ✅ sharex/sharey: Control independent vs shared scaling
- ✅ Orientation: Use `orient` parameter (Seaborn convention)
- ✅ Constant columns: Normalize to 0.5 + warning
- ✅ Categorical spacing: Uniform in [0,1]
- ✅ Axis spines: Remove shared spine when using independent ticks
- ✅ Palette: Follow Seaborn conventions
- ✅ Legend: Let Seaborn Objects handle automatically
- ✅ Interactive: Static labels acceptable (document limitation)
- ✅ Testing: Unit tests + visual regression

---

## Remaining Open Questions

### ✅ NONE - All Questions Resolved

**Q8 Implementation Detail - RESOLVED**:

After investigating Seaborn's internals, discovered `seaborn.utils.locator_to_legend_entries()` which:
- Is the exact function Seaborn uses for tick generation in legends
- Takes a locator and limits, returns formatted tick values and labels
- Handles integer vs float formatting intelligently
- Disables scientific notation for clean labels
- **Available as a public utility** - can be imported and used directly

**Decision**: **Import and use** `locator_to_legend_entries()` from `seaborn.utils` for all tick formatting. This perfectly aligns with Seaborn's internal behavior and **avoids any code duplication**.

**Implementation**:
```python
from seaborn.utils import locator_to_legend_entries  # Import, don't re-implement

# Use it directly
tick_values, tick_labels = locator_to_legend_entries(locator, limits, dtype)
```

All design decisions are now finalized with concrete implementation paths identified using **imported** Seaborn utilities (zero code duplication).

---## Implementation Checklist

Before starting each phase, verify:

- [ ] **Phase 1**: Research Seaborn's internal axis/tick formatting code to reuse
- [ ] **Phase 1**: Understand how Seaborn styles control grid rendering
- [ ] **Phase 1**: Study matplotlib.ticker.MaxNLocator and ScalarFormatter usage in Seaborn
- [ ] **Phase 1**: Plan data normalization edge cases (constant columns, NaN, inf)
- [ ] **Phase 2**: Test horizontal orientation thoroughly (mirror vertical logic)
- [ ] **Phase 3**: Review Seaborn's categorical_order() function
- [ ] **Phase 4**: Write clear documentation that faceting is out of scope
- [ ] **Phase 4**: Create visual regression test suite

---

## Key Implementation References

### ✅ Seaborn Internal Code to Reuse (DISCOVERED)

**1. `seaborn.utils.locator_to_legend_entries()` - Tick Generation & Formatting**

Located in: `seaborn/utils.py`

```python
from seaborn.utils import locator_to_legend_entries
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Generate ticks for a data range
locator = MaxNLocator(nbins=6)  # Seaborn default for legends
limits = (min_val, max_val)
tick_values, tick_labels = locator_to_legend_entries(locator, limits, dtype)
```

**What it does**:
- Takes a matplotlib locator and data limits
- Returns both raw tick values AND formatted labels
- Handles LogLocator and ScalarFormatter intelligently
- Disables scientific notation and offsets (clean labels)
- Clips ticks to stay within limits

**This is EXACTLY what we need for independent axes!**

**2. Seaborn Objects Scale System (`seaborn._core.scales`)**

The `Continuous` scale class shows how Seaborn chooses locators:

```python
# Default (no parameters): AutoLocator()
# With upto parameter: MaxNLocator(upto, steps=[1, 1.5, 2, 2.5, 3, 5, 10])
# With count parameter: LinearLocator(count) or FixedLocator with linspace
# For log scales: LogLocator(base)
```

**Key insight**: Seaborn's default is `AutoLocator()` for continuous scales, which adapts to data range. For brief legends (6 ticks), it uses `MaxNLocator(nbins=6)`.

**3. Additional Seaborn Utilities**

- `seaborn.utils.axis_ticklabels_overlap()` - detect if labels collide
- `seaborn.utils.axes_ticklabels_overlap()` - check all axes
- `seaborn.utils.axlabel()` - consistent axis label formatting

### Implementation Strategy (FINALIZED)

**Import and use Seaborn's existing utilities directly** (NO code duplication):

```python
# IMPORT Seaborn's utility - do NOT re-implement
from seaborn.utils import locator_to_legend_entries
from matplotlib.ticker import MaxNLocator
import numpy as np

# For each variable
for var, x_pos in variable_positions.items():
    min_val, max_val = original_ranges[var]

    # CALL Seaborn's existing function (not re-implemented)
    locator = MaxNLocator(nbins=6)
    tick_values, tick_labels = locator_to_legend_entries(
        locator, (min_val, max_val), data[var].dtype
    )

    # Normalize tick values to [0, 1] for positioning
    normalized_ticks = (tick_values - min_val) / (max_val - min_val)

    # Draw axis line at variable position (custom code - unavoidable)
    ax.plot([x_pos, x_pos], [0, 1], color='black', linewidth=1.5,
            clip_on=False, zorder=100)

    # Add tick marks and labels (custom code - unavoidable)
    for tick_val, tick_label, norm_pos in zip(tick_values, tick_labels, normalized_ticks):
        # Tick mark
        ax.plot([x_pos - 0.02, x_pos], [norm_pos, norm_pos],
                color='black', linewidth=1, clip_on=False, zorder=100)
        # Tick label (using labels generated by Seaborn's function)
        ax.text(x_pos - 0.04, norm_pos, tick_label,
                ha='right', va='center', fontsize=9, clip_on=False)
```

**What we're importing from Seaborn** (zero code duplication):
- ✅ `locator_to_legend_entries()` - tick value and label generation
- ✅ Can also import `axis_ticklabels_overlap()` if needed to detect collisions
- ✅ Can import `categorical_order()` for categorical variable ordering

**What we must implement ourselves** (unavoidable custom code):
- Drawing per-variable axis lines at specific x positions
- Positioning tick marks at normalized locations
- Placing text labels at custom positions
- This is the "post-processing" that makes parallel coordinates work

**Benefits of this approach**:
- ✅ Uses Seaborn's exact tick generation code via **import** (NOT duplication)
- ✅ No code duplication - calling existing, tested utilities
- ✅ Tick formatting matches Seaborn's legend formatting automatically
- ✅ Automatically handles integer vs float formatting
- ✅ No scientific notation or offsets (cleaner labels)
- ✅ Grid lines handled automatically by Seaborn style

### Other Code References

**Current implementation**:
- `src/seaborn_parallel/parallelplot.py` - parameter handling, palette integration

**Seaborn Objects APIs**:
- `so.Plot(data, x=, y=, color=)` - base plot
- `so.Lines(alpha=, linewidth=, **kwargs)` - line mark
- `.add(mark, group=)` - add layer
- `.on(ax)` - render to specific axes
- `.plot()` - render and return

---

---

## Risks & Mitigation (Updated)

| Risk | Impact | Mitigation Status |
|------|--------|------------------|
| Different visual appearance | User confusion | ✅ Acceptable - prototype phase |
| Performance regression | Slow with large data | ✅ Deferred - not critical yet |
| SO API changes | Future breakage | ⚠️ Pin SO version in requirements |
| Static tick labels (zoom/pan) | Interactive UX | ✅ Documented limitation, acceptable for static plots |
| Grid approach complexity | Maintenance burden | ✅ Well-documented in ultra_aggressive_workaround.py |

---

## Summary: No Open Questions - Implementation Ready

**ALL CRITICAL DECISIONS FINALIZED**:

✅ **Q1** - Return type: Always `plt.Axes`
✅ **Q2** - Faceting ticks: Outer facets only (though faceting is out of scope)
✅ **Q3** - kwargs: Pass to `so.Lines()`
✅ **Q4** - Tick formatting: Reuse matplotlib formatters
✅ **Q5** - Orientation: Use `orient` parameter
✅ **Q6** - Palette: Follow Seaborn conventions (no change needed)
✅ **Q7** - Faceting: OUT OF SCOPE (users open issue/PR if needed)
✅ **Q8** - Grid lines: Align with Seaborn defaults, reuse internal code
✅ **Q9** - Legend: Let SO handle
✅ **Q10** - Interactive: Static labels OK
✅ **Q11** - Testing: Unit + visual regression

**Implementation approach**: Full replacement, 4-phase plan documented above

**No blockers** - Ready to begin Phase 1 implementation immediately.
