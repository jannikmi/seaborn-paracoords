# Seaborn Objects Parallel Coordinates - Implementation Plan

## Overview

**Goal**: Replace current implementation with Seaborn Objects + post-processing for independent axes

**Approach**:
1. Normalize data to [0,1]
2. Plot with Seaborn Objects
3. Post-process axes to add per-variable tick labels

**Status**: Prototype - full replacement, no backward compatibility needed

---

## Core Architecture

### Data Flow

```
DataFrame → Normalize [0,1] → Melt to tidy → so.Plot() → Render → Post-process axes → plt.Axes
```

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Return type | Always `plt.Axes` | Consistent with Seaborn axis-level functions |
| Orientation | `orient='v'/'h'/'x'/'y'` | Follow Seaborn convention (breaking change OK) |
| Faceting | Out of scope | Document limitation, users can open issue/PR |
| Grid lines | Respect Seaborn style | Let style control grid rendering |
| Tick formatting | Import `seaborn.utils.locator_to_legend_entries()` | Zero code duplication |
| kwargs | Pass to `so.Lines()` | Let Seaborn Objects handle validation |
| Legend | Let SO handle | Automatic positioning |

---

## Implementation Phases

### Phase 1: Core Foundation (Vertical + Independent Axes)

**Goal**: Basic working implementation with vertical orientation

**Tasks**:

1. **Data normalization** (`_normalize_data`)
   - Normalize numeric columns to [0,1] independently or globally based on `sharey`
   - Store original ranges `{col: (min, max)}`
   - Handle constant columns → 0.5 + warning
   - Return: `normalized_df, original_ranges, categorical_info`

2. **Seaborn Objects plot** (`_create_seaborn_plot`)
   - Melt data: `id_vars=[hue], var_name='variable', value_name='value'`
   - Create: `so.Plot(melted, x='variable', y='value', color=hue)`
   - Add: `.add(so.Lines(alpha=alpha, linewidth=linewidth, **kwargs), group='index')`
   - Return: Plot object

3. **Post-processing** (`_add_independent_tick_labels_vertical`)
   ```python
   from seaborn.utils import locator_to_legend_entries
   from matplotlib.ticker import MaxNLocator

   # For each variable at position x_pos:
   for var, x_pos in enumerate(vars):
       min_val, max_val = original_ranges[var]

       # Generate ticks using Seaborn's utility
       locator = MaxNLocator(nbins=6)
       tick_vals, tick_labels = locator_to_legend_entries(
           locator, (min_val, max_val), data[var].dtype
       )

       # Normalize to [0,1] for positioning
       norm_ticks = (tick_vals - min_val) / (max_val - min_val)

       # Draw axis line
       ax.plot([x_pos, x_pos], [0, 1], 'k-', lw=1.5, clip_on=False, zorder=100)

       # Add ticks and labels
       for val, label, pos in zip(tick_vals, tick_labels, norm_ticks):
           ax.plot([x_pos-0.02, x_pos], [pos, pos], 'k-', lw=1, clip_on=False, zorder=100)
           ax.text(x_pos-0.04, pos, label, ha='right', va='center', fontsize=9, clip_on=False)

   # Remove shared y-axis
   ax.set_yticks([])
   ax.spines['left'].set_visible(False)
   ```

**Deliverable**: Working `parallelplot(data, orient='v', sharey=False)` with independent axes

---

### Phase 2: Full Orientation + Shared Scaling

**Goal**: Support both orientations and shared scaling

**Tasks**:

1. **Horizontal orientation** (`_add_independent_tick_labels_horizontal`)
   - Mirror vertical logic: swap x/y axes
   - Vertical axis lines, horizontal tick marks
   - Remove shared x-axis when `sharex=False`

2. **Shared scaling** (`_apply_shared_scaling`)
   ```python
   if sharey (vertical) or sharex (horizontal):
       global_min = min(data[vars].min())
       global_max = max(data[vars].max())
       # Normalize all columns to same [global_min, global_max] range
       # Let Seaborn Objects render naturally (no post-processing)
   ```

3. **Conditional post-processing**
   ```python
   if orient in ['v', 'y'] and not sharey:
       _add_independent_tick_labels_vertical(...)
   elif orient in ['h', 'x'] and not sharex:
       _add_independent_tick_labels_horizontal(...)
   # else: standard SO output, no post-processing
   ```

**Deliverable**: Both orientations + shared/independent scaling work

---

### Phase 3: Categorical Variables

**Goal**: Support categorical variables alongside numeric

**Tasks**:

1. **Categorical encoding** (`_encode_categoricals`)
   ```python
   from seaborn import categorical_order

   for var in categorical_vars:
       if category_orders and var in category_orders:
           order = category_orders[var]
       else:
           order = categorical_order(data[var])

       # Encode as integer codes
       data[var] = pd.Categorical(data[var], categories=order).codes

       # Store category info
       categorical_info[var] = {'categories': order, 'type': 'categorical'}

   # Normalize codes to [0, 1] with uniform spacing
   data[var] = data[var] / (len(order) - 1)
   ```

2. **Categorical tick labels** (`_add_categorical_tick_labels`)
   ```python
   # Similar to numeric, but use category names instead of formatted numbers
   categories = categorical_info[var]['categories']
   positions = np.linspace(0, 1, len(categories))

   for cat, pos in zip(categories, positions):
       # Draw tick and add category name as label
       ...
   ```

3. **Mixed handling**
   - Process numeric and categorical separately
   - Both normalized to [0,1]
   - Post-processing handles both types

**Deliverable**: Mixed numeric/categorical variables work

---

### Phase 4: Polish & Testing

**Goal**: Production-ready code

**Tasks**:

1. **Documentation**
   - Update docstring with all parameters
   - Add examples: basic, with hue, categorical, styling
   - Document limitations:
     - Faceting not supported (open issue/PR if needed)
     - Static tick labels (no zoom/pan updates)
   - Document `orient` parameter change (breaking)

2. **Testing**
   ```python
   # Unit tests
   test_normalize_data()  # handles constant cols, NaN, inf
   test_encode_categoricals()  # respects category_orders
   test_tick_generation()  # uses seaborn.utils correctly

   # Integration tests
   test_vertical_orientation()
   test_horizontal_orientation()
   test_with_hue()
   test_without_hue()
   test_sharey_true()
   test_sharey_false()
   test_categorical_vars()
   test_mixed_vars()

   # Visual regression tests
   - Compare against reference images
   - Test with different Seaborn styles (whitegrid, dark, etc.)
   ```

3. **Code cleanup**
   - Remove old implementation
   - Add type hints
   - Input validation (empty data, invalid orient, etc.)
   - Edge cases (single column, all NaN, etc.)

**Deliverable**: Fully tested, documented implementation

---

## Function Signature

```python
def parallelplot(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    hue: Optional[str] = None,
    orient: Literal['v', 'h', 'x', 'y'] = 'v',
    alpha: float = 0.6,
    linewidth: float = 1.0,
    palette: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
    categorical_axes: Optional[List[str]] = None,
    category_orders: Optional[Dict[str, List]] = None,
    **kwargs
) -> plt.Axes:
    """
    Draw parallel coordinates plot with Seaborn Objects integration.

    Parameters
    ----------
    data : DataFrame
        Input data
    vars : list of str, optional
        Variables to plot (default: all columns)
    hue : str, optional
        Variable for color encoding
    orient : {'v', 'h', 'x', 'y'}, default 'v'
        Orientation: 'v'/'y' for vertical, 'h'/'x' for horizontal
    alpha : float, default 0.6
        Line transparency
    linewidth : float, default 1.0
        Line width
    palette : str, optional
        Seaborn color palette
    ax : Axes, optional
        Matplotlib axes to plot on
    sharex : bool, default False
        Share x-axis range (horizontal orientation only)
    sharey : bool, default False
        Share y-axis range (vertical orientation only)
    categorical_axes : list of str, optional
        Variables to treat as categorical
    category_orders : dict, optional
        Category order for categorical variables
    **kwargs
        Passed to so.Lines()

    Returns
    -------
    ax : Axes
        Matplotlib axes

    Notes
    -----
    - Faceting not supported (open issue if needed)
    - Custom tick labels are static (won't update on zoom/pan)
    - Breaking change: uses `orient` instead of `orientation`
    """
```

---

## Seaborn Utilities to Import (Zero Duplication)

```python
from seaborn.utils import locator_to_legend_entries  # Tick generation
from seaborn.utils import axis_ticklabels_overlap     # Optional: collision detection
from seaborn import categorical_order                 # Categorical ordering
import seaborn.objects as so
from matplotlib.ticker import MaxNLocator
```

---

## Implementation Checklist

### Phase 1
- [ ] Implement `_normalize_data()` with edge case handling
- [ ] Implement data melting for Seaborn Objects
- [ ] Implement `_create_seaborn_plot()` with SO
- [ ] Implement `_add_independent_tick_labels_vertical()`
- [ ] Test with iris dataset (numeric only)
- [ ] Verify Seaborn style/theme integration works

### Phase 2
- [ ] Implement `_add_independent_tick_labels_horizontal()`
- [ ] Implement shared scaling logic
- [ ] Add conditional post-processing
- [ ] Test both orientations
- [ ] Test sharey=True/False and sharex=True/False

### Phase 3
- [ ] Implement `_encode_categoricals()`
- [ ] Implement `_add_categorical_tick_labels()`
- [ ] Test with tips dataset (mixed types)
- [ ] Verify category_orders parameter works

### Phase 4
- [ ] Write comprehensive docstring
- [ ] Add usage examples to README
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create visual regression tests
- [ ] Document breaking changes
- [ ] Remove old implementation code

---

## Edge Cases to Handle

| Case | Handling |
|------|----------|
| Constant column | Normalize to 0.5, emit warning |
| NaN values | Skip or handle per Seaborn's default |
| Single column | Degenerate case, render with single axis |
| Empty DataFrame | Raise ValueError |
| Invalid orient | Raise ValueError |
| No numeric columns | Raise ValueError (unless all categorical) |
| Duplicate column names | Raise ValueError |

---

## Testing Strategy

1. **Unit tests**: Data transformations (normalize, encode, melt)
2. **Integration tests**: Parameter combinations
3. **Visual tests**: Compare against reference images
4. **Style tests**: Verify theme/style integration
5. **Edge case tests**: All cases in table above

---

## Expected Benefits

✅ Native Seaborn theming integration
✅ Zero code duplication (import utilities)
✅ Consistent API with Seaborn ecosystem
✅ Support for both orientations
✅ Support for categorical variables
✅ Independent axes per variable (key feature)
✅ Clean, maintainable codebase

## Known Limitations

⚠️ Faceting not supported (out of scope)
⚠️ Custom tick labels are static (no zoom/pan updates)
⚠️ Breaking change: `orient` vs `orientation` parameter
