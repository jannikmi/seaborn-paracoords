# Categorical Axes Feature - Quick Start Guide

## What's New?

The `seaborn-paracoords` library now supports **categorical axes** in parallel coordinates plots! You can now visualize mixed-type datasets containing both numeric and categorical variables.

## Key Features

✨ **Automatic Detection** - Non-numeric columns are automatically treated as categorical  
🎯 **Explicit Control** - Override detection with `categorical_axes` parameter  
📊 **Custom Ordering** - Specify category order with `category_orders` parameter  
🔄 **Full Compatibility** - Works with both orientations and all existing features  

## Quick Examples

### 1. Automatic Detection (Simplest!)

```python
import pandas as pd
import seaborn_parallel as snp

df = pd.DataFrame({
    "species": ["setosa", "versicolor", "virginica", "setosa"],
    "sepal_length": [5.1, 7.0, 6.3, 4.9],
    "petal_width": [0.2, 1.3, 2.5, 0.2]
})

# 'species' is automatically detected as categorical
ax = snp.parallelplot(df, hue="species")
```

### 2. Custom Category Order

```python
df = pd.DataFrame({
    "size": ["small", "large", "medium", "small"],
    "score": [85, 95, 90, 88]
})

ax = snp.parallelplot(
    df,
    category_orders={"size": ["small", "medium", "large"]}
)
```

### 3. Multiple Categorical Axes

```python
df = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "product": ["A", "B", "C", "D"],
    "sales": [100, 150, 200, 250],
    "profit": [20, 30, 40, 50]
})

ax = snp.parallelplot(
    df,
    vars=["region", "product", "sales", "profit"],
    hue="region"
)
```

## Try the Demos!

Run the comprehensive demo script to see all features in action:

```bash
python scripts/demo_categorical_axes.py
```

This generates 5 example plots in `./tmp/`:
- Auto-detection demo
- Explicit specification demo
- Custom ordering demo
- Horizontal orientation demo
- Multiple categorical axes demo

## API Reference

### New Parameters

**`categorical_axes`**: `Optional[List[str]]`
- Explicitly specify which variables are categorical
- Default: `None` (auto-detection based on dtype)

**`category_orders`**: `Optional[dict]`
- Custom ordering of categories
- Format: `{"var_name": ["cat1", "cat2", ...]}`
- Default: `None` (alphabetical order)

### Changed Behavior

⚠️ **Important**: When `vars=None`, the function now selects **all columns** (not just numeric).

**Before:**
```python
snp.parallelplot(df)  # Only numeric columns
```

**Now:**
```python
snp.parallelplot(df)  # All columns (with auto-detection)
```

**To restore old behavior:**
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
snp.parallelplot(df, vars=numeric_cols)
```

## Testing

All features are fully tested! Run the test suite:

```bash
pytest tests/test_parallelplot.py -v
```

**Test Coverage:**
- ✅ Automatic categorical detection
- ✅ Explicit categorical specification
- ✅ Mixed numeric/categorical axes
- ✅ Custom category ordering
- ✅ Vertical and horizontal orientations
- ✅ Integration with `hue` parameter

## Documentation

For detailed implementation information, see:
- **Feature Documentation**: `CATEGORICAL_AXES_FEATURE.md`
- **Changelog**: `CHANGELOG.md`
- **API Docs**: Docstring in `src/seaborn_parallel/parallelplot.py`

## Questions?

Check the demos in `scripts/demo_categorical_axes.py` for complete working examples!

---

**Version**: Unreleased  
**Added**: October 27, 2025  
**Compatibility**: Python 3.8+, pandas 1.0+
