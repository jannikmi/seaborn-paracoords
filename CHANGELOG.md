# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Enhanced customization options for axis styling
- Integration with Seaborn's FacetGrid for multi-plot layouts

## [0.0.5] - 2025-11-10

### Added

- **Axis Flipping Feature**: New `flip` parameter to reverse coordinate axes
  - Accepts list of variable names whose axes should be reversed (flipped)
  - Works with both numeric and categorical variables
  - Compatible with both vertical and horizontal orientations
  - Compatible with shared axes (`sharex`/`sharey`), hue coloring, and `category_orders`
  - Useful for emphasizing negative correlations or aligning scales for visual comparison
  - Example: `parallelplot(iris, flip=['sepal_width'], hue='species')`
  - Validates input and warns about invalid variable names
  - Added 7 comprehensive test cases for flip functionality

- **Enhanced Category Ordering**: Improved `category_orders` parameter behavior
  - Now respects category orders for hue variables even when hue is not in `vars`
  - Better color consistency across different plot configurations
  - More intuitive top-to-bottom ordering (user input is automatically reversed for display)

- **Demo Scripts**: New iris dataset demonstration script
  - `demo_iris.py`: Showcases iris dataset with flip feature and category ordering

### Changed

- **Category Order Reversal**: User-provided category orders are now automatically reversed for intuitive top-to-bottom display
  - When you specify `category_orders={'species': ['setosa', 'versicolor', 'virginica']}`, they appear in that order from top to bottom
  - Internal implementation handles the reversal automatically
  - More natural and expected behavior for users

### Fixed

- **Category Color Ordering**: Fixed color assignment to respect `category_orders` for categorical hue variables
  - Colors now consistently map to categories in the specified order
  - Legend entries appear in the expected order
  - Applies to both cases: when hue is in `vars` and when it's not

### Technical Details

**Implementation of Flip Feature**:
- Normalization phase: After normalizing to [0, 1], inverts flipped variables using `1 - normalized_value`
- Tick generation: Reverses tick positions for flipped axes
- Simple design that works seamlessly with existing rendering pipeline
- No special cases needed for different orientations or scaling modes

**Test Coverage**:
- Basic flip functionality with numeric variables
- Multiple variables flipped simultaneously
- Flip with categorical variables
- Invalid variable name validation
- Both orientations (vertical/horizontal)
- Shared axis compatibility
- Data integrity verification (original data unchanged)

## [0.0.4] - 2025-10-29

### Fixed

- **plt.gcf() integration**: Fixed bug where `plt.gcf()` would return an empty figure when `parallelplot()` was called without explicitly providing an `ax` parameter
  - Previously, Seaborn Objects created its own internal figure separate from matplotlib's current figure
  - Now uses `plt.gca()` when `ax=None` to ensure plot renders to the current matplotlib figure
  - Users can now access and save plots using standard matplotlib commands: `plt.gcf().savefig('plot.png')`
  - Added regression test `test_gcf_contains_plot()` to prevent future occurrences

## [0.0.3] - 2025-10-28

### Changed

- ⚠️ **BREAKING**: Complete reimplementation using **Seaborn Objects** with post-processing
  - Previous implementation used matplotlib's `LineCollection` directly
  - New implementation uses `seaborn.objects.Plot()` and `seaborn.objects.Lines()` for core rendering
  - Post-processing adds independent axis tick labels when `sharex=False` or `sharey=False`
  - **Migration**: Code should work without changes, but visual appearance may differ slightly due to Seaborn Objects rendering

- ⚠️ **BREAKING**: Changed orientation parameter from `orientation` to `orient`
  - Old: `orientation="vertical"` or `orientation="horizontal"`
  - New: `orient="v"` or `orient="h"` (also accepts `orient="x"` and `orient="y"`)
  - Aligns with Seaborn's convention (used in `catplot`, `boxplot`, etc.)
  - **Migration**: Replace `orientation="vertical"` with `orient="v"` and `orientation="horizontal"` with `orient="h"`

- **Improved Seaborn Integration**: Full native integration with Seaborn theming system
  - Respects all Seaborn plotting contexts (`paper`, `notebook`, `talk`, `poster`)
  - Line widths, tick sizes, and font sizes now automatically scale with context
  - Grid lines respect current Seaborn style (`whitegrid`, `darkgrid`, `white`, `dark`, `ticks`)
  - Axis styling matches Seaborn's internal behavior (uses `matplotlib.ticker.MaxNLocator` and `ScalarFormatter`)

- **Enhanced Independent Axis Rendering**
  - Custom axis lines drawn at each variable position with proper thickness
  - Tick marks and labels positioned using Seaborn's `locator_to_legend_entries()` utility
  - Smart tick formatting: integers shown without decimals, floats with appropriate precision
  - No scientific notation or offset text (cleaner labels)
  - Categorical labels automatically rotated 45° in horizontal orientation for readability

### Added

- **Automatic dtype-based categorical detection**
  - Boolean columns (`True`/`False`) now treated as categorical
  - Datetime columns automatically treated as categorical
  - Object and category dtypes continue to work as categorical
  - More intuitive behavior without explicit `categorical_axes` parameter

- **Improved color handling for categorical hue variables**
  - Categorical hue variables now properly mapped to Seaborn palettes
  - Each category gets a distinct color from the palette
  - Consistent color assignment across multiple plots
  - Legend correctly shows all hue categories

- **New demo scripts**
  - `demo_datasets.py`: Showcases multiple datasets (iris, tips, penguins, diamonds)
  - `demo_orientations.py`: Demonstrates vertical vs horizontal orientations
  - `demo_scaling.py`: Shows shared vs independent axis scaling behavior
  - `demo_seaborn_contexts.py`: Demonstrates all Seaborn contexts (paper/notebook/talk/poster)
  - `scripts/README.md`: Documentation for all demo scripts

- **Internal code reuse from Seaborn**
  - Uses `seaborn.utils.locator_to_legend_entries()` for tick generation
  - Leverages `matplotlib.ticker.MaxNLocator` for smart tick placement
  - Follows Seaborn's axis formatting conventions throughout

### Fixed

- **Multi-plot layouts**: Fixed bugs when using `ax` parameter with subplot grids
- **Duplicate legends**: Eliminated extra legends that appeared in some configurations
- **Categorical hue alignment**: Categorical hue variables now correctly map to colors
- **Y-axis inversion**: Fixed upside-down plots in certain configurations
- **Shared axis behavior**: Shared axes now work correctly in both orientations
- **Tick label overlap**: Better spacing prevents label collisions

### Technical Details

**Design Rationale**:

The new implementation follows an "aggressive workaround" strategy to achieve the best of both worlds:

1. **Native Seaborn integration**: Use Seaborn Objects for plotting to get theming, palettes, and consistent API
2. **Independent axes**: Post-process matplotlib axes to add per-variable tick labels showing original data ranges
3. **Data normalization**: Internally normalize data to [0,1] for rendering, then overlay custom tick labels

**Key architectural decisions**:

- **Return type**: Always returns `matplotlib.axes.Axes` (matches Seaborn's axis-level functions like `boxplot()`)
- **Tick formatting**: Reuses Seaborn's `locator_to_legend_entries()` utility for consistency
- **Grid rendering**: Respects current Seaborn style settings automatically
- **Orientation API**: Follows Seaborn convention (`orient` parameter with values `'v'`, `'h'`, `'x'`, `'y'`)
- **kwargs handling**: Passes styling kwargs to `so.Lines()` for maximum flexibility
- **Constant columns**: Normalizes to 0.5 with warning message


**Known limitations**:

- Faceting not currently supported (users should use `so.Plot()` directly for faceting)
- Custom tick labels are static (don't update on interactive zoom/pan)
- Large datasets (>1000 lines) may render slowly due to matplotlib line rendering

**Performance notes**:

- For dense datasets, recommend `alpha=0.3` and `linewidth=0.5`
- Consider sampling data for exploratory analysis of very large datasets
- Shared axes (`sharex=True` or `sharey=True`) render faster with fewer custom elements

### Documentation

- Updated docstring with comprehensive examples and parameter documentation
- Added design rationale in `AGGRESSIVE_WORKAROUND_PLAN.md` (715 lines)
- Created `SEABORN_INTEGRATION_PLAN.md` explaining integration strategy
- Added `SEABORN_OBJECTS_COMPARISON.md` comparing approaches
- Added `SEABORN_WRAPPER_ADVANTAGES.md` documenting benefits
- Updated `TODOs.md` with future enhancement plans
- Added `scripts/README.md` explaining all demo scripts

## [0.0.2] - 2025-10-27

### Added

- **Categorical Axis Support**: Full support for categorical variables in parallel coordinates plots
  - Automatic detection of non-numeric columns as categorical axes
  - `categorical_axes` parameter for explicit categorical axis specification
  - `category_orders` parameter for custom ordering of categories
  - Visual distinction of categorical labels (bold font weight)
  - Seamless integration with both vertical and horizontal orientations
  - Compatibility with `hue`, `sharex`, and `sharey` parameters
- New demo script: `demo_categorical_axes.py` showcasing categorical features with 5 examples
- Comprehensive test coverage for categorical axes:
  - Automatic detection
  - Explicit specification
  - Mixed-type datasets (categorical + numeric)
  - Custom category ordering
- Documentation files:
  - `CATEGORICAL_AXES_FEATURE.md` - Detailed implementation guide
  - `CATEGORICAL_AXES_QUICKSTART.md` - Quick start guide with examples

### Changed

- ⚠️ **BREAKING**: When `vars=None`, `parallelplot()` now selects ALL columns (including categorical) instead of only numeric columns
  - Migration: Explicitly pass numeric columns via `vars` parameter to restore old behavior
  - New behavior enables automatic categorical axis detection
- Updated docstring with new parameters and usage examples
- Enhanced `_format_axis_ticks()` to guard against categorical ranges

## [0.0.1] - 2025-10-27

### Added

- Initial prototype implementation of parallel coordinates plotting
- Support for both vertical and horizontal orientations
- True orientation control (not just data transposition)
- Preservation of original axis values by default (no forced normalization)
- Optional shared axis scaling via `sharex` and `sharey` parameters
- Color encoding support through `hue` parameter
- Integration with Seaborn color palettes
- Customizable line transparency (`alpha`) and width (`linewidth`)
- Comprehensive test suite covering:
  - Both orientations
  - Shared and independent axis scaling
  - Color mapping with hue
  - Axis labels and tick values
- Example scripts demonstrating various features:
  - `demo_iris_vertical.py` - Classic Iris dataset in vertical orientation
  - `demo_iris_horizontal.py` - Classic Iris dataset in horizontal orientation
  - `demo_comparison.py` - Side-by-side comparison of orientations
  - `demo_scaling_verification.py` - Detailed scaling behavior examples
  - `demo_normalization.py` - Comparison with normalized plots
  - `demo_custom_styling.py` - Custom styling options
  - `demo_tips.py` - Using the tips dataset
  - `run_all_demos.py` - Batch execution of all demos
- Documentation:
  - Comprehensive README with motivation and feature comparison
  - Detailed API documentation in docstrings
  - Usage examples and demo scripts
- Development infrastructure:
  - Pre-commit hooks for code quality (ruff, mypy, etc.)
  - Test suite with pytest
  - Development Makefile for common tasks
  - `uv` package manager configuration

### Technical Details

- Python 3.12+ support
- Dependencies: numpy, pandas, matplotlib, seaborn
- NumPy-style docstrings
- Type hints throughout
- MIT License

---

## Release Notes Format

### Added

New features and capabilities

### Changed

Changes in existing functionality

### Deprecated

Features that will be removed in upcoming releases

### Removed

Features that have been removed

### Fixed

Bug fixes

### Security

Security-related changes

---

[Unreleased]: https://github.com/jannikmi/seaborn-paracoords/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/jannikmi/seaborn-paracoords/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/jannikmi/seaborn-paracoords/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/jannikmi/seaborn-paracoords/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/jannikmi/seaborn-paracoords/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/jannikmi/seaborn-paracoords/releases/tag/v0.0.1
