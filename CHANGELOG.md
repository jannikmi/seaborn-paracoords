# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Enhanced customization options for axis styling
- Integration with Seaborn's FacetGrid for multi-plot layouts

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

[Unreleased]: https://github.com/jannikmi/seaborn-paracoord/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/jannikmi/seaborn-paracoord/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/jannikmi/seaborn-paracoord/releases/tag/v0.0.1
