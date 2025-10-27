# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Add support for categorical variables on select axes
- Enhanced customization options for axis styling
- Integration with Seaborn's FacetGrid for multi-plot layouts

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

[Unreleased]: https://github.com/jannikmi/seaborn-paracoord/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/jannikmi/seaborn-paracoord/releases/tag/v0.0.1
