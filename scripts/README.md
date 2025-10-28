# Demo Scripts

This directory contains demonstration scripts for the seaborn-paracoord library.

## Quick Start

Run all demos:
```bash
uv run python scripts/run_all_demos.py
```

Run a specific demo:
```bash
uv run python scripts/run_all_demos.py orientations
```

## Available Demos

| Demo | Description | Output |
|------|-------------|--------|
| `orientations` | Vertical and horizontal orientations | `demo_orientations.png` |
| `datasets` | Multiple datasets (tips, flights, auto-selection) | `demo_datasets.png` |
| `scaling` | Independent vs shared axis scaling | `demo_scaling.png` |
| `categorical_axes` | Categorical variable support | 5 separate PNGs |
| `seaborn_contexts` | Seaborn plotting contexts (paper/notebook/talk/poster) | `demo_seaborn_contexts.png` |

## Output Location

All demo outputs are saved to `./tmp/seaborn-paracoords_demo_*.png`

## See Also

For detailed information about each demo, see [DEMOS_SUMMARY.md](../DEMOS_SUMMARY.md) in the root directory.
