# Agents Guide for Seaborn Parallel Coordinates Plot

## Mission

Guide AI agents working on this parallel coordinates visualization library. This project fills a gap in the Python ecosystem by providing true horizontal/vertical orientation support and preserving original data ranges (not normalized to [0,1]).

## Repository Structure

- `src/seaborn_parallel/parallelplot.py`: Main implementation
- `scripts/`: Demo scripts (`demo_*.py`, `run_all_demos.py`)
- `tests/test_parallelplot.py`: Test suite
- `tmp/`: Generated plot outputs

## Development Workflow

### Critical Rules

- **Always use `uv run`** for Python commands
- **Install pre-commit hooks** with `make hook` before making changes
- **Run tests before committing**: `uv run pytest tests/` or `make test-parallel`
- See `Makefile` for additional commands (`make install`, `make update`, etc.)

## Agent-Specific Guidance

### Common Tasks

1. **Adding features**: Edit `src/seaborn_parallel/parallelplot.py`
2. **Creating demos**: Add scripts to `scripts/demo_*.py`
3. **Writing tests**: Update `tests/test_parallelplot.py`
4. **Documentation**: Update docstrings and README.md

## Pitfalls & Implementation Details

### Key Concepts to Remember

**Orientation vs. Shared Scaling:**

- Vertical + shared scale → `sharey=True`
- Horizontal + shared scale → `sharex=True`
- Default: preserves original ranges (no shared axes)

**Testing Must Cover:**

- Both orientations (vertical/horizontal)
- With and without `hue` parameter
- Shared vs. independent scaling
- Correct axis labels and tick values

**Code Style:**

- Use ruff/isort formatting (configured in pre-commit)
- Add type hints to new functions
- Update docstrings for API changes

**Performance Considerations:**

- Large datasets: adjust `alpha` and `linewidth` for readability
- Matplotlib rendering is slow with thousands of lines
- Future: consider `rasterized=True` for very large plots (not yet implemented)
