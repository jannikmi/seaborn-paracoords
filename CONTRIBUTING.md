# Contributing to Seaborn Parallel Coordinates Plot

Thank you for your interest in contributing to this project! We welcome contributions of all kinds, from bug reports to feature suggestions to code improvements.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

This project is in its **early development stage** and welcomes contributions from developers of all experience levels. Whether you're fixing a typo, adding a feature, or improving documentation, your help is appreciated!

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue describing the problem
- 💡 **Suggest features** - Share your ideas for improvements
- 📝 **Improve documentation** - Help make the project more accessible
- 🔧 **Submit code** - Fix bugs or implement new features
- 🧪 **Add tests** - Improve test coverage
- 🤝 **Become a maintainer** - Help guide the project's direction

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Initial Setup

1. **Fork and clone the repository**

   ```bash
   gh repo fork jannikmi/seaborn-paracoords --clone
   cd seaborn-paracoords
   ```

2. **Install dependencies using uv**

   ```bash
   # Install all dependencies including dev tools
   uv sync --all-groups
   ```

3. **Install pre-commit hooks**

   ```bash
   make hook
   # or manually: pre-commit install
   ```

4. **Verify installation**

   ```bash
   # Run tests to ensure everything works
   make test-parallel
   # or: uv run pytest tests/
   ```

## Development Workflow

### Setting Up Your Environment

Always use `uv run` for Python commands to ensure you're using the correct environment:

```bash
# Run Python scripts
uv run python scripts/demo_iris_vertical.py

# Run tests
uv run pytest tests/

# Run Python commands
uv run python -c "import seaborn_parallel; print('OK')"
```

### Making Changes

1. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or: git checkout -b bugfix/issue-description
   ```

2. **Make your changes**
   - Edit code in `src/seaborn_parallel/`
   - Add tests in `tests/`
   - Update documentation as needed

3. **Test your changes**

   ```bash
   # Run all tests
   make test-parallel

   # Run specific test
   uv run pytest tests/test_parallelplot.py::test_name -v

   # Run with coverage
   uv run pytest tests/ --cov=src --cov-report=html
   ```

4. **Run demos to verify**

   ```bash
   # Test your changes visually
   uv run python scripts/demo_iris_vertical.py
   uv run python scripts/run_all_demos.py
   ```

5. **Check code quality**

   Pre-commit hooks will run automatically, but you can also run manually:

   ```bash
   # Run all linting checks
   make lint

   # Or run specific tools
   uv run ruff check src/ tests/
   uv run ruff format src/ tests/
   uv run mypy src/
   ```

## Coding Standards

### Code Style

This project uses automated code formatting and linting:

- **Ruff**: For linting and formatting (replaces black, flake8, isort)
- **mypy**: For type checking
- **Pre-commit hooks**: Automatically enforce standards

### Style Guidelines

- **Python version**: Target Python 3.12+
- **Type hints**: Add type hints to all new functions
- **Docstrings**: Use NumPy-style docstrings for all public functions
- **Line length**: 88 characters (Black default)
- **Imports**: Organized by ruff/isort (automatically)

### Example Code Structure

```python
def parallelplot(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    hue: Optional[str] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Draw a parallel coordinates plot.

    Parameters
    ----------
    data : DataFrame
        Input data structure
    vars : list of str, optional
        Variables to plot. If None, uses all numeric columns
    hue : str, optional
        Variable for color encoding

    Returns
    -------
    Axes
        Matplotlib axes containing the plot

    Examples
    --------
    >>> import pandas as pd
    >>> import seaborn_parallel as snp
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> ax = snp.parallelplot(df)
    """
    # Implementation here
    pass
```

### Commit Messages

Write clear, descriptive commit messages:

```
Add horizontal orientation support for parallel plots

- Implement orientation parameter in parallelplot()
- Add tests for vertical and horizontal modes
- Update documentation with examples
```

## Testing

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Aim for high test coverage

### Test Coverage

Tests should cover:

- Both orientations (vertical/horizontal)
- With and without `hue` parameter
- Shared vs. independent scaling
- Edge cases and error handling
- Correct axis labels and tick values

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_parallelplot.py

# Run specific test
uv run pytest tests/test_parallelplot.py::test_sharex_parameter -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Writing Tests

Example test structure:

```python
import pytest
import pandas as pd
import seaborn_parallel as snp

def test_feature_name():
    """Test description."""
    # Setup
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Execute
    ax = snp.parallelplot(df, orient='v')

    # Assert
    assert ax is not None
    assert len(ax.lines) > 0
```

## Submitting Changes

### Pull Request Process

1. **Update documentation**
   - Update README.md if adding features
   - Update docstrings
   - Add examples if appropriate

2. **Ensure tests pass**

   ```bash
   make test-parallel
   make lint
   ```

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a pull request**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Link any related issues

### PR Requirements

- [ ] Tests pass
- [ ] Code follows style guidelines (pre-commit hooks pass)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains changes

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

- **Description**: Clear description of the problem
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, package versions
- **Screenshots**: If applicable, especially for visual issues

### Feature Requests

When suggesting a feature:

- **Use case**: Explain why this feature would be useful
- **Proposed solution**: Describe how you envision it working
- **Alternatives**: Any alternative approaches you've considered
- **Examples**: Show examples from other libraries if relevant

## Questions?

- Open a [GitHub Discussion](https://github.com/jannikmi/seaborn-paracoords/discussions) for general questions
- Open an [Issue](https://github.com/jannikmi/seaborn-paracoords/issues) for bugs or feature requests
- Check existing issues and discussions first

---

Thank you for contributing! 🎉
