---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Reproduction Steps

Steps to reproduce the behavior:

1. Import/setup: '...'
2. Call function: '...'
3. Pass parameters: '...'
4. See error

## Minimal Reproducible Example

```python
import pandas as pd
import seaborn_parallel as snp

# Provide minimal code that reproduces the issue
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

ax = snp.parallelplot(df)  # Add your specific parameters
```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Error Message

If applicable, paste the complete error message/traceback:

```python
# Paste error here
```

## Screenshots

If applicable, add screenshots of the generated plot to help explain the problem.

## Environment

Please complete the following information:

- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.12.0]
- Package version: [e.g., 0.0.1]
- Installation method: [e.g., pip, uv, from source]

**Package versions** (run `pip list` or `uv pip list`):

```text
seaborn-paracoords==x.x.x
seaborn==x.x.x
matplotlib==x.x.x
pandas==x.x.x
numpy==x.x.x
```

## Additional Context

Add any other context about the problem here.

## Possible Solution

If you have an idea of what might be causing the issue or how to fix it, please share.

---

**Checklist:**

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal reproducible example
- [ ] I have included environment details
- [ ] I have included the complete error message (if applicable)
