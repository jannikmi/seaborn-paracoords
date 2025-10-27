"""
Parallel coordinates plotting with Seaborn integration.

This module provides a prototype implementation of parallel coordinates plots
compatible with Seaborn's API and styling system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, List, Any
import warnings


def parallelplot(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    hue: Optional[str] = None,
    orientation: str = "vertical",
    alpha: float = 0.6,
    linewidth: float = 1.0,
    palette: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
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
    orientation : {"vertical", "horizontal"}, default "vertical"
        Plot orientation
    alpha : float, default 0.6
        Line transparency
    linewidth : float, default 1.0
        Line width
    palette : str, optional
        Color palette name
    ax : Axes, optional
        Matplotlib axes to plot on
    sharex : bool, default False
        Share x-axis range across all variables (applies to horizontal orientation)
    sharey : bool, default False
        Share y-axis range across all variables (applies to vertical orientation)
    **kwargs
        Additional arguments passed to LineCollection

    Returns
    -------
    ax : Axes
        The matplotlib axes containing the plot
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data cannot be empty")

    if orientation not in ["vertical", "horizontal"]:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # Variable selection and validation
    if vars is None:
        vars = data.select_dtypes(include=[np.number]).columns.tolist()
        if not vars:
            raise ValueError("No numeric columns found in data")

    missing_vars = set(vars) - set(data.columns)
    if missing_vars:
        raise KeyError(f"Variables not found in data: {missing_vars}")

    if len(vars) < 2:
        raise ValueError("At least 2 variables required for parallel plot")

    # Prepare data
    df_plot = data[vars].copy()

    # Handle missing values
    if df_plot.isnull().any().any():
        warnings.warn("Missing values detected, rows will be dropped")
        df_plot = df_plot.dropna()

    if df_plot.empty:
        raise ValueError("No complete cases remaining after dropping missing values")

    # Warn if using wrong shared axis for orientation
    if orientation == "vertical" and sharex:
        warnings.warn(
            "sharex=True has no effect with orientation='vertical'. "
            "Use sharey=True to share the y-axis range instead.",
            UserWarning,
            stacklevel=2,
        )

    if orientation == "horizontal" and sharey:
        warnings.warn(
            "sharey=True has no effect with orientation='horizontal'. "
            "Use sharex=True to share the x-axis range instead.",
            UserWarning,
            stacklevel=2,
        )

    # Calculate shared range if requested
    shared_range = _calculate_shared_range(df_plot, sharex, sharey, orientation)

    # Normalize data to [0,1] for plotting
    # If shared_range is provided, all variables normalize against the global range
    # Otherwise, each variable normalizes to its own range
    df_normalized, ranges = _normalize_data(df_plot, shared_range)

    # Ranges for tick labels - use shared range if requested, otherwise individual ranges
    if shared_range is not None:
        ranges = {var: shared_range for var in vars}

    # Handle colors
    colors, color_map, unique_vals = _handle_colors(
        data.loc[df_plot.index] if hue else None, hue, palette, len(df_plot)
    )

    # Create line coordinates
    lines = _create_line_coordinates(df_normalized, orientation)

    # Setup plot
    ax = ax or plt.gca()

    # Create LineCollection
    lc = LineCollection(
        lines, colors=colors, alpha=alpha, linewidth=linewidth, **kwargs
    )
    ax.add_collection(lc)

    # Configure axes
    _configure_axes(
        ax, vars, orientation, ranges, shared_range, unique_vals, color_map, hue
    )

    return ax


def _normalize_data(
    df: pd.DataFrame, shared_range: Optional[tuple[float, float]] = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Safely normalize data to [0,1] range and return original ranges.

    Parameters
    ----------
    df : DataFrame
        Data to normalize
    shared_range : Optional[tuple[float, float]]
        If provided, normalize all columns using this (min, max) range.
        If None, normalize each column to its own range.

    Returns
    -------
    tuple of (normalized_df, ranges_dict)
        normalized_df: DataFrame with values in [0,1]
        ranges_dict: Dict mapping column names to (min, max) tuples
    """
    normalized = df.copy()
    ranges = {}

    if shared_range is not None:
        # Use shared range for all columns
        global_min, global_max = shared_range
        for col in df.columns:
            ranges[col] = shared_range

            if global_max == global_min:
                # Handle constant case
                normalized[col] = 0.5
            else:
                normalized[col] = (df[col] - global_min) / (global_max - global_min)
    else:
        # Normalize each column independently
        for col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            ranges[col] = (col_min, col_max)

            if col_max == col_min:
                # Handle constant columns
                normalized[col] = 0.5
            else:
                normalized[col] = (df[col] - col_min) / (col_max - col_min)

    return normalized, ranges


def _calculate_shared_range(
    df: pd.DataFrame, sharex: bool, sharey: bool, orientation: str
) -> Optional[tuple[float, float]]:
    """
    Calculate shared range across all variables if sharing is enabled.

    Parameters
    ----------
    df : DataFrame
        Data to analyze
    sharex : bool
        Whether to share x-axis range
    sharey : bool
        Whether to share y-axis range
    orientation : str
        Plot orientation ('vertical' or 'horizontal')

    Returns
    -------
    Optional[tuple[float, float]]
        Global (min, max) if sharing is enabled, None otherwise
    """
    # Determine if we need a shared range based on orientation
    needs_shared = (orientation == "vertical" and sharey) or (
        orientation == "horizontal" and sharex
    )

    if not needs_shared:
        return None

    # Calculate global min/max across all variables
    global_min = df.min().min()
    global_max = df.max().max()

    return (global_min, global_max)


def _format_axis_ticks(
    var_range: tuple[float, float], n_ticks: int = 6
) -> tuple[np.ndarray, list[str]]:
    """
    Generate tick positions and labels for an axis.

    Parameters
    ----------
    var_range : tuple[float, float]
        The (min, max) range for this variable
    n_ticks : int, default 6
        Number of ticks to generate

    Returns
    -------
    tuple of (positions, labels)
        positions: Array of tick positions in [0, 1] space
        labels: List of formatted tick labels with original values
    """
    vmin, vmax = var_range

    # Handle constant values
    if vmax == vmin:
        positions = np.array([0.5])
        labels = [f"{vmin:.2g}"]
        return positions, labels

    # Generate evenly spaced positions in [0, 1]
    positions = np.linspace(0, 1, n_ticks)

    # Map back to original value range
    original_values = vmin + positions * (vmax - vmin)

    # Format labels with appropriate precision
    value_range = vmax - vmin
    if value_range < 0.01:
        labels = [f"{v:.2e}" for v in original_values]
    elif value_range < 1:
        labels = [f"{v:.3f}" for v in original_values]
    elif value_range < 100:
        labels = [f"{v:.2f}" for v in original_values]
    else:
        labels = [f"{v:.1f}" for v in original_values]

    return positions, labels


def _handle_colors(data, hue, palette, n_rows):
    """Handle color mapping with proper fallbacks."""
    if hue is None or data is None:
        default_color = sns.color_palette("deep", 1)[0]
        return [default_color] * n_rows, None, None

    if hue not in data.columns:
        raise KeyError(f"hue column '{hue}' not found in data")

    hue_data = data[hue]
    unique_vals = sorted(hue_data.dropna().unique())

    if len(unique_vals) == 0:
        default_color = sns.color_palette("deep", 1)[0]
        return [default_color] * n_rows, None, None

    palette_colors = sns.color_palette(palette, len(unique_vals))
    color_map = dict(zip(unique_vals, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, unique_vals


def _create_line_coordinates(df, orientation):
    """Create line coordinate arrays for LineCollection."""
    n_vars = len(df.columns)
    xs = np.arange(n_vars)

    lines = []
    for _, row in df.iterrows():
        if orientation == "vertical":
            line = np.column_stack([xs, row.values])
        else:  # horizontal
            line = np.column_stack([row.values, xs])
        lines.append(line)

    return lines


def _configure_axes(
    ax, vars, orientation, ranges, shared_range, unique_vals, color_map, hue
):
    """Configure axis limits, labels, ticks, and legend."""
    n_vars = len(vars)

    if orientation == "vertical":
        ax.set_xlim(-0.1, n_vars - 0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(vars, rotation=45, ha="right")

        # Set y-axis ticks based on ranges
        if shared_range is not None:
            # Use shared range for all axes
            positions, labels = _format_axis_ticks(shared_range)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.set_ylabel("Value")
        else:
            # Individual ranges - show a representative range or hide y-axis
            # Since each variable has its own scale, y-axis ticks would be misleading
            ax.set_yticks([])
            ax.set_ylabel("")

    else:  # horizontal
        ax.set_ylim(-0.1, n_vars - 0.9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(vars)

        # Set x-axis ticks based on ranges
        if shared_range is not None:
            # Use shared range for all axes
            positions, labels = _format_axis_ticks(shared_range)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Value")
        else:
            # Individual ranges - hide x-axis since each variable has its own scale
            ax.set_xticks([])
            ax.set_xlabel("")

    # Add individual axis labels for each variable when not sharing ranges
    if shared_range is None:
        for i, var in enumerate(vars):
            var_range = ranges[var]
            positions, labels = _format_axis_ticks(var_range)

            if orientation == "vertical":
                # Add a secondary y-axis for each variable position
                twin_ax = ax.twiny()
                twin_ax.set_xlim(ax.get_xlim())
                twin_ax.set_xticks([i])
                twin_ax.set_xticklabels([""])
                twin_ax.tick_params(axis="x", which="both", length=0)

                # Add tick labels at the variable position
                for pos, label in zip(positions, labels):
                    ax.text(
                        i,
                        pos,
                        f"  {label}",
                        ha="left",
                        va="center",
                        fontsize=8,
                        alpha=0.7,
                    )
            else:  # horizontal
                # Add tick labels at the variable position
                for pos, label in zip(positions, labels):
                    ax.text(
                        pos,
                        i,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        alpha=0.7,
                        rotation=0,
                    )

    # Add legend if hue provided
    if unique_vals is not None and color_map is not None:
        handles = [
            plt.Line2D([0], [0], color=color_map[val], lw=3) for val in unique_vals
        ]
        ax.legend(handles, unique_vals, title=hue, loc="best")

        # Apply seaborn styling
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    """
    Example usage of the parallelplot function.

    This demonstrates creating a horizontal parallel coordinates plot
    of the iris dataset with independent axis scaling.
    """
    import os

    # Load the iris dataset
    df = sns.load_dataset("iris")

    # Create a horizontal parallel coordinates plot
    # Each variable uses its own axis range (no shared scaling)
    fig, ax = plt.subplots(figsize=(10, 8))

    parallelplot(
        data=df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        orientation="horizontal",
        alpha=0.7,
        linewidth=1.5,
        ax=ax,
    )

    ax.set_title(
        "Iris Dataset - Horizontal Parallel Coordinates\n(Independent Axis Scaling)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save to file
    os.makedirs("./tmp", exist_ok=True)
    output_path = "./tmp/parallelplot_example.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Example plot saved to: {output_path}")

    # Optionally show the plot (comment out for automated runs)
    # plt.show()
