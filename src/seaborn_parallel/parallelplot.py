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
    categorical_axes: Optional[List[str]] = None,
    category_orders: Optional[dict] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Draw a parallel coordinates plot with full Seaborn integration.

    This function creates parallel coordinates plots that fully integrate with
    Seaborn's theming system, automatically responding to context (paper, notebook,
    talk, poster) and style (white, whitegrid, darkgrid, etc.) settings.

    Parameters
    ----------
    data : DataFrame
        Input data structure
    vars : list of str, optional
        Variables to plot. If None, uses all columns (both numeric and categorical)
    hue : str, optional
        Variable for color encoding
    orientation : {"vertical", "horizontal"}, default "vertical"
        Plot orientation
    alpha : float, default 0.6
        Line transparency
    linewidth : float, default 1.0
        Line width
    palette : str, optional
        Color palette name (supports all Seaborn palettes)
    ax : Axes, optional
        Matplotlib axes to plot on
    sharex : bool, default False
        Share x-axis range across all numeric variables (applies to horizontal orientation).
        Does not affect categorical axes.
    sharey : bool, default False
        Share y-axis range across all numeric variables (applies to vertical orientation).
        Does not affect categorical axes.
    categorical_axes : list of str, optional
        Explicitly specify which variables should be treated as categorical.
        If None, non-numeric columns are automatically detected as categorical.
    category_orders : dict, optional
        Dictionary mapping categorical variable names to lists specifying the order
        of categories. If not provided, categories are ordered using Seaborn's
        categorical_order function.
        Example: {"size": ["small", "medium", "large"], "rating": ["low", "high"]}
    **kwargs
        Additional arguments passed to LineCollection

    Returns
    -------
    ax : Axes
        The matplotlib axes containing the plot

    Notes
    -----
    This function integrates with Seaborn's theming system:

    **Context**: Controls scaling of fonts, lines, and other elements
        - paper: Smallest (0.8×)
        - notebook: Standard (1.0×)
        - talk: Larger (1.5×)
        - poster: Largest (2.0×)

    **Style**: Controls visual appearance
        - white/dark: No grid
        - whitegrid/darkgrid: With grid
        - ticks: No grid, with tick marks

    **Palette**: All Seaborn color palettes are supported

    Examples
    --------
    Basic usage with Seaborn theming:

    >>> import seaborn as sns
    >>> import seaborn_parallel as snp
    >>>
    >>> # Set overall theme
    >>> sns.set_theme(context="talk", style="whitegrid", palette="muted")
    >>>
    >>> # Create plot - automatically uses theme settings
    >>> df = sns.load_dataset("iris")
    >>> ax = snp.parallelplot(df, hue="species")

    Using different contexts for different outputs:

    >>> # For presentations
    >>> with sns.plotting_context("poster"):
    ...     ax = snp.parallelplot(df, hue="species")
    ...     # Text and lines are automatically larger

    >>> # For papers
    >>> with sns.plotting_context("paper"):
    ...     ax = snp.parallelplot(df, hue="species")
    ...     # Text and lines are automatically smaller

    Grid control via style:

    >>> # With grid (whitegrid or darkgrid)
    >>> with sns.axes_style("whitegrid"):
    ...     ax = snp.parallelplot(df, hue="species")

    >>> # Without grid (white, dark, or ticks)
    >>> with sns.axes_style("white"):
    ...     ax = snp.parallelplot(df, hue="species")

    Categorical variables with custom ordering:

    >>> df = pd.DataFrame({
    ...     "size": ["small", "large", "medium", "small"],
    ...     "score": [85, 95, 90, 87],
    ...     "grade": ["B", "A", "A", "B"]
    ... })
    >>> ax = snp.parallelplot(
    ...     df,
    ...     categorical_axes=["size"],
    ...     category_orders={"size": ["small", "medium", "large"]},
    ...     hue="grade"
    ... )
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
        vars = list(data.columns)
    missing_vars = set(vars) - set(data.columns)
    if missing_vars:
        raise KeyError(f"Variables not found in data: {missing_vars}")
    if len(vars) < 2:
        raise ValueError("At least 2 variables required for parallel plot")

    # Detect categorical axes
    if categorical_axes is None:
        categorical_axes = [
            col for col in vars if not pd.api.types.is_numeric_dtype(data[col])
        ]
    else:
        for col in categorical_axes:
            if col not in vars:
                raise ValueError(f"categorical_axes column '{col}' is not in vars")

    # Prepare data
    df_plot = data[vars].copy()

    # Handle missing values
    if df_plot.isnull().any().any():
        warnings.warn("Missing values detected, rows will be dropped")
        df_plot = df_plot.dropna()
    if df_plot.empty:
        raise ValueError("No complete cases remaining after dropping missing values")

    # Build category mappings for categorical axes
    cat_maps = {}
    cat_orders = {}
    for col in categorical_axes:
        if category_orders and col in category_orders:
            cats = list(category_orders[col])
        else:
            cats = list(df_plot[col].astype(str).unique())
        cat_orders[col] = cats
        cat_maps[col] = {
            cat: i / (len(cats) - 1) if len(cats) > 1 else 0.5
            for i, cat in enumerate(cats)
        }
        # Map column to numeric positions for plotting
        df_plot[col] = df_plot[col].astype(str).map(cat_maps[col])

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

    # Calculate shared range if requested (only for numeric axes)
    numeric_vars = [v for v in vars if v not in categorical_axes]
    shared_range = (
        _calculate_shared_range(df_plot[numeric_vars], sharex, sharey, orientation)
        if numeric_vars
        else None
    )

    # Normalize numeric data to [0,1] for plotting
    # Categorical axes are already mapped to [0,1]
    df_normalized = df_plot.copy()
    ranges = {}
    if numeric_vars:
        normed, num_ranges = _normalize_data(df_plot[numeric_vars], shared_range)
        for v in numeric_vars:
            df_normalized[v] = normed[v]
            ranges[v] = num_ranges[v]
    for v in categorical_axes:
        ranges[v] = tuple(cat_orders[v])  # store categories for axis labeling
    if shared_range is not None:
        for v in numeric_vars:
            ranges[v] = shared_range

    # Handle colors
    colors, color_map, unique_vals = _handle_colors(
        data.loc[df_plot.index] if hue else None, hue, palette, len(df_plot)
    )

    # Create line coordinates (pass categorical_axes and cat_orders for labeling)
    lines = _create_line_coordinates(df_normalized, orientation)

    # Setup plot
    ax = ax or plt.gca()

    # Create LineCollection
    lc = LineCollection(
        lines, colors=colors, alpha=alpha, linewidth=linewidth, **kwargs
    )
    ax.add_collection(lc)

    # Configure axes (pass categorical_axes and cat_orders for labeling)
    _configure_axes(
        ax,
        vars,
        orientation,
        ranges,
        shared_range,
        unique_vals,
        color_map,
        hue,
        categorical_axes,
        cat_orders,
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
    # Only operate on numeric ranges
    if not (
        isinstance(var_range, tuple)
        and len(var_range) == 2
        and all(isinstance(x, (int, float, np.integer, np.floating)) for x in var_range)
    ):
        return np.array([]), []

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
    """
    Handle color mapping using Seaborn utilities.

    Delegates to Seaborn's categorical_order() and palette selection logic
    to match the behavior of HueMapping.categorical_mapping().
    """
    from seaborn._base import categorical_order
    from seaborn.utils import get_color_cycle

    if hue is None or data is None:
        # Use Seaborn's active color cycle instead of hardcoded "deep"
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    if hue not in data.columns:
        raise KeyError(f"hue column '{hue}' not found in data")

    hue_data = data[hue]

    # Use Seaborn's categorical_order (same as HueMapping uses)
    levels = categorical_order(hue_data, order=None)

    if len(levels) == 0:
        default_color = get_color_cycle()[0]
        return [default_color] * n_rows, None, None

    # Use Seaborn's palette selection logic (matches HueMapping.categorical_mapping)
    n_colors = len(levels)
    if palette is None:
        # Same logic as HueMapping: use color cycle if we have enough colors,
        # otherwise fall back to husl palette
        if n_colors <= len(get_color_cycle()):
            palette_colors = sns.color_palette(None, n_colors)
        else:
            palette_colors = sns.color_palette("husl", n_colors)
    else:
        palette_colors = sns.color_palette(palette, n_colors)

    color_map = dict(zip(levels, palette_colors))
    colors = [color_map.get(val, "gray") for val in hue_data]

    return colors, color_map, levels


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
    ax,
    vars,
    orientation,
    ranges,
    shared_range,
    unique_vals,
    color_map,
    hue,
    categorical_axes=None,
    cat_orders=None,
):
    """Configure axis limits, labels, ticks, and legend."""
    n_vars = len(vars)

    if categorical_axes is None:
        categorical_axes = []
    if cat_orders is None:
        cat_orders = {}

    # Get context-aware font size for tick labels
    import matplotlib as mpl

    # Use xtick.labelsize as base and scale down slightly for inline labels
    # Convert to float in case it's a string like "medium"
    base_size = mpl.rcParams["xtick.labelsize"]
    if isinstance(base_size, str):
        # If it's a string (like "medium"), get the actual size
        base_size = mpl.font_manager.FontProperties(size=base_size).get_size_in_points()
    tick_label_size = base_size * 0.8

    if orientation == "vertical":
        # Use ax.set() for cleaner property setting (Seaborn pattern)
        ax.set(
            xlim=(-0.1, n_vars - 0.9),
            ylim=(-0.05, 1.05),
            xticks=range(n_vars),
        )
        # Rotation requires separate call
        ax.set_xticklabels(vars, rotation=45, ha="right")

        # Set y-axis ticks for shared numeric range
        if shared_range is not None:
            positions, labels = _format_axis_ticks(shared_range)
            ax.set(yticks=positions, yticklabels=labels, ylabel="Value")
        else:
            ax.set(yticks=[], ylabel="")

        # Add per-axis tick labels
        for i, var in enumerate(vars):
            if var in categorical_axes:
                cats = cat_orders[var]
                n_cats = len(cats)
                if n_cats == 1:
                    positions = [0.5]
                else:
                    positions = np.linspace(0, 1, n_cats)
                for pos, label in zip(positions, cats):
                    ax.text(
                        i,
                        pos,
                        f"  {label}",
                        ha="left",
                        va="center",
                        fontsize=tick_label_size,
                        alpha=0.9,
                        fontweight="bold",
                    )
            elif shared_range is None:
                # Numeric axis, show numeric ticks as before
                var_range = ranges[var]
                # Only call _format_axis_ticks if var_range is numeric
                if (
                    isinstance(var_range, tuple)
                    and len(var_range) == 2
                    and all(
                        isinstance(x, (int, float, np.integer, np.floating))
                        for x in var_range
                    )
                ):
                    positions, labels = _format_axis_ticks(var_range)
                    for pos, label in zip(positions, labels):
                        ax.text(
                            i,
                            pos,
                            f"  {label}",
                            ha="left",
                            va="center",
                            fontsize=tick_label_size,
                            alpha=0.7,
                        )

    else:  # horizontal
        # Use ax.set() for cleaner property setting (Seaborn pattern)
        ax.set(
            ylim=(-0.1, n_vars - 0.9),
            xlim=(-0.05, 1.05),
            yticks=range(n_vars),
            yticklabels=vars,
        )

        if shared_range is not None:
            positions, labels = _format_axis_ticks(shared_range)
            ax.set(xticks=positions, xticklabels=labels, xlabel="Value")
        else:
            ax.set(xticks=[], xlabel="")

        # Add per-axis tick labels
        for i, var in enumerate(vars):
            if var in categorical_axes:
                cats = cat_orders[var]
                n_cats = len(cats)
                if n_cats == 1:
                    positions = [0.5]
                else:
                    positions = np.linspace(0, 1, n_cats)
                for pos, label in zip(positions, cats):
                    ax.text(
                        pos,
                        i,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=tick_label_size,
                        alpha=0.9,
                        fontweight="bold",
                    )
            elif shared_range is None:
                var_range = ranges[var]
                if (
                    isinstance(var_range, tuple)
                    and len(var_range) == 2
                    and all(
                        isinstance(x, (int, float, np.integer, np.floating))
                        for x in var_range
                    )
                ):
                    positions, labels = _format_axis_ticks(var_range)
                    for pos, label in zip(positions, labels):
                        ax.text(
                            pos,
                            i,
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=tick_label_size,
                            alpha=0.7,
                            rotation=0,
                        )

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
                        fontsize=tick_label_size,
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
                        fontsize=tick_label_size,
                        alpha=0.7,
                        rotation=0,
                    )

    # Add legend if hue provided
    if unique_vals is not None and color_map is not None:
        # Scale legend line width with context
        legend_linewidth = mpl.rcParams["lines.linewidth"] * 2
        handles = [
            plt.Line2D([0], [0], color=color_map[val], lw=legend_linewidth)
            for val in unique_vals
        ]
        ax.legend(handles, unique_vals, title=hue, loc="best")

    # Apply Seaborn styling
    sns.despine(ax=ax)
    # Grid is controlled by Seaborn's style system (whitegrid/darkgrid vs white/dark/ticks)
    # Don't override it here - respect the user's set_theme() choice


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
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
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
