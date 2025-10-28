"""
Parallel coordinates plotting with Seaborn Objects integration.

This module provides parallel coordinates plots using Seaborn Objects
with post-processing for independent axis support.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn.utils import locator_to_legend_entries
from matplotlib.ticker import MaxNLocator
from typing import Optional, List, Dict, Tuple, Literal, Any
import warnings


def _hide_all_spines(ax: plt.Axes) -> None:
    """Hide all axis spines (frame borders)."""
    for spine in ax.spines.values():
        spine.set_visible(False)


def _generate_ticks(
    var: str,
    original_ranges: Dict[str, Tuple[float, float]],
    categorical_info: Dict[str, dict],
    data: pd.DataFrame,
) -> Tuple[List, List, np.ndarray]:
    """
    Generate tick values, labels, and normalized positions for a variable.

    Parameters
    ----------
    var : str
        Variable name
    original_ranges : dict
        Original min/max for each variable
    categorical_info : dict
        Categorical variable information
    data : DataFrame
        Original data (for dtypes)

    Returns
    -------
    tick_vals : list
        Tick values in original scale
    tick_labels : list
        Tick labels as strings
    norm_ticks : ndarray
        Normalized tick positions [0, 1]
    """
    if var in categorical_info:
        # Categorical variable
        categories = categorical_info[var]["categories"]
        tick_vals = categories
        tick_labels = [str(cat) for cat in categories]
        norm_ticks = np.linspace(0, 1, len(categories))
    else:
        # Numeric variable
        min_val, max_val = original_ranges[var]

        # Generate ticks using Seaborn's utility
        locator = MaxNLocator(nbins=6)
        tick_vals, tick_labels = locator_to_legend_entries(
            locator, (min_val, max_val), data[var].dtype
        )

        # Normalize tick values to [0, 1] for positioning
        tick_vals_array = np.array(tick_vals)
        if max_val - min_val > 0:
            norm_ticks = (tick_vals_array - min_val) / (max_val - min_val)
        else:
            norm_ticks = np.array([0.5])

    return tick_vals, tick_labels, norm_ticks


def _draw_axis_with_ticks(
    ax: plt.Axes,
    var: str,
    position: float,
    original_ranges: Dict[str, Tuple[float, float]],
    categorical_info: Dict[str, dict],
    data: pd.DataFrame,
    orient: Literal["v", "h"],
) -> None:
    """
    Draw a single axis with ticks and labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    var : str
        Variable name
    position : float
        Position along the cross-axis (x for vertical, y for horizontal)
    original_ranges : dict
        Original min/max for each variable
    categorical_info : dict
        Categorical variable information
    data : DataFrame
        Original data (for dtypes)
    orient : {'v', 'h'}
        Orientation
    """
    # Generate ticks
    tick_vals, tick_labels, norm_ticks = _generate_ticks(
        var, original_ranges, categorical_info, data
    )

    # Determine if categorical for text rotation
    is_categorical = var in categorical_info

    if orient == "v":
        # Vertical: axis line from (position, 0) to (position, 1)
        ax.plot(
            [position, position],
            [0, 1],
            color="black",
            linewidth=1.5,
            clip_on=False,
            zorder=100,
        )

        # Add ticks and labels
        for label, pos in zip(tick_labels, norm_ticks):
            # Tick mark
            ax.plot(
                [position - 0.02, position],
                [pos, pos],
                color="black",
                linewidth=1,
                clip_on=False,
                zorder=100,
            )
            # Tick label
            ax.text(
                position - 0.04,
                pos,
                label,
                ha="right",
                va="center",
                fontsize=9,
                clip_on=False,
            )
    else:  # horizontal
        # Horizontal: axis line from (0, position) to (1, position)
        ax.plot(
            [0, 1],
            [position, position],
            color="black",
            linewidth=1.5,
            clip_on=False,
            zorder=100,
        )

        # Add ticks and labels
        for label, pos in zip(tick_labels, norm_ticks):
            # Tick mark
            ax.plot(
                [pos, pos],
                [position - 0.02, position],
                color="black",
                linewidth=1,
                clip_on=False,
                zorder=100,
            )
            # Tick label (rotate categorical labels)
            ax.text(
                pos,
                position - 0.04,
                label,
                ha="center",
                va="top",
                fontsize=9,
                clip_on=False,
                rotation=45 if is_categorical else 0,
            )


def _normalize_data(
    data: pd.DataFrame,
    vars: List[str],
    hue: Optional[str],
    orient: str,
    sharex: bool,
    sharey: bool,
    category_orders: Optional[Dict[str, List]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], Dict[str, dict]]:
    """
    Normalize data to [0, 1] range for plotting.

    Parameters
    ----------
    data : DataFrame
        Input data
    vars : list of str
        Variables to normalize
    hue : str or None
        Hue variable (not normalized)
    orient : str
        Orientation
    sharex : bool
        Share x-axis range (for horizontal)
    sharey : bool
        Share y-axis range (for vertical)
    category_orders : dict, optional
        Custom category orders

    Returns
    -------
    normalized_df : DataFrame
        Normalized data
    original_ranges : dict
        Original min/max for each variable
    categorical_info : dict
        Information about categorical variables
    """
    normalized_df = data.copy()
    original_ranges = {}
    categorical_info = {}

    # Determine if we need shared scaling
    use_shared = (orient in ["v", "y"] and sharey) or (orient in ["h", "x"] and sharex)

    # Separate numeric and categorical columns
    numeric_vars = []
    categorical_vars = []

    for var in vars:
        # Check dtype: boolean, datetime, and object types are categorical
        dtype = data[var].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            # Boolean and datetime are treated as categorical
            categorical_vars.append(var)
            categorical_info[var] = {
                "type": "categorical",
                "categories": data[var].unique().tolist(),
            }
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_vars.append(var)
        else:
            # String/object types are categorical
            categorical_vars.append(var)
            categorical_info[var] = {
                "type": "categorical",
                "categories": data[var].unique().tolist(),
            }

    # Normalize numeric variables
    if use_shared and numeric_vars:
        # Shared scaling: find global min/max
        all_values = data[numeric_vars].values.flatten()
        global_min = np.nanmin(all_values)
        global_max = np.nanmax(all_values)

        for var in numeric_vars:
            original_ranges[var] = (global_min, global_max)
            if global_max - global_min > 0:
                normalized_df[var] = (data[var] - global_min) / (
                    global_max - global_min
                )
            else:
                warnings.warn(f"All numeric columns have constant value {global_min}")
                normalized_df[var] = 0.5
    else:
        # Independent scaling: normalize each variable independently
        for var in numeric_vars:
            min_val = data[var].min()
            max_val = data[var].max()
            original_ranges[var] = (min_val, max_val)

            if max_val - min_val > 0:
                normalized_df[var] = (data[var] - min_val) / (max_val - min_val)
            else:
                warnings.warn(
                    f"Variable '{var}' has constant value {min_val}, normalizing to 0.5"
                )
                normalized_df[var] = 0.5

    # Handle categorical variables (encode as codes, then normalize)
    for var in categorical_vars:
        # Use custom order if provided, otherwise use data order
        if category_orders and var in category_orders:
            categories = category_orders[var]
        else:
            categories = categorical_info[var]["categories"]

        categorical_info[var]["categories"] = categories

        # Create categorical with explicit order
        cat_data = pd.Categorical(data[var], categories=categories)
        codes = cat_data.codes

        # Normalize codes to [0, 1]
        if len(categories) > 1:
            normalized_df[var] = codes / (len(categories) - 1)
        else:
            normalized_df[var] = 0.5

        original_ranges[var] = (0, len(categories) - 1)

    return normalized_df, original_ranges, categorical_info


def _create_seaborn_plot(
    data: pd.DataFrame,
    vars: List[str],
    hue: Optional[str],
    orient: str,
    alpha: float,
    linewidth: float,
    palette: Optional[str],
    ax: Optional[plt.Axes],
    **kwargs,
) -> so.Plot:
    """
    Create Seaborn Objects plot.

    Parameters
    ----------
    data : DataFrame
        Normalized data (already in [0, 1] range)
    vars : list of str
        Variables to plot
    hue : str or None
        Hue variable
    orient : str
        Orientation
    alpha : float
        Line transparency
    linewidth : float
        Line width
    palette : str or None
        Color palette
    ax : Axes or None
        Axes to plot on
    **kwargs
        Additional arguments for so.Lines()

    Returns
    -------
    plot : so.Plot
        Seaborn Objects Plot instance
    """
    # Add index column for grouping
    plot_data = data.copy()
    plot_data["_index"] = range(len(plot_data))

    # Select only the variables we want to plot
    id_vars = ["_index"]
    if hue is not None:
        id_vars.append(hue)

    # Melt data to tidy format
    melted = plot_data.melt(
        id_vars=id_vars, value_vars=vars, var_name="variable", value_name="value"
    )

    # Create plot
    if orient in ["v", "y"]:
        x, y = "variable", "value"
    else:  # horizontal
        x, y = "value", "variable"

    plot = so.Plot(melted, x=x, y=y, color=hue)

    # Add lines with grouping by index
    plot = plot.add(
        so.Lines(alpha=alpha, linewidth=linewidth, **kwargs), group="_index"
    )

    # Set palette if specified
    if palette is not None:
        plot = plot.scale(color=palette)  # type: ignore[arg-type]

    # Render to specific axes if provided
    if ax is not None:
        plot = plot.on(ax)

    return plot


def _add_independent_tick_labels(
    ax: plt.Axes,
    vars: List[str],
    original_ranges: Dict[str, Tuple[float, float]],
    categorical_info: Dict[str, dict],
    data: pd.DataFrame,
    orient: Literal["v", "h"],
) -> None:
    """
    Add independent tick labels for both orientations.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    vars : list of str
        Variables in order
    original_ranges : dict
        Original min/max for each variable
    categorical_info : dict
        Categorical variable information
    data : DataFrame
        Original data (for dtypes)
    orient : {'v', 'h'}
        Orientation
    """
    # Clear default axes and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Hide tick marks but keep labels for variable names
    if orient == "v":
        ax.tick_params(axis="x", length=0)
    else:  # horizontal
        ax.tick_params(axis="y", length=0)

    _hide_all_spines(ax)

    # Add independent axis for each variable
    for i, var in enumerate(vars):
        _draw_axis_with_ticks(
            ax, var, i, original_ranges, categorical_info, data, orient=orient
        )


def parallelplot(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    hue: Optional[str] = None,
    orient: Literal["v", "h", "x", "y"] = "v",
    alpha: float = 0.6,
    linewidth: float = 1.0,
    palette: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
    categorical_axes: Optional[List[str]] = None,
    category_orders: Optional[Dict[str, List]] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Draw parallel coordinates plot with Seaborn Objects integration.

    Parallel coordinates plots visualize multivariate data by representing each
    observation as a line connecting its values across multiple variables. Each
    variable is displayed on a separate vertical or horizontal axis.

    This implementation uses Seaborn Objects for rendering with post-processing
    to provide independent axes for each variable, preserving original data ranges
    (not normalized to [0,1] in the display).

    Parameters
    ----------
    data : DataFrame
        Input data in tidy (long) or wide format. Each row represents one observation.
    vars : list of str, optional
        Variables (column names) to include in the plot. If None (default), all
        numeric columns except `hue` are used. Columns can be numeric, categorical,
        boolean, or datetime.
    hue : str, optional
        Column name for color encoding. Each unique value gets a different color.
        A legend is automatically added when hue is specified.
    orient : {'v', 'h', 'x', 'y'}, default 'v'
        Orientation of the plot:
        - 'v' or 'y': Vertical (variables on x-axis, values on y-axis)
        - 'h' or 'x': Horizontal (variables on y-axis, values on x-axis)
    alpha : float, default 0.6
        Line transparency (0=transparent, 1=opaque). Lower values help visualize
        overlapping lines in dense datasets.
    linewidth : float, default 1.0
        Width of the lines in points. Decrease for datasets with many observations.
    palette : str, optional
        Seaborn color palette name (e.g., 'deep', 'muted', 'pastel', 'Set2').
        Only used when `hue` is specified. If None, uses the current color cycle.
    ax : Axes, optional
        Matplotlib axes object to draw the plot onto. If None, uses current axes.
    sharex : bool, default False
        If True (horizontal orientation), all variables share the same x-axis range.
        This normalizes all variables to a common scale. Has no effect with vertical
        orientation (use `sharey` instead).
    sharey : bool, default False
        If True (vertical orientation), all variables share the same y-axis range.
        This normalizes all variables to a common scale. Has no effect with horizontal
        orientation (use `sharex` instead).
    categorical_axes : list of str, optional
        Deprecated. Categorical variables are now automatically detected based on
        dtype (object, category, bool, datetime). Keep for backward compatibility
        but ignored in current implementation.
    category_orders : dict, optional
        Custom ordering for categorical variables. Keys are variable names, values
        are lists specifying the desired category order.
        Example: {'species': ['setosa', 'versicolor', 'virginica']}
    **kwargs
        Additional keyword arguments passed to seaborn.objects.Lines().
        Common options: 'color', 'linestyle', 'marker', etc.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the plot.

    See Also
    --------
    seaborn.objects.Plot : Seaborn's declarative plotting interface
    pandas.plotting.parallel_coordinates : Pandas parallel coordinates (different API)

    Notes
    -----
    **Key Features:**
    - Preserves original data ranges (not normalized to [0,1] in display)
    - Supports both numeric and categorical variables
    - Automatic dtype detection (bool, datetime treated as categorical)
    - Independent axes per variable (when sharex/sharey=False)
    - Full Seaborn theming integration

    **Limitations:**
    - Faceting not currently supported (may be added in future)
    - Custom tick labels are static (don't update on interactive zoom/pan)
    - Large datasets (>1000 lines) may render slowly

    **Performance Tips:**
    - For dense data, decrease `alpha` (e.g., 0.3) and `linewidth` (e.g., 0.5)
    - Consider filtering to representative sample for exploratory analysis
    - Use `sharex=True` or `sharey=True` for simpler plots with fewer custom ticks

    Examples
    --------
    Basic usage with iris dataset:

    >>> import seaborn as sns
    >>> import seaborn_parallel as snp
    >>> iris = sns.load_dataset('iris')
    >>> ax = snp.parallelplot(iris, hue='species')

    Horizontal orientation with custom styling:

    >>> ax = snp.parallelplot(
    ...     iris,
    ...     hue='species',
    ...     orient='h',
    ...     alpha=0.4,
    ...     linewidth=1.5,
    ...     palette='Set2'
    ... )

    Select specific variables and use shared scaling:

    >>> ax = snp.parallelplot(
    ...     iris,
    ...     vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    ...     hue='species',
    ...     sharey=True,
    ...     palette='muted'
    ... )

    Mixed numeric and categorical data (tips dataset):

    >>> tips = sns.load_dataset('tips')
    >>> ax = snp.parallelplot(
    ...     tips,
    ...     vars=['total_bill', 'day', 'time', 'size'],
    ...     hue='sex',
    ...     alpha=0.5
    ... )

    Custom category ordering:

    >>> ax = snp.parallelplot(
    ...     tips,
    ...     vars=['day', 'total_bill', 'tip'],
    ...     category_orders={'day': ['Thur', 'Fri', 'Sat', 'Sun']},
    ...     hue='time'
    ... )
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Input data must be a pandas DataFrame, got {type(data).__name__}. "
            "Convert your data to DataFrame first: pd.DataFrame(data)"
        )

    if data.empty:
        raise ValueError(
            "Input DataFrame is empty (no rows). "
            "Cannot create parallel coordinates plot from empty data."
        )

    if orient not in ["v", "h", "x", "y"]:
        raise ValueError(
            f"Invalid orient='{orient}'. Must be one of: "
            "'v' (vertical), 'h' (horizontal), 'x' (horizontal), or 'y' (vertical)"
        )

    # Validate hue parameter
    if hue is not None and hue not in data.columns:
        raise KeyError(
            f"Hue variable '{hue}' not found in DataFrame columns. "
            f"Available columns: {list(data.columns)}"
        )

    # Determine variables to plot
    if vars is None:
        vars = [col for col in data.columns if col != hue]
    else:
        # Validate that all specified vars exist
        missing_vars = [v for v in vars if v not in data.columns]
        if missing_vars:
            raise KeyError(
                f"Variables not found in DataFrame: {missing_vars}. "
                f"Available columns: {list(data.columns)}"
            )

    if not vars:
        raise ValueError(
            "No variables to plot. Either the DataFrame has no columns, "
            "or all columns were excluded. "
            f"DataFrame columns: {list(data.columns)}, hue={hue}"
        )

    if len(vars) < 2:
        raise ValueError(
            f"At least 2 variables required for parallel coordinates plot, got {len(vars)}. "
            f"Variables: {vars}. "
            "Parallel coordinates visualize relationships between multiple variables."
        )

    # Check for missing values
    if data[vars].isnull().any().any():
        missing_counts = data[vars].isnull().sum()
        missing_value_dict = missing_counts[missing_counts > 0].to_dict()
        warnings.warn(
            f"Data contains missing values (NaN) in {len(missing_value_dict)} variable(s): "
            f"{missing_value_dict}. Lines with NaN may not be displayed.",
            UserWarning,
            stacklevel=2,
        )

    # Warn about incorrect axis sharing
    if orient in ["v", "y"] and sharex:
        warnings.warn(
            "sharex=True has no effect with vertical orientation (orient='v'). "
            "Use sharey=True to share the y-axis range instead.",
            UserWarning,
            stacklevel=2,
        )
    if orient in ["h", "x"] and sharey:
        warnings.warn(
            "sharey=True has no effect with horizontal orientation (orient='h'). "
            "Use sharex=True to share the x-axis range instead.",
            UserWarning,
            stacklevel=2,
        )

    # Store original data for dtype info
    original_data = data.copy()

    # Normalize data
    normalized_df, original_ranges, categorical_info = _normalize_data(
        data, vars, hue, orient, sharex, sharey, category_orders
    )

    # Create Seaborn Objects plot
    plot = _create_seaborn_plot(
        normalized_df, vars, hue, orient, alpha, linewidth, palette, ax, **kwargs
    )

    # Render plot
    plot_result = plot.plot()

    # Extract axes - use the provided ax if available, otherwise get from result
    if ax is not None:
        # User provided axes, use it directly
        result_ax = ax
    elif hasattr(plot_result, "_figure"):
        # Get the axes from the rendered plot
        fig = plot_result._figure
        result_ax = fig.axes[0] if fig.axes else plt.gca()
    else:
        result_ax = plt.gca()

    # Seaborn Objects automatically adds legend when hue is specified
    # No need to manually add legend

    # Post-process for independent axes
    use_independent = (orient in ["v", "y"] and not sharey) or (
        orient in ["h", "x"] and not sharex
    )

    if use_independent:
        # Normalize orient to 'v' or 'h'
        normalized_orient: Literal["v", "h"] = "v" if orient in ["v", "y"] else "h"
        _add_independent_tick_labels(
            result_ax,
            vars,
            original_ranges,
            categorical_info,
            original_data,
            normalized_orient,
        )

        # Fix inverted y-axis in horizontal orientation
        if normalized_orient == "h":
            ylim = result_ax.get_ylim()
            if ylim[0] > ylim[1]:
                result_ax.set_ylim(ylim[1], ylim[0])

    return result_ax
