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

        # Adjust number of ticks based on font size to prevent overlap
        # Base nbins on tick label size - larger fonts need fewer ticks
        tick_fontsize = plt.rcParams.get("ytick.labelsize", 10)
        if isinstance(tick_fontsize, str):
            tick_fontsize = plt.rcParams.get("font.size", 10)

        # Scale nbins inversely with font size
        # paper (8.8pt) -> 6 bins, talk (16.5pt) -> 4 bins, poster (22pt) -> 3 bins
        if tick_fontsize >= 20:
            nbins = 3
        elif tick_fontsize >= 14:
            nbins = 4
        else:
            nbins = 6

        # Generate ticks using Seaborn's utility
        locator = MaxNLocator(nbins=nbins)
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

    # Get line width and font size from current matplotlib rcParams
    # This ensures we respect seaborn plotting contexts
    axis_linewidth = plt.rcParams["axes.linewidth"]
    tick_linewidth = plt.rcParams["xtick.major.width"]
    tick_labelsize = plt.rcParams["xtick.labelsize"]

    if orient == "v":
        # Vertical: axis line from (position, 0) to (position, 1)
        ax.plot(
            [position, position],
            [0, 1],
            color="black",
            linewidth=axis_linewidth,
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
                linewidth=tick_linewidth,
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
                fontsize=tick_labelsize,
                clip_on=False,
            )
    else:  # horizontal
        # Horizontal: axis line from (0, position) to (1, position)
        ax.plot(
            [0, 1],
            [position, position],
            color="black",
            linewidth=axis_linewidth,
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
                linewidth=tick_linewidth,
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
                fontsize=tick_labelsize,
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
    original_hue_data: Optional[pd.Series] = None,
    hue_categories: Optional[List] = None,
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
    original_hue_data : Series or None
        Original (non-normalized) hue data for categorical variables
    hue_categories : list or None
        Ordered categories for categorical hue variable. If provided,
        ensures colors are assigned in this order.
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

    # Prepare hue data with optional categorical ordering
    hue_data: Any = None
    if hue is not None and original_hue_data is not None:
        if hue_categories is not None:
            hue_data = pd.Categorical(
                original_hue_data, categories=hue_categories, ordered=False
            )
        else:
            hue_data = original_hue_data

    # Assign hue data to plot_data
    if hue is not None:
        if hue in vars:
            # Hue is plotted as a variable - duplicate it for grouping
            plot_data["_hue_for_color"] = (
                hue_data if hue_data is not None else data[hue]
            )
            hue_col_for_plot = "_hue_for_color"
        else:
            # Hue is not plotted - use it directly for coloring
            plot_data[hue] = hue_data if hue_data is not None else data[hue]
            hue_col_for_plot = hue
        hue_legend_title = hue
    else:
        hue_col_for_plot = None
        hue_legend_title = None

    # Select only the variables we want to plot
    id_vars = ["_index"]
    if hue_col_for_plot is not None:
        id_vars.append(hue_col_for_plot)

    # Melt data to tidy format
    melted = plot_data.melt(
        id_vars=id_vars, value_vars=vars, var_name="variable", value_name="value"
    )

    # Ensure hue column maintains categorical order for consistent coloring
    if hue_categories is not None and hue_col_for_plot is not None:
        if hue_col_for_plot in melted.columns:
            melted[hue_col_for_plot] = pd.Categorical(
                melted[hue_col_for_plot], categories=hue_categories, ordered=False
            )

    # Create plot
    if orient in ["v", "y"]:
        x, y = "variable", "value"
    else:  # horizontal
        x, y = "value", "variable"

    plot = so.Plot(melted, x=x, y=y, color=hue_col_for_plot)

    # Add lines with grouping by index
    plot = plot.add(
        so.Lines(alpha=alpha, linewidth=linewidth, **kwargs), group="_index"
    )

    # Configure palette and legend
    # Apply custom palette if provided
    # Note: When hue_categories is specified, Seaborn Objects automatically respects
    # the Categorical dtype ordering (set in _prepare_hue_data) for color assignment.
    if palette is not None:
        plot = plot.scale(color=palette)  # type: ignore[arg-type]

    # Fix legend title if we used a duplicate column name for hue
    if hue_col_for_plot == "_hue_for_color" and hue_legend_title is not None:
        # Label the color scale with the original hue variable name
        plot = plot.label(color=hue_legend_title)

    # Render to specific axes if provided
    if ax is not None:
        plot = plot.on(ax)

    return plot


def _clear_axis_labels(ax: plt.Axes) -> None:
    """Clear both x and y axis labels."""
    ax.set_xlabel("")
    ax.set_ylabel("")


def _fix_inverted_yaxis_if_needed(ax: plt.Axes) -> None:
    """Fix inverted y-axis in horizontal orientation plots."""
    ylim = ax.get_ylim()
    if ylim[0] > ylim[1]:
        ax.set_ylim(ylim[1], ylim[0])


def _map_normalized_ticks_to_range(
    ticks: np.ndarray,
    global_min: float,
    global_max: float,
) -> List[str]:
    """
    Map normalized [0, 1] tick positions to original data range labels.

    Parameters
    ----------
    ticks : ndarray
        Tick positions in normalized [0, 1] space
    global_min : float
        Minimum value of original data range
    global_max : float
        Maximum value of original data range

    Returns
    -------
    labels : list of str
        Tick labels showing original data values
    """
    return [f"{global_min + (global_max - global_min) * tick:.2g}" for tick in ticks]


def _extract_legend_alpha(colors: Any) -> float:
    """
    Extract alpha value from the first color in an array.

    Parameters
    ----------
    colors : Any
        Array of colors (RGB or RGBA tuples)

    Returns
    -------
    alpha : float
        Alpha value (default 1.0 if not present)
    """
    first_color = colors[0] if len(colors) > 0 else None
    default_alpha = 1.0
    if first_color is not None and hasattr(first_color, "__len__"):
        try:
            if len(first_color) > 3:  # type: ignore[arg-type]
                default_alpha = float(first_color[3])  # type: ignore[index]
        except (TypeError, IndexError):
            pass
    return default_alpha


def _get_unique_colors(colors: Any) -> List:
    """
    Get unique colors while preserving order.

    Parameters
    ----------
    colors : Any
        Array of colors

    Returns
    -------
    unique_colors : list
        List of unique colors in order of first appearance
    """
    seen_colors = {}
    unique_colors = []
    for color in colors:
        color_tuple = tuple(color)  # type: ignore[arg-type]
        if color_tuple not in seen_colors:
            seen_colors[color_tuple] = True
            unique_colors.append(color)
    return unique_colors


def _create_legend_handles(
    colors: Any, hue_values: Any, reverse_colors: bool = False
) -> Tuple[List, List[str]]:
    """
    Create legend handles and labels from colors and hue values.

    Parameters
    ----------
    colors : Any
        Array of colors from LineCollection
    hue_values : Any
        Unique hue values (can be ndarray or list)
    reverse_colors : bool
        If True, reverse the color order to match reversed hue_values

    Returns
    -------
    handles : list
        List of Line2D objects for legend
    labels : list of str
        List of labels for legend
    """
    from matplotlib.lines import Line2D

    unique_colors = _get_unique_colors(colors)
    default_alpha = _extract_legend_alpha(colors)

    # Get the colors for this hue (may be reversed)
    colors_to_use = unique_colors[: len(hue_values)]
    if reverse_colors:
        colors_to_use = list(reversed(colors_to_use))

    handles = [
        Line2D(
            [0],
            [0],
            color=color,  # type: ignore[arg-type]
            linewidth=2,
            alpha=default_alpha,
        )
        for color in colors_to_use
    ]
    labels = [str(val) for val in hue_values]

    return handles, labels


def _clear_figure_legends(ax: plt.Axes) -> None:
    """
    Clear accumulated legends from the figure.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    """
    fig = ax.figure
    if fig and hasattr(fig, "legends"):
        fig.legends.clear()


def _add_legend_from_line_collection(
    ax: plt.Axes,
    hue: str,
    original_data: pd.DataFrame,
    hue_categories: Optional[List] = None,
) -> None:
    """
    Create and add legend from LineCollection colors.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    hue : str
        Hue variable name
    original_data : DataFrame
        Original (non-normalized) data
    hue_categories : list or None
        Ordered categories for the hue variable. If provided, legend order
        will follow this ordering.
    """
    from matplotlib.collections import LineCollection

    should_reverse = hue_categories is not None

    if hue_categories is not None:
        # Use the provided ordered categories as numpy array for consistency
        # Reverse them for the legend display (inverted order)
        hue_values = np.array(list(reversed(hue_categories)))
    else:
        # Use unique values from data (in order of appearance)
        hue_values = original_data[hue].unique()

    line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]

    if line_collections:
        lc = line_collections[0]
        colors = lc.get_colors()

        handles, labels = _create_legend_handles(
            colors, hue_values, reverse_colors=should_reverse
        )

        _clear_figure_legends(ax)

        # Add legend with proper font size from rcParams
        ax.legend(
            handles,
            labels,
            title=hue,
            fontsize=plt.rcParams["legend.fontsize"],
            title_fontsize=plt.rcParams["legend.title_fontsize"],
        )


def _update_existing_legend_fonts(legend: Any) -> None:
    """
    Update legend font sizes to match current rcParams.

    Parameters
    ----------
    legend : Legend
        Matplotlib legend object
    """
    for text in legend.get_texts():
        text.set_fontsize(plt.rcParams["legend.fontsize"])
    legend.get_title().set_fontsize(plt.rcParams["legend.title_fontsize"])


def _configure_legend(
    ax: plt.Axes,
    hue: str,
    original_data: pd.DataFrame,
    hue_categories: Optional[List] = None,
) -> None:
    """
    Configure legend for hue variable.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    hue : str
        Hue variable name
    original_data : DataFrame
        Original (non-normalized) data
    hue_categories : list or None
        Ordered categories for the hue variable. If provided, legend order
        will follow this ordering.
    """
    existing_legend = ax.get_legend()

    if existing_legend is None:
        _add_legend_from_line_collection(ax, hue, original_data, hue_categories)
        existing_legend = ax.get_legend()
    else:
        # Legend already exists (created by seaborn objects in newer versions)
        _update_existing_legend_fonts(existing_legend)

    # Position legend outside plot area on the right side
    if existing_legend is not None:
        existing_legend.set_bbox_to_anchor((1.05, 0.5))
        existing_legend.set_loc("center left")


def _configure_tick_params(ax: plt.Axes, axis: Literal["x", "y"]) -> None:
    """
    Configure tick parameters for an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    axis : {'x', 'y'}
        Axis name
    """
    labelsize_key = f"{axis}tick.labelsize"
    ax.tick_params(axis=axis, labelsize=plt.rcParams[labelsize_key])


def _fix_shared_axis_ticks(
    ax: plt.Axes,
    orient: Literal["v", "h"],
    global_min: float,
    global_max: float,
) -> None:
    """
    Fix tick labels for shared axes to show original data range.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    orient : {'v', 'h'}
        Orientation
    global_min : float
        Minimum value of global range
    global_max : float
        Maximum value of global range
    """
    if orient == "v":
        # Vertical with sharey: fix y-axis ticks
        yticks = ax.get_yticks()
        new_labels = _map_normalized_ticks_to_range(yticks, global_min, global_max)
        ax.set_yticks(yticks, labels=new_labels)
        _configure_tick_params(ax, "x")
        _configure_tick_params(ax, "y")
    else:
        # Horizontal with sharex: fix x-axis ticks
        xticks = ax.get_xticks()
        new_labels = _map_normalized_ticks_to_range(xticks, global_min, global_max)
        ax.set_xticks(xticks, labels=new_labels)
        _configure_tick_params(ax, "x")
        _configure_tick_params(ax, "y")

        # Fix inverted y-axis in horizontal orientation
        _fix_inverted_yaxis_if_needed(ax)


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
    # For independent axes, we draw custom tick labels for values on each variable axis,
    # but keep the variable names on the cross-axis (x for vertical, y for horizontal)

    if orient == "v":
        # Vertical: clear y-axis ticks (values), keep x-axis ticks (variable names)
        ax.set_yticks([])
        # Hide tick marks, keep labels, and ensure labels use the correct font size
        ax.tick_params(axis="x", length=0, labelsize=plt.rcParams["xtick.labelsize"])
    else:  # horizontal
        # Horizontal: clear x-axis ticks (values), keep y-axis ticks (variable names)
        ax.set_xticks([])
        # Hide tick marks, keep labels, and ensure labels use the correct font size
        ax.tick_params(axis="y", length=0, labelsize=plt.rcParams["ytick.labelsize"])

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

    # Extract ordered categories for hue if it's categorical
    hue_categories = None
    if hue is not None:
        # Check if hue was already processed in _normalize_data (it would be in categorical_info)
        if hue in categorical_info:
            hue_categories = categorical_info[hue].get("categories")
        else:
            # Hue was not in vars, so it wasn't processed by _normalize_data
            # Check if it's categorical and apply category_orders if provided
            if hue in data.columns:
                dtype = data[hue].dtype
                is_categorical = (
                    pd.api.types.is_bool_dtype(dtype)
                    or pd.api.types.is_datetime64_any_dtype(dtype)
                    or not pd.api.types.is_numeric_dtype(dtype)
                )

                if is_categorical:
                    # Determine the category order for the hue column
                    if category_orders and hue in category_orders:
                        hue_categories = category_orders[hue]
                    else:
                        hue_categories = data[hue].unique().tolist()

    # If no axes provided, get the current axes to ensure we use the current figure
    # This prevents Seaborn Objects from creating its own internal figure
    if ax is None:
        ax = plt.gca()

    # Create Seaborn Objects plot
    # Pass original categorical hue data if hue exists
    original_hue_series = original_data[hue] if hue is not None else None
    plot = _create_seaborn_plot(
        normalized_df,
        vars,
        hue,
        orient,
        alpha,
        linewidth,
        palette,
        ax,
        original_hue_data=original_hue_series,
        hue_categories=hue_categories,
        **kwargs,
    )

    # Render plot - this must happen within the same plotting context
    # to inherit font sizes, line widths, etc. from seaborn contexts
    plot.plot()

    # Since we always set ax to plt.gca() if None, we can use it directly
    result_ax = ax

    # Add legend manually when hue is specified
    if hue is not None:
        _configure_legend(result_ax, hue, original_data, hue_categories)

    # Post-process for axes
    use_independent = (orient in ["v", "y"] and not sharey) or (
        orient in ["h", "x"] and not sharex
    )

    # Normalize orient to 'v' or 'h'
    normalized_orient: Literal["v", "h"] = "v" if orient in ["v", "y"] else "h"

    # Clear axis labels (done for both independent and shared axes)
    _clear_axis_labels(result_ax)

    if use_independent:
        # Independent axes: custom tick labels for each variable
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
            _fix_inverted_yaxis_if_needed(result_ax)
    else:
        # Shared axes: fix tick labels to show original data range instead of [0, 1]
        numeric_vars = [v for v in vars if v not in categorical_info]
        if numeric_vars:
            # All numeric vars have the same global range when shared scaling is used
            global_min, global_max = original_ranges[numeric_vars[0]]
            _fix_shared_axis_ticks(result_ax, normalized_orient, global_min, global_max)

    return result_ax
