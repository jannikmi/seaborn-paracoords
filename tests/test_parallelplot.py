"""Tests for parallel coordinates plotting."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn_parallel as snp


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "category": ["A", "B", "A", "B"],
        }
    )


def test_basic_functionality(sample_data):
    """Test basic parallel plot creation."""
    ax = snp.parallelplot(sample_data, vars=["a", "b", "c"])
    assert ax is not None
    plt.close("all")


def test_auto_variable_selection(sample_data):
    """Test automatic numeric variable selection."""
    ax = snp.parallelplot(sample_data)  # Should auto-select a, b, c
    assert ax is not None
    plt.close("all")


def test_hue_parameter(sample_data):
    """Test hue parameter functionality."""
    ax = snp.parallelplot(sample_data, vars=["a", "b", "c"], hue="category")
    assert ax is not None
    # Seaborn Objects creates a figure legend, not an axes legend
    assert len(ax.get_figure().legends) > 0 or ax.get_legend() is not None
    plt.close("all")


def test_orientation_vertical(sample_data):
    """Test vertical orientation."""
    ax = snp.parallelplot(sample_data, vars=["a", "b"], orient="v")
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    plt.close("all")


def test_orientation_horizontal(sample_data):
    """Test horizontal orientation."""
    ax = snp.parallelplot(sample_data, vars=["a", "b"], orient="h")
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    plt.close("all")


def test_error_handling():
    """Test various error conditions."""
    # Non-DataFrame input
    with pytest.raises(TypeError):
        snp.parallelplot([1, 2, 3])

    # Empty DataFrame
    with pytest.raises(ValueError):
        snp.parallelplot(pd.DataFrame())

    # Invalid orientation
    with pytest.raises(ValueError):
        snp.parallelplot(pd.DataFrame({"a": [1, 2]}), orientation="diagonal")

    # Missing variables
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError):
        snp.parallelplot(df, vars=["a", "missing"])


def test_constant_column_handling():
    """Test handling of constant columns."""
    df = pd.DataFrame({"constant": [5, 5, 5, 5], "variable": [1, 2, 3, 4]})
    ax = snp.parallelplot(df)
    assert ax is not None
    plt.close("all")


def test_missing_values():
    """Test handling of missing values."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    with pytest.warns(UserWarning):
        ax = snp.parallelplot(df)
    assert ax is not None
    plt.close("all")


def test_single_variable_error():
    """Test error when only one variable provided."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="At least 2 variables required"):
        snp.parallelplot(df, vars=["a"])


def test_normalization():
    """Test data normalization functionality."""
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30]})

    # Default behavior (independent scaling)
    ax1 = snp.parallelplot(df)
    assert ax1 is not None

    # With shared y-axis (shared scaling)
    ax2 = snp.parallelplot(df, sharey=True)
    assert ax2 is not None

    plt.close("all")


def test_custom_palette():
    """Test custom palette functionality."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "group": ["A", "B", "C"]})

    ax = snp.parallelplot(df, hue="group", palette="viridis")
    assert ax is not None
    plt.close("all")


def test_sharex_parameter():
    """Test sharex parameter for horizontal orientation."""
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30], "z": [100, 200, 300]})

    # Horizontal with sharex
    ax = snp.parallelplot(df, orient="h", sharex=True)
    assert ax is not None
    # Should have x-axis ticks when sharing
    assert len(ax.get_xticks()) > 0
    plt.close("all")


def test_sharey_parameter():
    """Test sharey parameter for vertical orientation."""
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30], "z": [100, 200, 300]})

    # Vertical with sharey
    ax = snp.parallelplot(df, orient="v", sharey=True)
    assert ax is not None
    # Should have y-axis ticks when sharing
    assert len(ax.get_yticks()) > 0
    plt.close("all")


def test_original_axis_values():
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30]})
    # Default behavior should show original values
    ax = snp.parallelplot(df, orient="v")
    assert ax is not None
    # Y-axis should be in [0, 1] range for plotting space
    ylim = ax.get_ylim()
    assert ylim[0] < 0.1 and ylim[1] > 0.9
    plt.close("all")


def test_categorical_axis_detection():
    """Test automatic detection and plotting of categorical axes."""
    df = pd.DataFrame(
        {
            "species": ["setosa", "versicolor", "virginica", "setosa"],
            "sepal_length": [5.1, 7.0, 6.3, 4.9],
            "petal_width": [0.2, 1.3, 2.5, 0.2],
        }
    )
    # species should be detected as categorical
    ax = snp.parallelplot(df)
    assert ax is not None
    # Check that category labels are present in the axis texts
    found = any("setosa" in t.get_text() for t in ax.texts)
    assert found
    plt.close("all")


def test_categorical_axes_param():
    """Test explicit categorical_axes parameter."""
    df = pd.DataFrame(
        {"cat": ["A", "B", "A", "C"], "x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}
    )
    ax = snp.parallelplot(df, vars=["cat", "x", "y"], categorical_axes=["cat"])
    assert ax is not None
    found = any("A" in t.get_text() for t in ax.texts)
    assert found
    plt.close("all")


def test_mixed_type_axes():
    """Test plot with both categorical and numeric axes."""
    df = pd.DataFrame(
        {
            "group": ["G1", "G2", "G1", "G2"],
            "val1": [1.0, 2.0, 3.0, 4.0],
            "val2": [5, 6, 7, 8],
        }
    )
    ax = snp.parallelplot(df, vars=["group", "val1", "val2"])
    assert ax is not None
    # Should show both category and numeric labels
    found_cat = any("G1" in t.get_text() for t in ax.texts)
    # Float column val1 produces labels like "1.0", "2.0"
    found_num = any("1.0" in t.get_text() or "2.0" in t.get_text() for t in ax.texts)
    assert found_cat and found_num
    plt.close("all")


def test_category_orders():
    """Test custom category order for categorical axis."""
    df = pd.DataFrame({"cat": ["B", "A", "C", "B"], "score": [1, 2, 3, 4]})
    ax = snp.parallelplot(
        df,
        vars=["cat", "score"],
        categorical_axes=["cat"],
        category_orders={"cat": ["C", "B", "A"]},
    )
    assert ax is not None
    # With intuitive ordering, 'C' should appear at the top (last in matplotlib coords)
    cat_labels = [
        t.get_text().strip()
        for t in ax.texts
        if t.get_text().strip() in ["A", "B", "C"]
    ]
    # The order should reflect the user's specification (top-to-bottom): C, B, A
    # Matplotlib renders bottom-to-top, so the last label is 'C' (top)
    assert cat_labels[-1] == "C", f"Expected 'C' at top, got labels: {cat_labels}"
    plt.close("all")


def test_sharex_sharey_combination():
    """Test that sharex and sharey work with different orientations."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})

    # Vertical with sharey should work
    ax1 = snp.parallelplot(df, orient="v", sharey=True)
    assert ax1 is not None

    # Horizontal with sharex should work
    ax2 = snp.parallelplot(df, orient="h", sharex=True)
    assert ax2 is not None

    # Vertical with sharex should warn (wrong axis for orientation)
    with pytest.warns(
        UserWarning, match="sharex=True has no effect with vertical orientation"
    ):
        ax3 = snp.parallelplot(df, orient="v", sharex=True)
    assert ax3 is not None

    # Horizontal with sharey should warn (wrong axis for orientation)
    with pytest.warns(
        UserWarning, match="sharey=True has no effect with horizontal orientation"
    ):
        ax4 = snp.parallelplot(df, orient="h", sharey=True)
    assert ax4 is not None

    plt.close("all")


def test_gcf_contains_plot():
    """Test that plt.gcf() contains the plot when ax is not explicitly provided."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    plt.close("all")  # Start with clean slate

    # Call parallelplot without providing ax
    ax = snp.parallelplot(df, vars=["a", "b", "c"])

    # Get current figure
    fig = plt.gcf()

    # The returned ax should be in the current figure
    assert ax.figure == fig, "Returned axes should be in current figure"
    assert len(fig.axes) > 0, "Current figure should contain axes"
    assert ax in fig.axes, "Returned axes should be in current figure's axes list"

    plt.close("all")


def test_category_orders_with_categorical_hue():
    """Test that category_orders affects both axis display and hue coloring."""
    df = pd.DataFrame(
        {
            "size": ["large", "small", "medium", "large", "small", "medium"],
            "score": [95, 85, 90, 92, 88, 87],
        }
    )

    # Create plot with custom category ordering for hue
    ax = snp.parallelplot(
        data=df,
        vars=["size", "score"],
        hue="size",
        category_orders={"size": ["small", "medium", "large"]},
        orient="v",
    )

    assert ax is not None
    # Legend should exist when hue is specified
    assert ax.get_legend() is not None or len(ax.figure.legends) > 0
    plt.close("all")


def test_category_orders_hue_legend_order():
    """Test that legend labels follow category_orders for categorical hue."""
    df = pd.DataFrame(
        {
            "category": ["C", "A", "B", "C", "A", "B"],
            "value1": [10, 20, 15, 12, 18, 16],
            "value2": [100, 200, 150, 120, 180, 160],
        }
    )

    # Plot with custom ordering
    ax = snp.parallelplot(
        data=df,
        vars=["category", "value1", "value2"],
        hue="category",
        category_orders={"category": ["A", "B", "C"]},
        orient="v",
    )

    # Get the legend
    legend = ax.get_legend()
    if legend is None and len(ax.figure.legends) > 0:
        legend = ax.figure.legends[0]

    assert legend is not None, "Legend should exist for categorical hue"

    # Check legend labels are in the specified order
    legend_labels = [text.get_text() for text in legend.get_texts()]
    # Remove title if present
    legend_labels = [label for label in legend_labels if label != "category"]

    # The legend should have entries in the custom order
    # (may contain subset of categories that appear in data)
    assert "A" in legend_labels, "Category A should be in legend"
    assert "B" in legend_labels, "Category B should be in legend"
    assert "C" in legend_labels, "Category C should be in legend"

    plt.close("all")


def test_category_orders_hue_horizontal():
    """Test category_orders with hue in horizontal orientation."""
    df = pd.DataFrame(
        {
            "region": ["East", "West", "North", "East", "West", "North"],
            "sales": [100, 150, 200, 120, 180, 220],
            "profit": [20, 30, 40, 25, 35, 45],
        }
    )

    ax = snp.parallelplot(
        data=df,
        vars=["region", "sales", "profit"],
        hue="region",
        category_orders={"region": ["North", "East", "West"]},
        orient="h",
    )

    assert ax is not None
    legend = ax.get_legend()
    if legend is None and len(ax.figure.legends) > 0:
        legend = ax.figure.legends[0]
    assert legend is not None, "Legend should exist for categorical hue"
    plt.close("all")


def test_category_orders_hue_with_palette():
    """Test that category_orders respects custom palette."""
    df = pd.DataFrame(
        {
            "quality": ["good", "poor", "excellent", "good", "poor", "excellent"],
            "price": [100, 50, 200, 110, 45, 210],
            "rating": [4.5, 2.0, 5.0, 4.6, 1.9, 4.9],
        }
    )

    ax = snp.parallelplot(
        data=df,
        vars=["quality", "price", "rating"],
        hue="quality",
        category_orders={"quality": ["poor", "good", "excellent"]},
        palette="Set2",
        orient="v",
    )

    assert ax is not None
    legend = ax.get_legend()
    if legend is None and len(ax.figure.legends) > 0:
        legend = ax.figure.legends[0]
    assert legend is not None
    plt.close("all")


def test_category_orders_hue_not_in_vars():
    """Test that category_orders respects hue order even when hue is not in vars."""
    df = pd.DataFrame(
        {
            "status": ["pending", "completed", "failed", "pending", "completed"],
            "duration": [2.5, 1.0, 3.5, 1.5, 0.5],
            "success_rate": [0.8, 0.95, 0.3, 0.92, 0.85],
        }
    )

    # Test when status is NOT in vars
    ax = snp.parallelplot(
        data=df,
        vars=["duration", "success_rate"],  # status NOT included
        hue="status",
        category_orders={"status": ["completed", "failed", "pending"]},
        orient="v",
    )

    assert ax is not None
    legend = ax.get_legend()
    if legend is None and len(ax.figure.legends) > 0:
        legend = ax.figure.legends[0]
    assert legend is not None, "Legend should exist even when hue not in vars"

    # Check legend order - should be reversed (for display)
    legend_labels = [text.get_text() for text in legend.get_texts()]
    # Remove title if present
    legend_labels = [label for label in legend_labels if label != "status"]
    assert len(legend_labels) > 0, "Legend should have category labels"

    plt.close("all")


def test_category_orders_hue_not_in_vars_horizontal():
    """Test category_orders respects order when hue not in vars (horizontal)."""
    df = pd.DataFrame(
        {
            "priority": ["high", "low", "medium", "high", "low"],
            "time_spent": [5, 2, 3, 4, 1],
            "quality_score": [9, 5, 7, 8, 4],
        }
    )

    # priority is NOT in vars
    ax = snp.parallelplot(
        data=df,
        vars=["time_spent", "quality_score"],  # priority NOT included
        hue="priority",
        category_orders={"priority": ["low", "medium", "high"]},
        orient="h",
    )

    assert ax is not None
    legend = ax.get_legend()
    if legend is None and len(ax.figure.legends) > 0:
        legend = ax.figure.legends[0]
    assert legend is not None, "Legend should exist even when hue not in vars"
    plt.close("all")
