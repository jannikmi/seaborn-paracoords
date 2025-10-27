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
    assert ax.get_legend() is not None
    plt.close("all")


def test_orientation_vertical(sample_data):
    """Test vertical orientation."""
    ax = snp.parallelplot(sample_data, vars=["a", "b"], orientation="vertical")
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    plt.close("all")


def test_orientation_horizontal(sample_data):
    """Test horizontal orientation."""
    ax = snp.parallelplot(sample_data, vars=["a", "b"], orientation="horizontal")
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
    ax = snp.parallelplot(df, orientation="horizontal", sharex=True)
    assert ax is not None
    # Should have x-axis ticks when sharing
    assert len(ax.get_xticks()) > 0
    plt.close("all")


def test_sharey_parameter():
    """Test sharey parameter for vertical orientation."""
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30], "z": [100, 200, 300]})

    # Vertical with sharey
    ax = snp.parallelplot(df, orientation="vertical", sharey=True)
    assert ax is not None
    # Should have y-axis ticks when sharing
    assert len(ax.get_yticks()) > 0
    plt.close("all")


def test_original_axis_values():
    """Test that original axis values are shown by default."""
    df = pd.DataFrame({"x": [0, 50, 100], "y": [10, 20, 30]})

    # Default behavior should show original values
    ax = snp.parallelplot(df, orientation="vertical")
    assert ax is not None
    # Y-axis should be in [0, 1] range for plotting space
    ylim = ax.get_ylim()
    assert ylim[0] < 0.1 and ylim[1] > 0.9
    plt.close("all")


def test_sharex_sharey_combination():
    """Test that sharex and sharey work with different orientations."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})

    # Vertical with sharey should work
    ax1 = snp.parallelplot(df, orientation="vertical", sharey=True)
    assert ax1 is not None

    # Horizontal with sharex should work
    ax2 = snp.parallelplot(df, orientation="horizontal", sharex=True)
    assert ax2 is not None

    # Vertical with sharex should warn (wrong axis for orientation)
    with pytest.warns(
        UserWarning, match="sharex=True has no effect with orientation='vertical'"
    ):
        ax3 = snp.parallelplot(df, orientation="vertical", sharex=True)
    assert ax3 is not None

    # Horizontal with sharey should warn (wrong axis for orientation)
    with pytest.warns(
        UserWarning, match="sharey=True has no effect with orientation='horizontal'"
    ):
        ax4 = snp.parallelplot(df, orientation="horizontal", sharey=True)
    assert ax4 is not None

    plt.close("all")
