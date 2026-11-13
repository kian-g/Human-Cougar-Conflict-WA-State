"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

Checks the assumptions behind the Result Validity tests planned for RQ1
(Mann-Whitney U and Welch's t-test on distance to road, WUI vs. Wildland)
and RQ2 (Spearman correlation between road density and cougar records
per grid cell).
"""

import numpy as np
import pandas as pd
from scipy import stats

DENSITY_EDGES = [-np.inf, 0, 0.5, 1, 1.5, 2, 3, np.inf]
DENSITY_LABELS = ["0 (no road)", "0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3",
                  "3+"]


def distance_assumptions(points, column="dist_to_road_m",
                         groups=("WUI", "Wildland")):
    """
    Returns one row per WUI group with the sample size, median, mean,
    standard deviation, skew, and Shapiro-Wilk p-value of `column`, plus
    the skew and Shapiro-Wilk p-value after a log(1 + d) transform.
    """
    rows = []
    for group in groups:
        values = points.loc[points["wui_group"] == group, column]
        logged = np.log1p(values)
        rows.append({
            "group": group,
            "n": len(values),
            "median": values.median(),
            "mean": values.mean(),
            "sd": values.std(),
            "skew": stats.skew(values),
            "shapiro_p": stats.shapiro(values).pvalue,
            "log_skew": stats.skew(logged),
            "log_shapiro_p": stats.shapiro(logged).pvalue,
        })
    return pd.DataFrame(rows).set_index("group")


def levene_p(points, column="dist_to_road_m", groups=("WUI", "Wildland")):
    """
    Returns the p-value of Levene's test for equal variances of `column`
    across the WUI groups in `groups`.
    """
    samples = [points.loc[points["wui_group"] == group, column]
               for group in groups]
    return stats.levene(*samples).pvalue


def wilson_interval(k, n, z=1.96):
    """
    Returns the Wilson score interval (low, high) for a proportion with `k`
    successes out of `n` trials, at the confidence level given by the
    normal quantile `z` (1.96 for 95%). Works elementwise on arrays; bins
    with n = 0 give NaN.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = n + z ** 2
        center = (k + z ** 2 / 2) / denom
        half = z / denom * np.sqrt(k * (n - k) / n + z ** 2 / 4)
    return center - half, center + half


def presence_by_density(grid, edges=DENSITY_EDGES, labels=DENSITY_LABELS):
    """
    Bins grid cells by `road_density_km2` and returns, for each bin, the
    number of cells, the share of cells with at least one GBIF record
    (`share`) with its 95% Wilson interval (`ci_low`, `ci_high`), and the
    mean number of records per cell (`mean_count`).
    """
    bins = pd.cut(grid["road_density_km2"], edges, labels=labels)
    present = grid["cougar_count"] > 0
    cells = present.groupby(bins, observed=False).size()
    hits = present.groupby(bins, observed=False).sum()
    low, high = wilson_interval(hits, cells)
    return pd.DataFrame({
        "cells": cells,
        "share": present.groupby(bins, observed=False).mean(),
        "ci_low": low,
        "ci_high": high,
        "mean_count": grid["cougar_count"].groupby(
            bins, observed=False).mean(),
    })
