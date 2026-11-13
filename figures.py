"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

One function per EDA figure. Each takes the data it plots and the path to
save the PNG to, applies the shared style from plot_style, and closes the
figure when done. Each figure is drawn at the width it is printed at in
eda.tex (a fraction of plot_style.TEXT_W), so its text prints at the font
sizes set in plot_style.
"""

import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plot_style as ps
import rasterio
import seaborn as sns
from spatial import WUI_CODES

HEXBIN_DOWNSAMPLE = 10
HEXBIN_NX = 80
HEXBIN_MINCNT = 50
WUI_GROUP_ORDER = ["WUI", "Wildland", "Other"]
MIN_PLOT_DIST_M = 1
MAP_MARKER_SIZE = 1.2


def _save(fig, path):
    """
    Saves `fig` to `path` with the project's export settings and closes it.
    """
    fig.savefig(path)
    plt.close(fig)


def _title(fig, text):
    """
    Adds `text` as a bold title at the left edge of `fig`, wrapped so each
    line fits the figure's width.
    """
    chars = int(fig.get_figwidth() * 72 / (0.6 * ps.TITLE_SIZE))
    fig.suptitle(textwrap.fill(text, chars), x=0.01, ha="left",
                 fontsize=ps.TITLE_SIZE, fontweight="bold", color=ps.INK)


def plot_incidents_per_type(incidents, path):
    """
    Bar chart of WDFW incident counts by Incident Type
    """
    fig, ax = plt.subplots(figsize=(0.42 * ps.TEXT_W, 2.2))
    sns.countplot(data=incidents, x="Incident Type", hue="Incident Type",
                  order=ps.INCIDENT_ORDER, palette=ps.INCIDENT_COLORS,
                  saturation=1, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="{:,.0f}", padding=2, color=ps.INK_2,
                     fontsize=6.5)
    ax.set_xticks(range(len(ps.INCIDENT_ORDER)),
                  [textwrap.fill(t, 11, break_long_words=False)
                   for t in ps.INCIDENT_ORDER],
                  fontsize=6)
    ax.margins(y=0.12)
    ax.set_xlabel("")
    ax.set_ylabel("Number of incidents")
    _title(fig, "Cougar Incidents per Type")
    fig.tight_layout()
    _save(fig, path)


def plot_incidents_by_county(incidents, path):
    """
    Heatmap of WDFW incident counts
    """
    counts = pd.crosstab(incidents["County"], incidents["Incident Type"])
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    counts = counts.T.reindex(ps.INCIDENT_ORDER)

    fig, ax = plt.subplots(figsize=(ps.TEXT_W, 1.8))
    sns.heatmap(counts, annot=True, fmt="d", cmap=ps.SEQ_CMAP, cbar=False,
                linewidths=0.3, linecolor="white", ax=ax,
                annot_kws={"fontsize": 5})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.set_yticklabels([textwrap.fill(t, 16) for t in counts.index],
                       rotation=0, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor", fontsize=6)
    _title(fig, "Cougar Incidents by County")
    fig.tight_layout()
    _save(fig, path)


def plot_incident_map(incidents, roads, boundary, path):
    """
    Map of every WDFW incident colored by type
    """
    fig, ax = plt.subplots(figsize=(0.56 * ps.TEXT_W, 2.6))
    roads.plot(ax=ax, linewidth=0.1, color=ps.BASELINE)
    boundary.boundary.plot(ax=ax, color=ps.MUTED, linewidth=0.4)
    for incident_type in ps.INCIDENT_ORDER:
        pts = incidents[incidents["Incident Type"] == incident_type]
        pts.plot(ax=ax, color=ps.INCIDENT_COLORS[incident_type],
                 markersize=MAP_MARKER_SIZE, linewidth=0,
                 label=f"{incident_type} ({len(pts):,})")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0), markerscale=4,
              fontsize=5.5, handletextpad=0.1, borderaxespad=0)
    ax.set_axis_off()
    _title(fig, "Where Cougar Incidents Were Reported")
    fig.tight_layout()
    _save(fig, path)


def plot_gbif_by_year(gbif, path):
    """
    Bar chart of the number of GBIF records observed in each year. Records
    with no year are left out.
    """
    year_counts = gbif["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(0.49 * ps.TEXT_W, 1.9))
    ax.bar(year_counts.index, year_counts.values, color=ps.HIGHLIGHT,
           width=0.8)
    ax.set_xlabel("Year observed")
    ax.set_ylabel("GBIF records")
    _title(fig, "Most GBIF Records Are From 2015 or Later")
    fig.tight_layout()
    _save(fig, path)


def plot_gbif_basis(gbif, path):
    """
    Horizontal bar chart of GBIF records by basisOfRecord
    """
    basis = gbif["basisOfRecord"].value_counts()
    labels = basis.index.str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(0.49 * ps.TEXT_W, 1.5))
    ax.barh(labels, basis.values, color=ps.HIGHLIGHT)
    ax.invert_yaxis()
    for y, count in enumerate(basis.values):
        share = count / basis.sum()
        share = f"{share:.0%}" if share >= 0.005 else "<1%"
        ax.text(count + 4, y, f"{count:,} ({share})", va="center",
                color=ps.INK_2, fontsize=6.5)
    ax.set_xlabel("GBIF records")
    ax.set_xlim(0, basis.max() * 1.35)
    _title(fig, "GBIF Records Are Mostly Human Observations")
    fig.tight_layout()
    _save(fig, path)


def plot_wui_classes(classes, path):
    """
    Log-scale horizontal bar chart of raster cells per WUI class, with the
    Intermix and Interface classes highlighted and each bar labeled with
    its cell count and share of all classified cells.
    """
    is_wui = classes["VALUE"].isin(WUI_CODES)
    labels = (classes["WUI_DESC_N"]
              .str.replace("Structure Density", "Density")
              .str.replace("Non-Vegetated", "Non-Veg."))
    colors = dict(zip(labels, np.where(is_wui, ps.HIGHLIGHT, ps.CONTEXT)))
    data = classes.assign(label=labels)
    fig, ax = plt.subplots(figsize=(0.46 * ps.TEXT_W, 2.7))
    sns.barplot(data=data, x="COUNT", y="label", hue="label",
                legend=False, saturation=1, palette=colors, ax=ax)
    ax.set_xlabel("Pixel count (log scale)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6)

    pct = classes["COUNT"] / classes["COUNT"].sum() * 100
    for y, (count, share) in enumerate(zip(classes["COUNT"], pct)):
        share_text = f"{share:.1f}%" if share >= 0.1 else "<0.1%"
        ax.text(count * 1.3, y, f"{count:,} ({share_text})", va="center",
                fontsize=5.5, color=ps.INK_2)
    ax.set_xscale("log")

    ax.set_xlim(2e4, classes["COUNT"].max() * 1000)
    ax.set_xticks([1e5, 1e7], ["100k", "10M"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.spines["bottom"].set_bounds(2e4, classes["COUNT"].max())
    ax.xaxis.set_label_coords(0.3, -0.11)

    wui_pct = classes.loc[is_wui, "COUNT"].sum() / classes["COUNT"].sum()
    _title(fig, f"Intermix and Interface WUI Cover {wui_pct:.1%} of Total "
                "Area")
    fig.tight_layout()
    _save(fig, path)


def plot_wui_hexbin(tif, path):
    """
    Hexbin map of the share of each hexagon that is WUI
    """
    with rasterio.open(tif) as src:
        cells = src.read(1, out_shape=(src.height // HEXBIN_DOWNSAMPLE,
                                       src.width // HEXBIN_DOWNSAMPLE))
        transform = src.transform * src.transform.scale(HEXBIN_DOWNSAMPLE,
                                                        HEXBIN_DOWNSAMPLE)
        nodata = src.nodata
    rows, cols = np.nonzero(cells != nodata)
    is_wui = np.isin(cells[rows, cols], WUI_CODES) * 100
    xs, ys = transform * (cols, rows)
    ny = int(HEXBIN_NX * (ys.max() - ys.min()) / (xs.max() - xs.min())
             / np.sqrt(3))

    fig, ax = plt.subplots(figsize=(0.52 * ps.TEXT_W, 1.9))
    hexes = ax.hexbin(xs, ys, C=is_wui, reduce_C_function=np.mean,
                      gridsize=(HEXBIN_NX, ny), mincnt=HEXBIN_MINCNT,
                      cmap=ps.SEQ_CMAP, vmin=0, vmax=50,
                      linewidths=0, edgecolors="white")
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(hexes, ax=ax, shrink=0.6, extend="max",
                        label="% of hexagon in WUI")
    cbar.outline.set_visible(False)
    _title(fig, "WUI Clusters Along I-5 and Around Spokane")
    fig.tight_layout()
    _save(fig, path)


def plot_road_density_map(grid, boundary, path):
    """
    Map of county road density (km / km^2) on the 5 km grid.
    """
    fig, ax = plt.subplots(figsize=(0.5 * ps.TEXT_W, 1.9))
    grid.plot(ax=ax, column="road_density_km2", cmap=ps.SEQ_CMAP, vmin=0,
              vmax=4, legend=True, linewidth=0,
              legend_kwds={"shrink": 0.6, "extend": "max",
                           "label": "County road density\n"
                                    "(km of road / km$^2$)"})
    boundary.boundary.plot(ax=ax, color=ps.INK_2, linewidth=0.4)
    ax.set_axis_off()
    _title(fig, "County Road Density Peaks Around the I-5 Cities and "
                "Spokane")
    fig.tight_layout()
    _save(fig, path)


def plot_road_density_hist(grid, path):
    """
    Histogram of county road density per grid cell. Cells above 3 km per
    km^2 are grouped into the last bar, and the median is marked.
    """
    density = grid["road_density_km2"]
    fig, ax = plt.subplots(figsize=(0.48 * ps.TEXT_W, 2.1))
    ax.hist(density.clip(upper=3), bins=np.arange(0, 3.25, 0.25),
            color=ps.HIGHLIGHT, edgecolor="white", linewidth=0.5)
    ticks = np.arange(0, 3.5, 0.5)
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + ["3+"])
    median = density.median()
    ax.axvline(median, color=ps.INK, linewidth=0.6, linestyle="--")
    ax.text(median + 0.05, ax.get_ylim()[1] * 0.92, f"median {median:.2f}",
            color=ps.INK_2, fontsize=6.5)
    ax.set_xlabel("County road density (km / km$^2$, 5 km cells)")
    ax.set_ylabel("Grid cells")
    _title(fig, "Road Density Is Right-Skewed, With Many Roadless Cells")
    fig.tight_layout()
    _save(fig, path)


def plot_dist_to_road_by_wui(gbif, incidents, path):
    """
    Side-by-side log-scale box plots of distance to the nearest county road
    by WUI group, for GBIF occurrences and WDFW incidents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_W, 2.5), sharey=True)
    datasets = (("GBIF occurrences", gbif), ("WDFW incidents", incidents))
    for ax, (name, points) in zip(axes, datasets):
        data = points[points["wui_group"].isin(WUI_GROUP_ORDER)].copy()
        data["plot_dist"] = data["dist_to_road_m"].clip(
            lower=MIN_PLOT_DIST_M)
        sns.boxplot(data=data, x="wui_group", y="plot_dist",
                    order=WUI_GROUP_ORDER, color=ps.HIGHLIGHT, saturation=1,
                    fliersize=1, width=0.5, linewidth=0.6,
                    linecolor=ps.INK_2,
                    flierprops={"markeredgecolor": ps.MUTED}, ax=ax)
        sizes = data["wui_group"].value_counts()
        ax.set_xticks(range(len(WUI_GROUP_ORDER)),
                      [f"{g}\n(n={sizes.get(g, 0):,})"
                       for g in WUI_GROUP_ORDER])
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e4)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("WUI class at point")
        ax.set_title(name, fontsize=7.5)
    axes[0].set_ylabel("Distance to nearest county road\n"
                       f"(m, log scale; under {MIN_PLOT_DIST_M} m drawn at "
                       f"{MIN_PLOT_DIST_M} m)")
    _title(fig, "Records in the WUI Sit Closer to County Roads Than "
                "Wildland Records")
    fig.tight_layout()
    _save(fig, path)


def plot_presence_by_density(table, path):
    """
    Bar chart of the share of grid cells with at least one GBIF record in
    each road-density bin, with 95% Wilson intervals and a dashed line at
    the share across all cells, from validity.presence_by_density.
    """
    share = table["share"] * 100
    low = table["ci_low"] * 100
    high = table["ci_high"] * 100
    overall = (table["share"] * table["cells"]).sum() / table["cells"].sum()

    fig, ax = plt.subplots(figsize=(0.75 * ps.TEXT_W, 2.4))
    positions = np.arange(len(table))
    ax.bar(positions, share, color=ps.HIGHLIGHT, width=0.7)
    ax.errorbar(positions, share, yerr=[share - low, high - share],
                fmt="none", ecolor=ps.INK_2, elinewidth=0.6, capsize=2,
                capthick=0.6, label="95% Wilson interval")
    ax.axhline(overall * 100, color=ps.INK_2, linewidth=0.6,
               linestyle="--", label=f"All cells ({overall:.1%})")
    for i, (value, top) in enumerate(zip(share, high)):
        ax.text(i, top + 0.4, f"{value:.1f}%", ha="center", va="bottom",
                color=ps.INK_2, fontsize=6)
    ax.set_xticks(positions, [f"{label}\n(n={n:,})"
                              for label, n in zip(table.index,
                                                  table["cells"])])
    ax.set_xlabel("County road density (km per km$^2$), number of 5 km "
                  "cells in parentheses")
    ax.set_ylabel("Cells with $\\geq$1 GBIF record (%)")
    ax.legend(loc="upper left")
    _title(fig, "GBIF Records Do Not Decline With Road Density")
    fig.tight_layout()
    _save(fig, path)
