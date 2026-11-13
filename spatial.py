"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

Spatial joins between cougar records (GBIF occurrences and WDFW incidents)
and the landscape layers.
"""

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import shapely
from cougar_data import FT_TO_M
from scipy import ndimage

WUI_CODES = list(range(7, 15))  # Intermix (7-10) and Interface (11-14)
WILDLAND_CODES = [1, 99]  # Vegetated Uninhabited, Long-term Non-Buildable
DISTANCE_DOWNSAMPLE = 4  # 30 m cells -> ~120 m cells for distance maps


def wui_group(code, nodata=255):
    """
    Returns the WUI group of a WUI raster class code: "WUI" for Intermix or
    Interface, "Wildland" for uninhabited vegetation or non-buildable land,
    "No data" for the NoData value, and "Other" for everything else.
    """
    if code in WUI_CODES:
        return "WUI"
    if code in WILDLAND_CODES:
        return "Wildland"
    if code == nodata:
        return "No data"
    return "Other"


def distance_to_codes(cells, transform, codes):
    """
    Takes a 2D array of raster class codes and its affine transform, and
    returns an array of the same shape giving, for each cell, the distance
    (in CRS units) to the nearest cell whose code is in `codes`. Cells
    with a code in `codes` have distance 0.
    """
    target = np.isin(cells, codes)
    return ndimage.distance_transform_edt(~target,
                                          sampling=(abs(transform.e),
                                                    transform.a))


def add_wui_columns(points, tif, factor=DISTANCE_DOWNSAMPLE):
    """
    Returns a tuple (joined, mismatched). `joined` is a copy of the
    GeoDataFrame `points` (EPSG:2927) with three new columns: `wui_code`,
    the class of the full-resolution raster cell each point falls in;
    `wui_group`, from wui_group; and `dist_to_wui_m`, the distance in
    meters to the nearest WUI cell on a raster downsampled by `factor`.
    Points whose full-resolution class is WUI get distance 0.
    `mismatched` is the number of WUI points that the coarse raster placed
    outside the WUI (before they were set to 0).
    """
    joined = points.copy()
    coords = list(zip(joined.geometry.x, joined.geometry.y))
    with rasterio.open(tif) as src:
        nodata = int(src.nodata)
        joined["wui_code"] = [value[0] for value in src.sample(coords)]
        small = src.read(1, out_shape=(src.height // factor,
                                       src.width // factor))
        transform = src.transform * src.transform.scale(
            src.width / small.shape[1], src.height / small.shape[0])
    distance = distance_to_codes(small, transform, WUI_CODES)

    rows, cols = rasterio.transform.rowcol(transform, joined.geometry.x,
                                           joined.geometry.y)
    rows = np.clip(rows, 0, small.shape[0] - 1)
    cols = np.clip(cols, 0, small.shape[1] - 1)
    joined["dist_to_wui_m"] = distance[rows, cols] * FT_TO_M
    joined["wui_group"] = [wui_group(code, nodata)
                           for code in joined["wui_code"]]

    in_wui = joined["wui_group"] == "WUI"
    mismatched = int((in_wui & (joined["dist_to_wui_m"] > 0)).sum())
    joined.loc[in_wui, "dist_to_wui_m"] = 0.0
    return joined, mismatched


def nearest_distance_m(points, lines):
    """
    Returns a Series with the distance in meters
    from each point to the nearest geometry in `lines`. Both inputs must
    be in EPSG:2927.
    """
    joined = gpd.sjoin_nearest(points[["geometry"]], lines[["geometry"]],
                               distance_col="distance_ft")
    # Ties return one row per tied line. keep the first for each point
    return joined.groupby(level=0)["distance_ft"].first() * FT_TO_M


def make_grid(boundary_geom, cell_m=5000, min_land_frac=0.25):
    """
    Returns a GeoDataFrame of square cells `cell_m` meters on a side that
    cover `boundary_geom` (EPSG:2927). Each cell has a `cell_id` and its
    area inside the boundary in `land_km2`. Cells with less than
    `min_land_frac` of their area inside the boundary are dropped.
    """
    side = cell_m / FT_TO_M
    xmin, ymin, xmax, ymax = boundary_geom.bounds
    boxes = [shapely.box(x, y, x + side, y + side)
             for x in np.arange(xmin, xmax, side)
             for y in np.arange(ymin, ymax, side)]
    grid = gpd.GeoDataFrame(geometry=boxes, crs="EPSG:2927")
    grid = grid[grid.intersects(boundary_geom)]
    grid["land_km2"] = grid.intersection(boundary_geom).area * FT_TO_M ** 2
    grid["land_km2"] = grid["land_km2"] / 1e6
    cell_km2 = (cell_m / 1000) ** 2
    grid = grid[grid["land_km2"] >= min_land_frac * cell_km2]
    grid = grid.reset_index(drop=True)
    grid["cell_id"] = grid.index
    return grid


def add_road_density(grid, roads):
    """
    Returns a copy of `grid` with `road_km` (km of road inside each cell)
    and `road_density_km2` (road km per km^2 of land) added. Roads are
    clipped to cell edges.
    """
    pieces = gpd.overlay(roads[["geometry"]],
                         grid[["cell_id", "geometry"]],
                         how="intersection", keep_geom_type=True)
    road_km = (pieces.length * FT_TO_M / 1000).groupby(
        pieces["cell_id"]).sum()
    result = grid.copy()
    result["road_km"] = result["cell_id"].map(road_km).fillna(0)
    result["road_density_km2"] = result["road_km"] / result["land_km2"]
    return result


def add_point_counts(grid, points, column):
    """
    Returns a copy of `grid` with a column named `column` giving the number
    of `points` that fall inside each cell.
    """
    joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]],
                       predicate="within")
    counts = joined["cell_id"].value_counts()
    result = grid.copy()
    result[column] = result["cell_id"].map(counts).fillna(0).astype(int)
    return result
