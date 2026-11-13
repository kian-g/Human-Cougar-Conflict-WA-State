"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

Tests for cougar_data, spatial, and validity.
"""

import math
import os
import tempfile

import cougar_data as cd
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import spatial
import validity
from rasterio.transform import Affine

FT_TO_M = cd.FT_TO_M


def close(a, b, tol=1e-6):
    """
    Returns True if numbers `a` and `b` are equal within relative tolerance
    `tol` (or within `tol` of each other when both are near 0).
    """
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def points_ft(coords):
    """
    Returns a GeoDataFrame of points in EPSG:2927 from (x, y) feet tuples.
    """
    return gpd.GeoDataFrame(geometry=[shapely.Point(xy) for xy in coords],
                            crs="EPSG:2927")


def test_wui_group():
    """
    Each WUI class code maps to the right group, including the edges of the
    Intermix/Interface range and the NoData value.
    """
    assert spatial.wui_group(7) == "WUI"
    assert spatial.wui_group(14) == "WUI"
    assert spatial.wui_group(1) == "Wildland"
    assert spatial.wui_group(99) == "Wildland"
    assert spatial.wui_group(0) == "Other"
    assert spatial.wui_group(6) == "Other"
    assert spatial.wui_group(15) == "Other"
    assert spatial.wui_group(255) == "No data"


def test_distance_to_codes():
    """
    On a 3x3 raster of 10-unit cells with one WUI cell in the middle, the
    middle is 0 away, edge cells are 10 away, and corners are 10*sqrt(2).
    """
    cells = np.array([[1, 1, 1],
                      [1, 8, 1],
                      [1, 1, 1]])
    transform = Affine(10, 0, 0, 0, -10, 30)
    distance = spatial.distance_to_codes(cells, transform, spatial.WUI_CODES)
    assert distance[1, 1] == 0
    assert close(distance[0, 1], 10)
    assert close(distance[1, 2], 10)
    assert close(distance[0, 0], 10 * math.sqrt(2))

    # No WUI anywhere except one corner: the far corner is 20*sqrt(2) away
    cells = np.ones((3, 3), dtype=int)
    cells[0, 0] = 12
    distance = spatial.distance_to_codes(cells, transform, spatial.WUI_CODES)
    assert close(distance[2, 2], 20 * math.sqrt(2))


def test_nearest_distance_m():
    """
    A point 100 m beside a road is 100 m from it, a point on the road is
    0 m away, and a point past the end of the road is measured to the end.
    """
    one_km_ft = 1000 / FT_TO_M
    hundred_m_ft = 100 / FT_TO_M
    road = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(0, 0), (one_km_ft, 0)])],
        crs="EPSG:2927")
    points = points_ft([(one_km_ft / 2, hundred_m_ft),
                        (one_km_ft / 4, 0),
                        (one_km_ft + 3 / FT_TO_M * 100,
                         4 / FT_TO_M * 100)])
    distance = spatial.nearest_distance_m(points, road)
    assert close(distance[0], 100)
    assert close(distance[1], 0)
    assert close(distance[2], 500)  # a 300-400-500 triangle


def test_make_grid_drops_slivers():
    """
    A 2.1 x 1 km boundary with 1 km cells keeps the two full cells (1 km^2
    of land each) and drops the 0.1 km^2 sliver.
    """
    side = 1000 / FT_TO_M
    boundary = shapely.box(0, 0, 2.1 * side, side)
    grid = spatial.make_grid(boundary, cell_m=1000, min_land_frac=0.25)
    assert len(grid) == 2
    assert list(grid["cell_id"]) == [0, 1]
    assert all(close(area, 1) for area in grid["land_km2"])

    # With no minimum, the sliver is kept with its partial land area
    grid = spatial.make_grid(boundary, cell_m=1000, min_land_frac=0)
    assert len(grid) == 3
    assert close(grid["land_km2"].min(), 0.1)


def test_road_density_and_counts():
    """
    A 2 km road crossing two 1 km cells puts 1 km of road (density 1.0) in
    each, a cell with no road has density 0, and points are counted in the
    cell they fall inside.
    """
    side = 1000 / FT_TO_M
    boundary = shapely.box(0, 0, 2 * side, 2 * side)
    grid = spatial.make_grid(boundary, cell_m=1000)
    roads = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(0, side / 2), (2 * side, side / 2)])],
        crs="EPSG:2927")
    grid = spatial.add_road_density(grid, roads)
    on_road = grid.geometry.centroid.y < side
    assert all(close(d, 1) for d in grid.loc[on_road, "road_density_km2"])
    assert (grid.loc[~on_road, "road_density_km2"] == 0).all()
    assert close(grid["road_km"].sum(), 2)

    points = points_ft([(side / 2, side / 2), (side / 3, side / 3),
                        (1.5 * side, 1.5 * side)])
    counted = spatial.add_point_counts(grid, points, "n")
    assert counted["n"].sum() == 3
    assert sorted(counted["n"]) == [0, 0, 1, 2]
    assert "n" not in grid.columns  # the input grid is not modified


def test_add_wui_columns():
    """
    On an 8x8 raster of 100 ft cells whose left 4 columns are WUI (class 8)
    and right 4 columns are wildland (class 1), with a NoData cell in the
    top-right corner: a point in the WUI half is WUI at distance 0, a point
    just inside the wildland half is Wildland one coarse cell (200 ft) from
    the WUI on the 2x coarser raster, and a point on the NoData cell is
    "No data".
    """
    cells = np.ones((8, 8), dtype=np.uint8)
    cells[:, :4] = 8
    cells[0, 7] = 255
    transform = Affine(100, 0, 0, 0, -100, 800)
    with tempfile.TemporaryDirectory() as folder:
        tif = os.path.join(folder, "wui.tif")
        with rasterio.open(tif, "w", driver="GTiff", height=8, width=8,
                           count=1, dtype="uint8", crs="EPSG:2927",
                           transform=transform, nodata=255) as dst:
            dst.write(cells, 1)
        points = points_ft([(150, 450), (450, 450), (750, 750)])
        joined, mismatched = spatial.add_wui_columns(points, tif, factor=2)

    assert list(joined["wui_code"]) == [8, 1, 255]
    assert list(joined["wui_group"]) == ["WUI", "Wildland", "No data"]
    assert joined.loc[0, "dist_to_wui_m"] == 0
    # x = 450 ft is in coarse column 2; the nearest WUI is column 1
    assert close(joined.loc[1, "dist_to_wui_m"], 200 * FT_TO_M)
    assert mismatched == 0
    assert "wui_group" not in points.columns


def test_clean_gbif():
    """
    clean_gbif keeps a record inside the boundary and drops one outside it
    and a fossil inside it.
    """
    lonlat_box = shapely.box(-121, 47, -120, 48)
    boundary = gpd.GeoSeries([lonlat_box], crs="EPSG:4326").to_crs(2927)
    raw = pd.DataFrame({
        "decimalLongitude": [-120.5, -110.0, -120.5],
        "decimalLatitude": [47.5, 47.5, 47.6],
        "basisOfRecord": ["HUMAN_OBSERVATION", "HUMAN_OBSERVATION",
                          "FOSSIL_SPECIMEN"],
    })
    clean = cd.clean_gbif(raw, boundary.iloc[0])
    assert list(clean.index) == [0]
    assert clean.crs.to_epsg() == 2927


def test_missing_and_seven_number_summary():
    """
    missing_summary reports only columns with missing values, with correct
    counts and percentages, and seven_number_summary matches values worked
    out by hand.
    """
    df = pd.DataFrame({"a": [1, 2, 3, 4, None], "b": [1, 1, 1, 1, 1],
                       "c": [None, None, "x", "y", "z"]})
    missing = cd.missing_summary(df)
    assert list(missing.index) == ["a", "c"]
    assert missing.loc["a", "n"] == 1 and missing.loc["a", "pct"] == 20
    assert missing.loc["c", "n"] == 2 and missing.loc["c", "pct"] == 40

    summary = cd.seven_number_summary(df, ["a"]).loc["a"]
    assert summary["count"] == 4
    assert close(summary["mean"], 2.5)
    assert close(summary["median"], 2.5)
    assert summary["min"] == 1 and summary["max"] == 4
    assert close(summary["q1"], 1.75) and close(summary["q3"], 3.25)


def test_presence_by_density():
    """
    Cells land in the right road-density bins, and the share and mean count
    in each bin match a hand count.
    """
    grid = pd.DataFrame({
        "road_density_km2": [0, 0, 0.2, 0.4, 3.5],
        "cougar_count": [0, 2, 1, 0, 0],
    })
    table = validity.presence_by_density(grid)
    assert table.loc["0 (no road)", "cells"] == 2
    assert close(table.loc["0 (no road)", "share"], 0.5)
    assert close(table.loc["0 (no road)", "mean_count"], 1)
    assert table.loc["0-0.5", "cells"] == 2
    assert close(table.loc["0-0.5", "share"], 0.5)
    assert table.loc["3+", "cells"] == 1
    assert table.loc["1-1.5", "cells"] == 0
    row = table.loc["0 (no road)"]
    assert row["ci_low"] < row["share"] < row["ci_high"]


def test_wilson_interval():
    """
    The 95% Wilson interval for 5 of 10 is (0.2366, 0.7634) and for 0 of 10
    its lower end is exactly 0.
    """
    low, high = validity.wilson_interval(5, 10)
    assert close(low, 0.2366, tol=1e-4) and close(high, 0.7634, tol=1e-4)
    low, high = validity.wilson_interval(0, 10)
    assert close(low, 0) and close(high, 0.2775, tol=1e-3)


def test_distance_assumptions():
    """
    distance_assumptions reports each group's size and median, and a
    perfectly symmetric sample has zero skew.
    """
    points = pd.DataFrame({
        "wui_group": ["WUI"] * 5 + ["Wildland"] * 4 + ["Other"],
        "dist_to_road_m": [1, 2, 3, 4, 5, 10, 20, 30, 1000, 7],
    })
    table = validity.distance_assumptions(points)
    assert list(table.index) == ["WUI", "Wildland"]
    assert table.loc["WUI", "n"] == 5
    assert table.loc["WUI", "median"] == 3
    assert close(table.loc["WUI", "skew"], 0)
    assert table.loc["Wildland", "skew"] > 1  # one long tail
    assert 0 <= validity.levene_p(points) <= 1


def test_real_data_invariants():
    """
    Checks on the real data: every kept GBIF record is inside Washington
    and is not a fossil, every WDFW incident projects to within 2 km of the
    state (the 1:20M outline cuts a few coastal points), Incident Type has
    exactly the three expected values plus 25 missing, and the WUI raster
    has no missing cells inside the state.
    """
    wa_geom = cd.load_wa_boundary().geometry.iloc[0]
    gbif = cd.clean_gbif(cd.load_gbif_raw(), wa_geom)
    assert len(gbif) > 0
    assert gbif.within(wa_geom).all()
    assert (gbif["basisOfRecord"] != "FOSSIL_SPECIMEN").all()

    incidents = cd.load_incidents()
    assert (incidents.distance(wa_geom) * FT_TO_M < 2000).all()
    assert incidents["lat"].between(45.5, 49.1).all()
    types = incidents["Incident Type"]
    assert set(types.dropna()) == {"Sighting",
                                   "Livestock or Pet Injury/Loss",
                                   "Human Safety Confrontation"}
    assert types.isna().sum() == 25

    classes = cd.load_wui_classes()
    check = cd.check_wui_complete(cd.WUI_TIF, classes)
    assert abs(check["area_km2"] / cd.WA_AREA_KM2 - 1) < 0.01


def main():
    """
    Runs every test and reports success.
    """
    test_wui_group()
    test_distance_to_codes()
    test_nearest_distance_m()
    test_make_grid_drops_slivers()
    test_road_density_and_counts()
    test_add_wui_columns()
    test_clean_gbif()
    test_missing_and_seven_number_summary()
    test_presence_by_density()
    test_wilson_interval()
    test_distance_assumptions()
    test_real_data_invariants()
    print("All tests passed")


if __name__ == "__main__":
    main()
