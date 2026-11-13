"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

Loads and cleans the four EDA datasets (WDFW dangerous incidents, GBIF
cougar occurrences, the WADNR Wildland-Urban Interface raster, and the
WSDOT county road network) and computes the per-dataset summaries used in
the EDA report.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests

CRS = 2927
FT_TO_M = 0.3048006096  # US survey foot in meters
WUI_CELL_M = 30  # side length of a WUI raster cell in meters
WA_AREA_KM2 = 184827  # total area of Washington State (land + water)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
WA_PATH = os.path.join(DATA_DIR, "wa_boundary.geojson")
INCIDENTS_PATH = os.path.join(DATA_DIR, "Dangerous Incidents - Cougar.csv")
GBIF_PATH = os.path.join(DATA_DIR, "gbif_cougar_wa.csv")
ROADS_PATH = os.path.join(DATA_DIR, "WSDOT_-_County_Road_(CRAB).geojson")
WUI_TIF = os.path.join(DATA_DIR, "wui_wa_tif", "wui.tif")

STATES_URL = ("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
              "cb_2023_us_state_20m.zip")
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_PAGE_SIZE = 300
GBIF_FIELDS = ["gbifID", "decimalLongitude", "decimalLatitude", "year",
               "month", "basisOfRecord", "coordinateUncertaintyInMeters",
               "datasetName", "institutionCode"]
WA_LAT_RANGE = "45.5,49.1"
WA_LON_RANGE = "-124.9,-116.9"


def load_wa_boundary(path=WA_PATH):
    """
    Returns Washington's outline as a one-row GeoDataFrame in EPSG:2927.
    Downloads the Census 1:20M state boundaries and caches Washington to
    `path` the first time it is called.
    """
    if not os.path.exists(path):
        states = gpd.read_file(STATES_URL)
        states[states["STUSPS"] == "WA"].to_file(path, driver="GeoJSON")
    return gpd.read_file(path).to_crs(CRS)


def load_incidents(path=INCIDENTS_PATH):
    """
    Returns the WDFW cougar incident reports as a GeoDataFrame in EPSG:2927.
    The x/y columns are Web Mercator (EPSG:3857). Adds a parsed
    `Incident Date`, its `year`, and WGS84 `lat`/`long` columns.
    """
    df = pd.read_csv(path)
    incidents = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:3857")
    lonlat = incidents.geometry.to_crs("EPSG:4326")
    incidents["lat"] = lonlat.y
    incidents["long"] = lonlat.x
    incidents["Incident Date"] = pd.to_datetime(incidents["Incident Date"],
                                                format="%m/%d/%Y")
    incidents["year"] = incidents["Incident Date"].dt.year
    return incidents.to_crs(CRS)


def fetch_gbif(path=GBIF_PATH):
    """
    Downloads every georeferenced Puma concolor record in Washington's
    bounding box from the GBIF occurrence API and saves the fields in
    GBIF_FIELDS to `path` as a CSV.
    """
    taxon_key = requests.get(GBIF_MATCH_URL,
                             params={"name": "Puma concolor"}).json()
    taxon_key = taxon_key["usageKey"]
    records = []
    offset = 0
    end_of_records = False
    while not end_of_records:
        page = requests.get(GBIF_SEARCH_URL, params={
            "taxonKey": taxon_key,
            "decimalLatitude": WA_LAT_RANGE,
            "decimalLongitude": WA_LON_RANGE,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "limit": GBIF_PAGE_SIZE,
            "offset": offset,
        }).json()
        records += page["results"]
        end_of_records = page["endOfRecords"]
        offset += GBIF_PAGE_SIZE
    pd.DataFrame(records).reindex(columns=GBIF_FIELDS).to_csv(path,
                                                              index=False)


def load_gbif_raw(path=GBIF_PATH):
    """
    Returns every GBIF record in Washington's bounding box as a DataFrame,
    downloading them with fetch_gbif first if `path` does not exist.
    """
    if not os.path.exists(path):
        fetch_gbif(path)
    return pd.read_csv(path)


def clean_gbif(raw, boundary_geom):
    """
    Takes the raw GBIF DataFrame and the Washington polygon (EPSG:2927) and
    returns a GeoDataFrame in EPSG:2927 without fossil specimens and with
    only the records that fall inside the polygon.
    """
    points = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy(raw["decimalLongitude"],
                                    raw["decimalLatitude"]),
        crs="EPSG:4326",
    ).to_crs(CRS)
    keep = ((points["basisOfRecord"] != "FOSSIL_SPECIMEN")
            & points.within(boundary_geom))
    return points[keep].copy()


def load_roads(path=ROADS_PATH):
    """
    Returns the WSDOT county road segments as a GeoDataFrame in EPSG:2927
    with each segment's length in meters (`length_m`).
    """
    roads = gpd.read_file(path).to_crs(CRS)
    roads["length_m"] = roads.length * FT_TO_M
    return roads


def load_wui_classes(tif=WUI_TIF):
    """
    Returns the WUI raster's attribute table (one row per class) as a
    DataFrame with integer VALUE and COUNT columns and the class name in
    WUI_DESC_N.
    """
    classes = pd.DataFrame(gpd.read_file(tif + ".vat.dbf",
                                         ignore_geometry=True))
    classes["VALUE"] = classes["VALUE"].astype(int)
    classes["COUNT"] = classes["COUNT"].astype(int)
    return classes[["VALUE", "WUI_DESC_N", "COUNT"]]


def check_wui_complete(tif, classes):
    """
    Reads every cell of the WUI raster and checks that each one is either a
    class in `classes` or NoData, and that the per-class cell counts match
    the attribute table. Returns a dict with the raster shape, the number
    of NoData cells, and the classified area in km^2.
    """
    with rasterio.open(tif) as src:
        cells = src.read(1)
        nodata = int(src.nodata)
    counts = np.bincount(cells.ravel(), minlength=256)
    present = set(np.nonzero(counts)[0])
    assert present <= set(classes["VALUE"]) | {nodata}
    assert (counts[classes["VALUE"]] == classes["COUNT"]).all()
    assert classes["COUNT"].sum() + counts[nodata] == cells.size
    return {
        "shape": cells.shape,
        "n_nodata": int(counts[nodata]),
        "area_km2": classes["COUNT"].sum() * WUI_CELL_M ** 2 / 1e6,
    }


def missing_summary(df):
    """
    Returns a DataFrame with the number and percentage of missing values
    for each column of `df` that has any, ignoring a geometry column.
    """
    missing = df.drop(columns="geometry", errors="ignore").isna().sum()
    missing = missing[missing > 0]
    return pd.DataFrame({"n": missing,
                         "pct": (missing / len(df) * 100).round(2)})


def seven_number_summary(df, columns):
    """
    Returns the count and seven-number summary (mean, standard deviation,
    minimum, first quartile, median, third quartile, maximum) of each
    column in `columns`, one row per column.
    """
    summary = df[columns].describe().T
    return summary.rename(columns={"25%": "q1", "50%": "median",
                                   "75%": "q3"})


def print_categorical(df, columns):
    """
    Prints every unique value of each column in `columns`, with its count,
    including missing values.
    """
    for column in columns:
        print(f"-- {column} --")
        print(df[column].value_counts(dropna=False).to_string())
