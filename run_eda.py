"""
Kian Giuliano
CSE 163 Final Project: Human-Cougar Conflict in Washington State

Runs the full exploratory data analysis: loads the four datasets, prints
the size, missing values, and summaries of each one, joins the cougar
records to the WUI raster, county roads, and a 5 km grid, prints the
Result Validity assumption checks, and saves each figure in the report
to figs/.
"""

import os

import cougar_data as cd
import figures
import plot_style as ps
import spatial
import validity

FIGS_DIR = os.path.join(cd.HERE, "figs")
PRECISE_M = 1000  # GBIF records located to within 1 km


def header(title):
    """
    Prints `title` as a section header.
    """
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def summarize_incidents(incidents):
    """
    Prints the size, missing values, duplicates, and variable summaries of
    the WDFW incident data.
    """
    header("WDFW dangerous incidents")
    source_columns = [c for c in incidents.columns
                      if c not in ("geometry", "lat", "long", "year")]
    print(f"{len(incidents):,} rows x {len(source_columns)} columns")
    print(cd.missing_summary(incidents[source_columns]).to_string())
    print("Fully duplicated rows:",
          incidents[source_columns].duplicated().sum())
    print(cd.seven_number_summary(incidents, ["year", "lat", "long"])
          .round(3).to_string())
    cd.print_categorical(incidents, ["Incident Type", "Confirmation Code",
                                     "WDFW Region", "County"])
    print("Date range:", incidents["Incident Date"].min().date(), "to",
          incidents["Incident Date"].max().date())


def summarize_gbif(raw, gbif):
    """
    Prints how many GBIF records stay after filtering, plus the size, missing
    values, and variable summaries of the filtered data.
    """
    header("GBIF occurrences")
    print(f"{len(raw):,} records in bounding box -> {len(gbif):,} inside WA")
    print(f"{len(gbif):,} rows x {len(raw.columns)} columns")
    print(cd.missing_summary(gbif).to_string())
    print(cd.seven_number_summary(
        gbif, ["year", "coordinateUncertaintyInMeters", "decimalLatitude",
               "decimalLongitude"]).round(2).to_string())
    cd.print_categorical(gbif, ["basisOfRecord", "institutionCode"])
    uncertainty = gbif["coordinateUncertaintyInMeters"]
    print(f"Uncertainty > 1 km: {(uncertainty > 1000).mean():.1%}, "
          f"> 5 km: {(uncertainty > 5000).mean():.1%}")
    print(f"Observed 2015 or later: {(gbif['year'] >= 2015).mean():.1%}")


def summarize_wui(classes):
    """
    Prints the WUI attribute table and the result of the raster
    completeness check.
    """
    header("WADNR Wildland-Urban Interface raster")
    print(classes.to_string(index=False))
    check = cd.check_wui_complete(cd.WUI_TIF, classes)
    rows, cols = check["shape"]
    print(f"{rows:,} x {cols:,} cells; NoData {check['n_nodata']:,} "
          f"({check['n_nodata'] / (rows * cols):.1%})")
    print(f"Classified area {check['area_km2']:,.0f} km^2 "
          f"({check['area_km2'] / cd.WA_AREA_KM2:.2%} of WA)")


def summarize_roads(roads, grid):
    """
    Prints the size, missing values, and summaries of the road segments and
    of road length and density per grid cell.
    """
    header("WSDOT county roads")
    source_columns = [c for c in roads.columns
                      if c not in ("geometry", "length_m")]
    print(f"{len(roads):,} rows x {len(source_columns)} columns")
    print(cd.missing_summary(roads).to_string())
    print("Missing or empty geometries:",
          (roads.geometry.isna() | roads.geometry.is_empty).sum())
    print(f"Total length: {roads['length_m'].sum() / 1000:,.0f} km")
    print(cd.seven_number_summary(roads, ["length_m"]).round(2).to_string())
    print(f"{len(grid):,} grid cells; "
          f"{(grid['road_km'] == 0).sum():,} with no county road")
    print(cd.seven_number_summary(
        grid, ["road_km", "road_density_km2", "cougar_count",
               "incident_count"]).round(3).to_string())


def summarize_joins(gbif, incidents, grid):
    """
    Prints the WUI groups and distance summaries of each point dataset, the
    precise-location sensitivity check, and the RQ3 class balance.
    """
    header("Joined data")
    for name, points in (("GBIF", gbif), ("WDFW", incidents)):
        print(f"-- {name} --")
        print(points["wui_group"].value_counts().to_string())
        print(cd.seven_number_summary(
            points, ["dist_to_road_m", "dist_to_wui_m"]).round(1).to_string())
        print(points.groupby("wui_group")["dist_to_road_m"].median()
              .round(1).to_string())

    precise = gbif[gbif["coordinateUncertaintyInMeters"] <= PRECISE_M]
    print(f"GBIF located within {PRECISE_M} m, median distance to road:")
    print(precise.groupby("wui_group")["dist_to_road_m"]
          .agg(["median", "size"]).round(0).to_string())
    present = grid["cougar_count"] > 0
    print(f"Cells with a GBIF record: {present.sum()} of {len(grid):,} "
          f"({present.mean():.1%})")


def summarize_validity(gbif, incidents, grid):
    """
    Prints the assumption checks for the RQ1 and RQ2 Result Validity tests
    and returns the presence-by-density table used in its figure.
    """
    header("Result Validity assumption checks")
    for name, points in (("GBIF", gbif), ("WDFW", incidents)):
        print(f"-- RQ1, {name} --")
        print(validity.distance_assumptions(points).to_string(
            float_format=lambda v: f"{v:.3g}"))
        print(f"Levene p = {validity.levene_p(points):.2e}")
    print("-- RQ2 --")
    print(f"Cells with zero records: {(grid['cougar_count'] == 0).mean():.1%}"
          f"; cells with zero road: {(grid['road_density_km2'] == 0).sum()}")
    table = validity.presence_by_density(grid)
    print(table.round(3).to_string())
    return table


def save_figures(data, presence):
    """
    Saves report figures to FIGS_DIR.
    """
    os.makedirs(FIGS_DIR, exist_ok=True)

    def out(name):
        """Returns the path in FIGS_DIR for the file `name`."""
        return os.path.join(FIGS_DIR, name)

    incidents, gbif = data["incidents"], data["gbif"]
    wa, grid = data["wa"], data["grid"]
    figures.plot_incidents_per_type(incidents,
                                    out("cougar_incidents_per_type.png"))
    figures.plot_incidents_by_county(incidents, out("incidents_by_county.png"))
    figures.plot_incident_map(incidents, data["roads"], wa,
                              out("incidents_map_by_type.png"))
    figures.plot_gbif_by_year(gbif, out("gbif_records_by_year.png"))
    figures.plot_gbif_basis(gbif, out("gbif_basis_of_record.png"))
    figures.plot_wui_classes(data["classes"], out("px_count_wui.png"))
    figures.plot_wui_hexbin(cd.WUI_TIF, out("wui_hexbin.png"))
    figures.plot_road_density_map(grid, wa, out("road_density_grid.png"))
    figures.plot_road_density_hist(grid, out("road_density_hist.png"))
    figures.plot_dist_to_road_by_wui(gbif, incidents,
                                     out("dist_to_road_by_wui.png"))
    figures.plot_presence_by_density(presence,
                                     out("road_density_vs_presence.png"))
    print(f"\nSaved 11 figures to {FIGS_DIR}")


def join_landscape(name, points, roads):
    """
    Returns a copy of `points` with the distance to the nearest county road
    and the WUI columns from spatial.add_wui_columns, and prints how many
    WUI points the coarse distance raster would have misclassified. `name`
    labels the printed line.
    """
    joined, mismatched = spatial.add_wui_columns(points, cd.WUI_TIF)
    joined["dist_to_road_m"] = spatial.nearest_distance_m(joined, roads)
    print(f"{name}: {mismatched} WUI points were outside WUI on the "
          "coarse distance raster (set to 0 m)")
    return joined


def main():
    """
    Loads, summarizes, joins, and plots each dataset in the EDA.
    """
    ps.apply()

    wa = cd.load_wa_boundary()
    wa_geom = wa.geometry.iloc[0]
    incidents = cd.load_incidents()
    gbif_raw = cd.load_gbif_raw()
    gbif = cd.clean_gbif(gbif_raw, wa_geom)
    roads = cd.load_roads()
    classes = cd.load_wui_classes()

    summarize_incidents(incidents)
    summarize_gbif(gbif_raw, gbif)
    summarize_wui(classes)

    gbif = join_landscape("GBIF", gbif, roads)
    incidents = join_landscape("WDFW", incidents, roads)

    grid = spatial.make_grid(wa_geom)
    grid = spatial.add_road_density(grid, roads)
    grid = spatial.add_point_counts(grid, gbif, "cougar_count")
    grid = spatial.add_point_counts(grid, incidents, "incident_count")

    summarize_roads(roads, grid)
    summarize_joins(gbif, incidents, grid)
    presence = summarize_validity(gbif, incidents, grid)

    data = {"wa": wa, "incidents": incidents, "gbif": gbif, "roads": roads,
            "classes": classes, "grid": grid}
    save_figures(data, presence)


if __name__ == "__main__":
    main()
