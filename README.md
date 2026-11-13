# Human-cougar conflict in Washington State

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place these files in `data/`:

- `Dangerous Incidents - Cougar.csv`: WDFW dangerous wildlife incidents
- `WSDOT_-_County_Road_(CRAB).geojson`: WSDOT county roads
- `wui_wa_tif/wui.tif`: Wildland-Urban Interface raster for Washington (with its sidecar files)

The other two download automatically on the first run if they are missing:

- `wa_boundary.geojson`: Washington outline (US Census)
- `gbif_cougar_wa.csv`: cougar occurrence records (GBIF API)

## Run

From this folder:

```bash
python run_eda.py
```

This prints the dataset summaries and assumption checks, and saves every
figure to `figs/`.

## Tests

```bash
python test_eda.py
```

Prints `All tests passed` on success. Some tests read the real files in
`data/`, so set up the data first.
