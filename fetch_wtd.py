#!/usr/bin/env python3
"""
fetch_wtd.py
============
Download the Ma et al. (2025) water table depth raster from HydroFrame
for a given study-area bounding box and save it as a GeoTIFF.

Prerequisites
-------------
1.  pip install hf-hydrodata
2.  Create a free account at https://hydrogen.princeton.edu/
3.  Request a 4-digit API PIN (valid ~48 h).
4.  Register once per machine:
        import hf_hydrodata as hf
        hf.register_api_pin("you@example.com", "1234")

Usage
-----
    # From the project root:
    python fetch_wtd.py

    # Or with a custom bounding box (lat_min, lon_min, lat_max, lon_max):
    python fetch_wtd.py --bbox 39.55 -120.30 39.95 -119.90

    # Save to a custom path:
    python fetch_wtd.py --output /path/to/wtd.tif

The output GeoTIFF is in the native CRS of the Ma 2025 dataset (EPSG:5070,
NAD83 / Conus Albers).  The ETg baseline-fill workflow reprojects it
automatically to match the ETg grid — just point wtd_tif in the basin's
config.toml at the output file.

Reference
---------
Ma, L., Condon, L. E., Behrens, D., & Maxwell, R. M. (2025).
  High resolution water table depth estimates across the contiguous
  United States. *Water Resources Research*, 61(2), e2024WR038658.

License: MIT (see LICENSE)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import hf_hydrodata as hf
except ImportError:
    sys.exit(
        "hf_hydrodata is not installed.\n"
        "  pip install hf-hydrodata\n"
        "Then create an account at https://hydrogen.princeton.edu/ and\n"
        "register your PIN:\n"
        "  import hf_hydrodata as hf\n"
        '  hf.register_api_pin("you@example.com", "1234")'
    )

# ── Default bounding box: Sierra Valley, CA ─────────────────────────────────
# (lat_min, lon_min, lat_max, lon_max)
DEFAULT_BBOX = (39.55, -120.30, 39.95, -119.90)

# Default output location (project data directory)
_here = Path(__file__).resolve().parent
DEFAULT_OUTPUT = _here / "wtd_Ma2025_SierraValley.tif"


def fetch_wtd(
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    output: Path | str = DEFAULT_OUTPUT,
    variable: str = "wtd",
) -> Path:
    """
    Download the Ma et al. (2025) WTD raster for *bbox* and write a GeoTIFF.

    Parameters
    ----------
    bbox : (lat_min, lon_min, lat_max, lon_max)
        Geographic bounding box in decimal degrees (WGS 84).
    output : path-like
        Destination GeoTIFF path.
    variable : str
        HydroFrame variable name.  Options include:
          "wtd"  – mean water table depth estimate (m below surface)
          Other variables may be available; check the HydroFrame catalog.

    Returns
    -------
    Path to the written GeoTIFF.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lat_min, lon_min, lat_max, lon_max = bbox
    print(f"Fetching Ma 2025 '{variable}' for bbox: {bbox}")
    print(f"  Output: {output}")

    # ── Method 1: get_gridded_files (saves directly to GeoTIFF) ─────────
    # This is the cleanest approach — hf_hydrodata writes the file for us
    # with correct georeferencing.
    try:
        hf.get_gridded_files(
            options={
                "dataset": "Ma_2025",
                "variable": variable,
                "latlng_bounds": [lat_min, lon_min, lat_max, lon_max],
            },
            filename_template=str(output),
        )
        print(f"  ✓ Saved via get_gridded_files: {output}")
        return output

    except Exception as e:
        print(f"  get_gridded_files failed ({e}), falling back to numpy + rasterio …")

    # ── Method 2: Fallback — get numpy array and write with rasterio ────
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        sys.exit("Fallback requires rasterio.  pip install rasterio")

    data = hf.get_gridded_data(
        options={
            "dataset": "Ma_2025",
            "variable": variable,
            "latlng_bounds": [lat_min, lon_min, lat_max, lon_max],
        }
    )

    if data is None or data.size == 0:
        sys.exit("No data returned — check your bounding box and API credentials.")

    # Squeeze extra dimensions (time, etc.) if present
    while data.ndim > 2:
        data = data[0]

    print(f"  Array shape: {data.shape}  dtype: {data.dtype}")
    print(f"  Range: {np.nanmin(data):.2f} – {np.nanmax(data):.2f}")

    nrows, ncols = data.shape
    # The raw numpy array is in geographic coords (EPSG:4326) when using
    # latlng_bounds.  Build an Affine transform from the bounding box.
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": ncols,
        "height": nrows,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 2,
    }

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

    print(f"  ✓ Saved via rasterio fallback: {output}")
    print(
        "  NOTE: This file is in EPSG:4326 (lat/lon). The workflow will\n"
        "  reproject it to match the ETg grid automatically."
    )
    return output


def register_pin():
    """Interactive helper to register a HydroFrame API PIN."""
    print("─── HydroFrame API Registration ───")
    print("1. Create an account at https://hydrogen.princeton.edu/")
    print("2. Request a 4-digit PIN (valid ~48 hours)")
    print()
    email = input("Email: ").strip()
    pin = input("PIN:   ").strip()
    if email and pin:
        hf.register_api_pin(email, pin)
        print("✓ PIN registered successfully.")
    else:
        print("Aborted — email and PIN are both required.")


def main():
    parser = argparse.ArgumentParser(
        description="Download Ma et al. (2025) WTD raster from HydroFrame."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        default=list(DEFAULT_BBOX),
        help="Bounding box in decimal degrees (default: Sierra Valley, CA)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output GeoTIFF path (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--variable", "-v",
        default="wtd",
        help='HydroFrame variable name (default: "wtd")',
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register your HydroFrame API PIN interactively, then exit.",
    )
    args = parser.parse_args()

    if args.register:
        register_pin()
        return

    fetch_wtd(
        bbox=tuple(args.bbox),
        output=args.output,
        variable=args.variable,
    )


if __name__ == "__main__":
    main()
