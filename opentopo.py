"""
OpenTopography DEM downloader.

Used by prep_basin.py (and prep_custom_basin.py) as a fallback when no
statewide DEM is available.  Downloads the Copernicus GLO-30 DEM (COP30)
for a basin's bounding box via the OpenTopography public API, then
reprojects + clips it onto the basin grid.

The per-basin AOI is small enough (tens of MB download) that this runs
in seconds, which is why we do it per-basin rather than trying to fetch
a Nevada-scale DEM via py3dep.

Default demtype is COP30.  See
    https://portal.opentopography.org/apidocs/#/Public/getGlobalDem
for the full list (SRTMGL1, SRTMGL3, AW3D30, NASADEM, COP30, COP90, ...).

API key:
    Primary:  $OPENTOPOGRAPHY_API_KEY env var
    Fallback: _DEFAULT_API_KEY below (project key, replace as needed).

Requires: ``requests`` (already a transitive dependency in this project
via py3dep).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject

# ---------------------------------------------------------------------------
# Fallback API key.  Prefer setting $OPENTOPOGRAPHY_API_KEY in your
# environment instead of editing this file -- but for a single-operator
# setup, dropping the key here is fine.  The key below is Matt's personal
# OpenTopography key (see conversation 2026-04-14).
# ---------------------------------------------------------------------------
_DEFAULT_API_KEY = "9e0b3bce799a1beaf2cd823c09e96c68"

_API_URL = "https://portal.opentopography.org/API/globaldem"

# Extra bbox buffer (degrees) to ensure we cover the clip geometry's
# reprojection footprint after basin-grid warping.  ~0.01 deg ≈ 1 km.
_BBOX_PAD_DEG = 0.02


def _log(msg: str):
    print(msg, flush=True)


def get_api_key() -> str:
    """Return the OpenTopography API key from env or the project default."""
    return os.environ.get("OPENTOPOGRAPHY_API_KEY", _DEFAULT_API_KEY)


def _bbox_4326(clip_geom, clip_crs, pad_deg: float = _BBOX_PAD_DEG):
    """Project clip_geom to EPSG:4326 and return (south, north, west, east)
    with a small degree pad."""
    gs = gpd.GeoSeries([clip_geom], crs=clip_crs)
    if clip_crs is None or not gs.crs.equals("EPSG:4326"):
        gs = gs.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gs.total_bounds
    return (miny - pad_deg, maxy + pad_deg,
            minx - pad_deg, maxx + pad_deg)


def download_dem(
    clip_geom,
    clip_crs,
    dst_path: Path,
    target_crs,
    *,
    demtype: str = "COP30",
    buffer_m: float = 0.0,
    api_key: str | None = None,
    resampling=Resampling.bilinear,
    raw_path: Path | None = None,
) -> bool:
    """
    Download a DEM for ``clip_geom`` from OpenTopography and write it at
    ``dst_path`` reprojected + clipped onto the basin grid in ``target_crs``.

    Parameters
    ----------
    clip_geom : shapely geometry
        Basin clip polygon (already buffered by the caller, if desired).
    clip_crs : CRS
        CRS of ``clip_geom``.
    dst_path : Path
        Output GeoTIFF (e.g. ``basins/<key>/input/DEM.tif``).
    target_crs : CRS
        CRS for the final raster (typically the NWI shapefile CRS).
    demtype : str
        OpenTopography DEM type.  Default ``"COP30"`` (Copernicus GLO-30).
    buffer_m : float
        Extra buffer (meters) added to the clip geometry before writing
        the final raster.  The download bbox is always a geographic
        envelope with ``_BBOX_PAD_DEG`` of pad regardless.
    api_key : str or None
        OpenTopography API key.  If ``None``, falls back to
        ``$OPENTOPOGRAPHY_API_KEY`` or the hardcoded project key.
    raw_path : Path or None
        Where to cache the raw EPSG:4326 download.  If ``None``, defaults
        to ``dst_path.with_name(dst_path.stem + "_raw.tif")``.

    Returns True on success.  Exits on network / API errors so the failure
    surfaces clearly in pipeline logs.
    """
    try:
        import requests
    except ImportError:
        sys.exit(
            "ERROR: the `requests` package is required for OpenTopography "
            "downloads.  Install it with `pip install requests`."
        )

    key = api_key or get_api_key()
    if not key:
        sys.exit(
            "ERROR: no OpenTopography API key found.  Set "
            "$OPENTOPOGRAPHY_API_KEY or edit opentopo._DEFAULT_API_KEY."
        )

    south, north, west, east = _bbox_4326(clip_geom, clip_crs)
    _log(f"    OpenTopography {demtype} bbox (EPSG:4326): "
         f"S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")

    params = {
        "demtype":      demtype,
        "south":        f"{south}",
        "north":        f"{north}",
        "west":         f"{west}",
        "east":         f"{east}",
        "outputFormat": "GTiff",
        "API_Key":      key,
    }

    if raw_path is None:
        raw_path = dst_path.with_name(dst_path.stem + "_raw.tif")

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"    Requesting {demtype} from OpenTopography …")
    t0 = time.time()
    try:
        resp = requests.get(_API_URL, params=params, stream=True, timeout=300)
    except requests.RequestException as e:
        sys.exit(f"ERROR: OpenTopography request failed: {e}")

    if resp.status_code != 200:
        snippet = resp.text[:500] if resp.text else "(no body)"
        sys.exit(
            f"ERROR: OpenTopography returned HTTP {resp.status_code}.\n"
            f"  URL: {resp.url}\n"
            f"  Body: {snippet}"
        )
    # Basic content-type sanity check: API returns text/xml on errors.
    ctype = resp.headers.get("Content-Type", "")
    if "tiff" not in ctype.lower() and "octet-stream" not in ctype.lower():
        snippet = resp.text[:500] if resp.text else "(no body)"
        sys.exit(
            f"ERROR: OpenTopography did not return a GeoTIFF "
            f"(Content-Type={ctype!r}).\n  Body: {snippet}"
        )

    bytes_written = 0
    with open(raw_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    dt = time.time() - t0
    _log(f"    Downloaded {bytes_written / 1e6:.1f} MB → {raw_path.name} "
         f"({dt:.1f} s)")

    if bytes_written < 1024:
        sys.exit(
            f"ERROR: OpenTopography response was only {bytes_written} bytes "
            f"— likely an API error.  See {raw_path}."
        )

    # ── Reproject + clip the raw download onto the basin grid ───────────────
    _log(f"    Reprojecting & clipping {demtype} → {dst_path.name} …")

    with rasterio.open(raw_path) as src:
        src_crs = src.crs
        src_dtype = src.dtypes[0]
        src_nodata = src.nodata
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)

        # Compute target transform + grid in basin CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            *src.bounds,
        )

        warped = np.full((dst_height, dst_width),
                         fill_value=src_nodata if src_nodata is not None else np.nan,
                         dtype=src_dtype)

        reproject(
            source=rasterio.band(src, 1),
            destination=warped,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=src_nodata,
            resampling=resampling,
        )

    # Write the reprojected raster to a temp path, then mask-clip to the
    # basin polygon (so pixels outside the buffered basin are nodata).
    tmp_path = dst_path.with_name(dst_path.stem + "_warped.tif")
    profile = {
        "driver":    "GTiff",
        "dtype":     src_dtype,
        "count":     1,
        "crs":       dst_crs,
        "transform": dst_transform,
        "width":     dst_width,
        "height":    dst_height,
        "nodata":    src_nodata,
        "compress":  "DEFLATE",
        "predictor": 2,
    }
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(warped, 1)

    # Clip to the basin polygon in target CRS
    gs = gpd.GeoSeries([clip_geom], crs=clip_crs).to_crs(dst_crs)
    clip_poly = gs.iloc[0]
    if buffer_m and buffer_m > 0:
        clip_poly = clip_poly.buffer(buffer_m)

    with rasterio.open(tmp_path) as src:
        out_image, out_transform = rio_mask(
            src, [clip_poly.__geo_interface__],
            crop=True, all_touched=True, nodata=src.nodata,
        )
        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="DEFLATE",
            predictor=2,
        )

    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(out_image)

    # Clean up the intermediate warped file; keep raw_path as a cache.
    try:
        tmp_path.unlink()
    except OSError:
        pass

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB, {demtype} via OpenTopography)")
    return True
