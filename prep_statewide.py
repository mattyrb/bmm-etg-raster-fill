#!/usr/bin/env python3
"""
prep_statewide.py
=================
One-time script: clip CONUS-wide DEM, BpS, and WTD rasters to the dissolved
NWI investigation boundary (with buffer) so per-basin prep can use fast
windowed reads instead of touching the full CONUS files every time.  All
outputs are reprojected into the NWI shapefile CRS so downstream prep
operates on a single consistent grid.

Also downloads the 3DEP 30-m DEM via py3dep if no CONUS DEM is provided.

HAND (Height Above Nearest Drainage) is *not* derived at the statewide
level — whitebox-tools' ElevationAboveStream runs out of memory on the
full Nevada DEM.  HAND is derived per-basin inside prep_basin.py, which
imports _derive_hand_from_dem() from this module.

Usage
-----
    python prep_statewide.py \
        --bps  /path/to/LF2020_BPS_CONUS.tif \
        --wtd  /path/to/wtd_conus.tif \
        [--dem /path/to/conus_dem.tif]   # optional; downloads 3DEP if omitted

Outputs are written to  <project>/statewide/

License: MIT
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import geopandas as gpd
    from shapely.ops import unary_union
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install rasterio geopandas shapely")


_here = Path(__file__).resolve().parent
PROJECT_DIR = _here
STATEWIDE_DIR = PROJECT_DIR / "statewide"
NWI_SHP = _here / "NWI_Investigations_EPSG_32611.shp"

# Buffer (metres) around the dissolved NWI boundary when clipping.
# Ensures edge basins have covariate data right up to and slightly beyond
# their boundary so reprojection doesn't produce NaN strips.
CLIP_BUFFER_M = 10_000  # 10 km

# HAND derivation defaults (whitebox-tools).  See _derive_hand_from_dem().
HAND_STREAM_THRESHOLD_CELLS = 1_000   # ≈0.9 km² at 30 m
HAND_BREACH_DIST_CELLS      = 50      # ≈1.5 km — keeps closed basins isolated


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _crs_equal(a, b) -> bool:
    """
    Compare two CRS-like objects (rasterio.crs.CRS, pyproj.CRS, EPSG int,
    WKT string, etc.) for equality.  rasterio.crs.CRS does not expose an
    ``.equals()`` method, so this helper normalises both sides to
    ``rasterio.crs.CRS`` and uses ``==``.
    """
    if a is None or b is None:
        return a is None and b is None
    try:
        ca = rasterio.crs.CRS.from_user_input(a)
        cb = rasterio.crs.CRS.from_user_input(b)
        return ca == cb
    except Exception:
        return str(a) == str(b)


def _clip_raster_to_geometry(
    src_path: Path,
    dst_path: Path,
    clip_geom,
    clip_crs,
    target_crs=None,
    resampling=Resampling.bilinear,
    label: str = "",
):
    """
    Clip a raster to a geometry and optionally reproject the clipped result
    into ``target_crs``.  The clip is done in the raster's native CRS to
    minimise resampling artefacts; the final output is then warped into the
    target CRS (by default the NWI shapefile CRS).  Writes a GeoTIFF with
    DEFLATE compression.
    """
    _log(f"  Clipping {label or src_path.name} …")

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to raster CRS for the mask step
        geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
        if not _crs_equal(geom_series.crs, src.crs):
            geom_series_src = geom_series.to_crs(src.crs)
        else:
            geom_series_src = geom_series
        geom = [geom_series_src.iloc[0].__geo_interface__]

        clipped_image, clipped_transform = rio_mask(
            src, geom, crop=True, all_touched=True, nodata=src.nodata,
        )
        src_crs = src.crs
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_count = src.count

    # If no target CRS or it matches the source, write the clipped raster as-is
    need_reproject = True
    if target_crs is None:
        need_reproject = False
    else:
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
        if _crs_equal(dst_crs, src_crs):
            need_reproject = False

    if not need_reproject:
        out_profile = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "count": src_count,
            "crs": src_crs,
            "transform": clipped_transform,
            "width": clipped_image.shape[2],
            "height": clipped_image.shape[1],
            "nodata": src_nodata,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(clipped_image)
    else:
        _log(f"    reprojecting {label or src_path.name} "
             f"from {src_crs.to_string()} → {dst_crs.to_string()}")

        src_height = clipped_image.shape[1]
        src_width = clipped_image.shape[2]
        left, bottom, right, top = rasterio.transform.array_bounds(
            src_height, src_width, clipped_transform
        )

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            left=left, bottom=bottom, right=right, top=top,
        )

        out_profile = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "count": src_count,
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": src_nodata,
            "compress": "DEFLATE",
            "predictor": 2,
        }

        with rasterio.open(dst_path, "w", **out_profile) as dst:
            for b in range(1, src_count + 1):
                reproject(
                    source=clipped_image[b - 1],
                    destination=rasterio.band(dst, b),
                    src_transform=clipped_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=src_nodata,
                    resampling=resampling,
                )

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")


def _download_3dep(clip_geom, clip_crs, dst_path: Path, target_crs=None):
    """
    Download USGS 3DEP 30-m DEM for the clip geometry using py3dep.
    Falls back to instructions if py3dep is not available.  If ``target_crs``
    is provided, the DEM is reprojected from EPSG:4326 into that CRS before
    being written.
    """
    _log("  Downloading 3DEP 30-m DEM via py3dep …")
    try:
        import py3dep
    except ImportError:
        sys.exit(
            "py3dep is not installed.  Install with:\n"
            "  pip install py3dep\n"
            "Or provide a CONUS DEM with --dem /path/to/dem.tif"
        )

    # py3dep needs a bounding box in EPSG:4326
    geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
    geom_4326 = geom_series.to_crs("EPSG:4326")
    bounds = geom_4326.total_bounds  # (minx, miny, maxx, maxy)
    _log(f"    bbox (EPSG:4326): {bounds}")

    # py3dep.get_dem returns an xarray DataArray in EPSG:4326
    dem = py3dep.get_dem(tuple(bounds), resolution=30, crs="EPSG:4326")

    # Build a rasterio-friendly source array + transform in EPSG:4326
    src_transform = rasterio.transform.from_bounds(
        *bounds, dem.shape[1], dem.shape[0]
    )
    src_crs = rasterio.crs.CRS.from_epsg(4326)
    src_arr = dem.values.astype(np.float32)

    # If we don't need to reproject, write directly
    if target_crs is None:
        dst_crs = src_crs
        need_reproject = False
    else:
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
        need_reproject = not _crs_equal(dst_crs, src_crs)

    if not need_reproject:
        _log(f"    Writing DEM to {dst_path.name} …")
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": src_arr.shape[1],
            "height": src_arr.shape[0],
            "count": 1,
            "crs": dst_crs,
            "transform": src_transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(src_arr, 1)
    else:
        _log(f"    Reprojecting DEM {src_crs.to_string()} → {dst_crs.to_string()}")
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs,
            src_arr.shape[1], src_arr.shape[0],
            left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3],
        )
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": dst_width,
            "height": dst_height,
            "count": 1,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=src_arr,
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")


def _derive_hand_from_dem(
    dem_path: Path,
    dst_path: Path,
    threshold_cells: int = HAND_STREAM_THRESHOLD_CELLS,
    breach_dist_cells: int = HAND_BREACH_DIST_CELLS,
    keep_intermediates: bool = False,
):
    """
    Derive Height Above Nearest Drainage (HAND) from a DEM using whitebox-tools.

    Pipeline:
        0. Fill DEM nodata with an elevation wall (max + 1000 m) so whitebox
           has a complete surface and flow routes toward the basin interior.
        1. FillDepressions (with flat_increment) — fast hydrologic
           conditioning that floods small artifact depressions and adds a
           tiny gradient across flat areas so D8 can route flow.
        2. D8 flow accumulation on the filled DEM.
        3. Extract a stream network at ``threshold_cells`` cells of upstream
           drainage area.
        4. ElevationAboveStream — for every cell, height in metres above the
           nearest downslope stream cell along the D8 flow path.
        5. Mask HAND back to NaN where the original DEM was nodata.

    The result is HAND in metres on the same grid / CRS as ``dem_path``.

    Parameters
    ----------
    dem_path : Path
        Source DEM (clipped basin DEM from prep_basin.py).
    dst_path : Path
        Output HAND GeoTIFF.
    threshold_cells : int
        Stream initiation threshold in cells.  Larger = sparser stream network
        = larger HAND values.  Default 1000 (~0.9 km^2 at 30 m).
    breach_dist_cells : int
        Retained for API compatibility but no longer used.  The pipeline now
        uses FillDepressions instead of BreachDepressionsLeastCost because
        the elevation wall makes the breach algorithm extremely slow.
    keep_intermediates : bool
        If True, leave the filled DEM, flow accumulation, and stream rasters
        in the working directory for inspection.  Default False.
    """
    try:
        import whitebox
    except ImportError:
        sys.exit(
            "whitebox is not installed.  Install with:\n"
            "  pip install whitebox\n"
            "Required for deriving HAND from the statewide DEM."
        )

    import shutil

    if not dem_path.exists():
        sys.exit(f"ERROR: DEM not found for HAND derivation: {dem_path}")

    _log(f"  Deriving HAND from {dem_path.name} "
         f"(stream threshold = {threshold_cells} cells) …")

    # Use a dedicated work directory with absolute paths everywhere.  Some
    # whitebox versions on Windows ignore set_working_dir for certain tools,
    # so we avoid relying on it.
    work_dir = (dst_path.parent / "_hand_work").resolve()
    work_dir.mkdir(exist_ok=True)

    wbt = whitebox.WhiteboxTools()
    # Verbose mode forwards the tool binary's stdout to our log — without it a
    # silent whitebox failure (non-zero exit inside the binary but rc still 0
    # from the Python wrapper) gives us no clues.
    wbt.set_verbose_mode(True)
    wbt.set_working_dir(str(work_dir))

    # Whitebox's stdout contains both progress ticks and any error messages.
    # We capture it so we can surface the real failure reason if a tool fails
    # or produces no output file.
    wbt_buf: list[str] = []

    def _wbt_cb(msg: str):
        line = msg.strip()
        if not line:
            return
        wbt_buf.append(line)
        # Whitebox emits many "*" progress ticks — skip those from the log.
        if line in ("*", "0%", "100%") or line.endswith("%"):
            return
        _log(f"      [wbt] {line}")

    def _run_tool(label: str, func, **kwargs) -> int:
        _log(f"    {label} …")
        wbt_buf.clear()
        try:
            rc_local = func(callback=_wbt_cb, **kwargs)
        except Exception as exc:
            sys.exit(f"ERROR: whitebox {label} raised: {exc}")
        if rc_local not in (0, None):
            tail = "\n".join(wbt_buf[-10:])
            sys.exit(
                f"ERROR: whitebox {label} failed (rc={rc_local}).\n"
                f"Last whitebox output:\n{tail}"
            )
        return rc_local if rc_local is not None else 0

    # Stage the DEM into the working directory as an absolute path.
    # CRITICAL: clipped DEMs have nodata outside the basin polygon (from
    # rasterio.mask).  Whitebox treats nodata cells as barriers, so D8
    # flow paths die at the polygon edge rather than accumulating into
    # interior channels.  The fix is to fill nodata with a high elevation
    # wall so flow routes inward toward the valley floor.  After HAND is
    # computed we mask back to NaN where the original DEM was nodata.
    dem_local   = (work_dir / "dem_in.tif").resolve()
    breached    = (work_dir / "dem_breached.tif").resolve()
    flow_accum  = (work_dir / "flow_accum.tif").resolve()
    streams     = (work_dir / "streams.tif").resolve()
    hand_temp   = (work_dir / "hand_out.tif").resolve()

    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1).astype(np.float32)
        dem_prof = src.profile.copy()
        src_nodata = src.nodata

    # Build a boolean mask of originally-nodata pixels.
    if src_nodata is not None:
        nodata_mask = (dem_arr == np.float32(src_nodata)) | ~np.isfinite(dem_arr)
    else:
        nodata_mask = ~np.isfinite(dem_arr)
    n_nodata = int(nodata_mask.sum())
    n_total  = dem_arr.size
    _log(f"    DEM has {n_nodata:,} nodata pixels ({100*n_nodata/n_total:.1f}%) "
         f"— filling with elevation wall for whitebox …")

    # Replace nodata with a wall: max valid elevation + 1000 m.  This
    # ensures flow routes toward the basin interior rather than toward
    # the clipped edges.
    valid_vals = dem_arr[~nodata_mask]
    if valid_vals.size == 0:
        _log("    WARNING: DEM has 0 valid pixels — cannot derive HAND")
        return
    wall_elev = float(np.nanmax(valid_vals)) + 1000.0
    dem_filled = dem_arr.copy()
    dem_filled[nodata_mask] = wall_elev

    # Write the filled DEM (no nodata — every cell has a valid value).
    dem_prof.update(dtype="float32", nodata=None, count=1)
    with rasterio.open(dem_local, "w", **dem_prof) as dst:
        dst.write(dem_filled, 1)

    # FillDepressions is used instead of BreachDepressionsLeastCost.
    # With the elevation wall around the clipped DEM, the breach algorithm
    # spends hours searching for least-cost paths through the wall.
    # FillDepressions is O(n), handles the wall gracefully, and produces
    # equivalent results for HAND derivation.  flat_increment ensures
    # that filled flat areas (playas) still have a tiny gradient so D8
    # can route flow across them.
    _run_tool(
        "1/4  FillDepressions",
        wbt.fill_depressions,
        dem=str(dem_local),
        output=str(breached),
        flat_increment=True,
    )

    _run_tool(
        "2/4  D8FlowAccumulation",
        wbt.d8_flow_accumulation,
        i=str(breached),
        output=str(flow_accum),
        out_type="cells",
    )

    _run_tool(
        f"3/4  ExtractStreams (threshold = {threshold_cells} cells)",
        wbt.extract_streams,
        flow_accum=str(flow_accum),
        output=str(streams),
        threshold=threshold_cells,
    )

    _run_tool(
        "4/4  ElevationAboveStream (HAND)",
        wbt.elevation_above_stream,
        dem=str(breached),
        streams=str(streams),
        output=str(hand_temp),
    )

    if not hand_temp.exists():
        # Something went wrong silently.  Dump what's in the work directory
        # plus the tail of whitebox output so the user can see what happened.
        _log(f"  Contents of {work_dir}:")
        try:
            for p in sorted(work_dir.iterdir()):
                sz = p.stat().st_size / 1e6
                _log(f"    {p.name:<32s} {sz:8.1f} MB")
        except OSError as exc:
            _log(f"    (failed to list work dir: {exc})")
        tail = "\n".join(wbt_buf[-20:]) if wbt_buf else "(no whitebox output captured)"
        sys.exit(
            f"ERROR: HAND output not produced by whitebox: {hand_temp}\n"
            f"Last whitebox output:\n{tail}"
        )

    # ── Normalise HAND: mask back original nodata, set NaN ───────────────────
    # The filled DEM has elevation-wall values where the original DEM was
    # nodata.  Whitebox computed HAND for those wall cells (huge positive
    # values that are meaningless).  We mask them back to NaN, along with
    # any whitebox internal nodata sentinel and extreme negatives.
    _log("    Normalising HAND to float32 / NaN-nodata …")
    with rasterio.open(hand_temp) as src:
        arr = src.read(1).astype(np.float32)
        src_nd = src.nodata
        prof = src.profile.copy()
    # 1) Mask where the original DEM had nodata (the elevation wall).
    arr[nodata_mask] = np.nan
    # 2) Mask whitebox's own nodata sentinel if it set one.
    if src_nd is not None:
        arr[arr == np.float32(src_nd)] = np.nan
    # 3) Mask any extreme negative values that whitebox sometimes writes.
    arr[arr < -1e4] = np.nan

    n_valid = int(np.isfinite(arr).sum())
    _log(f"    HAND valid pixels after masking: {n_valid:,} / {n_total:,}")

    prof.update(dtype="float32", nodata=float("nan"),
                compress="DEFLATE", predictor=2)
    if dst_path.exists():
        dst_path.unlink()
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(arr, 1)
    # Remove the raw whitebox output
    if hand_temp.exists():
        try:
            hand_temp.unlink()
        except OSError:
            pass

    # Clean up intermediates unless asked to keep them
    if not keep_intermediates:
        for fp in (breached, flow_accum, streams, dem_local):
            if fp.exists():
                try:
                    fp.unlink()
                except OSError:
                    pass
        try:
            work_dir.rmdir()
        except OSError:
            pass  # leave non-empty work dir alone

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")

    # Quick sanity check: report HAND value range
    try:
        with rasterio.open(dst_path) as ds:
            arr = ds.read(1, masked=True)
            if arr.count() > 0:
                _log(f"    HAND range (m): "
                     f"{float(arr.min()):.1f} – {float(arr.max()):.1f}  "
                     f"(median {float(np.ma.median(arr)):.1f})")
    except Exception as e:
        _log(f"    (could not summarise HAND raster: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Clip CONUS rasters to NWI investigation extent."
    )
    parser.add_argument("--bps", type=Path, required=True,
                        help="Path to CONUS BpS raster (LF2020)")
    parser.add_argument("--wtd", type=Path, required=True,
                        help="Path to CONUS WTD raster (Ma et al. 2025)")
    parser.add_argument("--dem", type=Path, default=None,
                        help="Path to CONUS DEM raster (optional; downloads "
                             "3DEP 30m if omitted)")
    parser.add_argument("--buffer-m", type=float, default=CLIP_BUFFER_M,
                        help=f"Buffer around NWI boundary (default: {CLIP_BUFFER_M} m)")
    # HAND is now derived per-basin in prep_basin.py (see _derive_hand_from_dem
    # below for the shared implementation).  The statewide DEM is too large
    # for whitebox-tools' ElevationAboveStream to process in one pass.
    args = parser.parse_args()

    t0 = time.time()
    STATEWIDE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and dissolve NWI boundary ──────────────────────────────────────
    _log("Loading NWI investigation boundaries …")
    if not NWI_SHP.exists():
        sys.exit(f"ERROR: NWI shapefile not found: {NWI_SHP}")

    gdf = gpd.read_file(NWI_SHP)
    _log(f"  {len(gdf)} basin polygons loaded")
    _log(f"  NWI shapefile CRS: {gdf.crs.to_string() if gdf.crs else 'UNKNOWN'}")

    # Dissolve into a single geometry and buffer
    dissolved = unary_union(gdf.geometry)
    # Buffer in the shapefile's native CRS (should be projected, metres)
    if gdf.crs is not None and gdf.crs.is_geographic:
        _log("  WARNING: NWI shapefile is in geographic CRS — "
             "reprojecting to EPSG:5070 for buffering")
        gdf_proj = gdf.to_crs("EPSG:5070")
        dissolved = unary_union(gdf_proj.geometry)
        clip_crs = gdf_proj.crs
    else:
        clip_crs = gdf.crs

    clip_geom = dissolved.buffer(args.buffer_m)
    _log(f"  Dissolved + {args.buffer_m:.0f} m buffer ready")

    # All statewide outputs will be reprojected into the NWI shapefile CRS
    # so downstream per-basin prep can use a single consistent grid.
    target_crs = gdf.crs
    _log(f"  Output CRS for statewide rasters: {target_crs.to_string()}")

    # ── Clip BpS ────────────────────────────────────────────────────────────
    _log("Clipping BpS …")
    if not args.bps.exists():
        sys.exit(f"ERROR: BpS raster not found: {args.bps}")
    _clip_raster_to_geometry(
        args.bps, STATEWIDE_DIR / "BpS_statewide.tif",
        clip_geom, clip_crs,
        target_crs=target_crs,
        resampling=Resampling.nearest,
        label="BpS",
    )

    # ── Clip WTD ────────────────────────────────────────────────────────────
    _log("Clipping WTD …")
    if not args.wtd.exists():
        sys.exit(f"ERROR: WTD raster not found: {args.wtd}")
    _clip_raster_to_geometry(
        args.wtd, STATEWIDE_DIR / "WTD_statewide.tif",
        clip_geom, clip_crs,
        target_crs=target_crs,
        resampling=Resampling.bilinear,
        label="WTD",
    )

    # ── DEM: clip from CONUS file or download 3DEP ──────────────────────────
    dem_dst = STATEWIDE_DIR / "DEM_statewide.tif"
    if args.dem is not None:
        _log("Clipping DEM …")
        if not args.dem.exists():
            sys.exit(f"ERROR: DEM raster not found: {args.dem}")
        _clip_raster_to_geometry(
            args.dem, dem_dst,
            clip_geom, clip_crs,
            target_crs=target_crs,
            resampling=Resampling.bilinear,
            label="DEM",
        )
    else:
        _download_3dep(clip_geom, clip_crs, dem_dst, target_crs=target_crs)

    # ── HAND note ───────────────────────────────────────────────────────────
    # HAND (Height Above Nearest Drainage) is no longer derived at the
    # statewide level.  whitebox-tools' ElevationAboveStream tried to
    # allocate ~25 GB on the full Nevada 30-m DEM and ran out of memory.
    # HAND is now derived per-basin inside prep_basin.py, where each basin
    # DEM easily fits in RAM.  The _derive_hand_from_dem() helper below is
    # still exported so prep_basin.py can import it.
    _log("HAND will be derived per-basin in prep_basin.py "
         "(statewide DEM is too large for ElevationAboveStream).")

    # ── Done ────────────────────────────────────────────────────────────────
    _log(f"\nStatewide rasters written to: {STATEWIDE_DIR.resolve()}")
    _log(f"  DEM_statewide.tif")
    _log(f"  BpS_statewide.tif")
    _log(f"  WTD_statewide.tif")
    _log(f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
