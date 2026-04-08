"""
Feature engineering for Phoenix land use classification.

Loads the best Sentinel-2 L2A scene, clips to the Phoenix study area,
computes 29 spectral + terrain + texture features, and writes a single
multi-band GeoTIFF ready for model training.

Inputs
------
  data/raw/sentinel2/   Downloaded S2 scene folders (B02-B12 GeoTIFFs)
  data/raw/dem/         Copernicus GLO-30 DEM tiles

Outputs
-------
  outputs/feature_stack.tif       29-band feature stack (EPSG:32612, 10 m)
  outputs/phoenix_false_color.png NIR/Red/Green false-colour composite

Feature bands (29 total)
------------------------
  1  B02  Blue        (S2 raw, 10 m)
  2  B03  Green       (S2 raw, 10 m)
  3  B04  Red         (S2 raw, 10 m)
  4  B08  NIR         (S2 raw, 10 m)
  5  B11  SWIR1       (S2 raw, resampled to 10 m)
  6  B12  SWIR2       (S2 raw, resampled to 10 m)
  7  NDVI  Vegetation index
  8  NDBI  Built-up index
  9  NDWI  Water index
 10  SAVI  Soil-adjusted vegetation index
 11  BSI   Bare soil index
 12  EVI   Enhanced vegetation index
 13  MNDWI Modified water index
 14  UI    Urban index
 15  NBI   New built-up index
 16  RedGreen  Red / Green ratio
 17  SR        Simple Ratio (NIR / Red)
 18  SWIRRatio SWIR1 / NIR
 19  Elevation  DEM (m)
 20  Slope      (degrees)
 21  Aspect_sin (terrain direction — sine component)
 22  Aspect_cos (terrain direction — cosine component)
 23  PlanCurv   Plan curvature
 24  ProfCurv   Profile curvature
 25  TRI        Terrain Ruggedness Index
 26  TPI        Topographic Position Index
 27  NDVI_std   NDVI local std (7x7 px)
 28  NIR_mean   NIR local mean (7x7 px)
 29  NIR_std    NIR local std  (7x7 px)

Usage
-----
    conda run -n arcgis-dl python src/feature_engineering.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message=".*TIFFReadDirectory.*")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\Projects\land-use-classification")
S2_DIR      = PROJECT_DIR / "data" / "raw" / "sentinel2"
DEM_DIR     = PROJECT_DIR / "data" / "raw" / "dem"
OUT_DIR     = PROJECT_DIR / "outputs"
OUT_STACK   = OUT_DIR / "feature_stack.tif"
OUT_VIS     = OUT_DIR / "phoenix_false_color.png"

# Phoenix study area (WGS84)
AOI_WGS84 = (-112.5, 33.2, -111.7, 33.7)

# Band filenames and their native S2 resolution (metres)
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B11", "B12"]

# Texture window size (pixels at 10 m → 70 m neighbourhood)
TEX_WIN = 7

# Band names for the output stack (must match 29 total)
BAND_NAMES = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDBI", "NDWI", "SAVI", "BSI",
    "EVI", "MNDWI", "UI", "NBI",
    "RedGreen", "SR", "SWIRRatio",
    "Elevation", "Slope", "Aspect_sin", "Aspect_cos",
    "PlanCurv", "ProfCurv", "TRI", "TPI",
    "NDVI_std", "NIR_mean", "NIR_std",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _divider(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _safe_divide(num: np.ndarray, den: np.ndarray,
                 fill: float = 0.0) -> np.ndarray:
    """Element-wise division; fills with `fill` where denominator is ~0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(den) > 1e-6, num / den, fill)
    return result.astype(np.float32)


def _local_stats(arr: np.ndarray, window: int):
    """Return (local_mean, local_std) computed with a uniform sliding window."""
    mean   = uniform_filter(arr.astype(np.float64), size=window)
    mean_sq = uniform_filter(arr.astype(np.float64) ** 2, size=window)
    var    = np.maximum(mean_sq - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1 — Choose best scene
# ---------------------------------------------------------------------------
def find_best_scene() -> Path:
    _divider("=")
    print("  STEP 1 - Select best Sentinel-2 scene")
    _divider()

    scenes = [d for d in S2_DIR.iterdir() if d.is_dir()]
    if not scenes:
        print("  ERROR: no scenes found in", S2_DIR)
        sys.exit(1)

    # Pick scene that has all required bands; prefer lowest cloud cover
    def cloud_pct(scene_dir: Path) -> float:
        meta = scene_dir / "metadata.txt"
        if not meta.exists():
            return 100.0
        for line in meta.read_text().splitlines():
            if "cloud_cover" in line:
                try:
                    return float(line.split(":")[-1].strip().rstrip("%"))
                except ValueError:
                    pass
        return 100.0

    def complete(scene_dir: Path) -> bool:
        return all((scene_dir / f"{b}.tif").exists()
                   for b in BANDS_10M + BANDS_20M)

    valid = [s for s in scenes if complete(s)]
    if not valid:
        print("  ERROR: no complete scenes (missing bands)")
        sys.exit(1)

    best = min(valid, key=cloud_pct)
    print(f"  Scenes available : {len(scenes)}")
    print(f"  Complete scenes  : {len(valid)}")
    print(f"  Selected         : {best.name}  (cloud={cloud_pct(best):.2f}%)")
    return best


# ---------------------------------------------------------------------------
# Step 2 — Clip AOI to scene intersection & build reference grid
# ---------------------------------------------------------------------------
def build_reference_grid(scene_dir: Path) -> tuple:
    """
    Transform AOI to scene CRS, intersect with scene bounds, return
    (ref_transform, ref_crs, clip_shape, utm_bounds) for the reference 10m grid.
    """
    _divider("=")
    print("  STEP 2 - Clip AOI to scene extent")
    _divider()

    ref_path = scene_dir / "B04.tif"
    with rasterio.open(ref_path) as src:
        scene_crs    = src.crs
        scene_bounds = src.bounds   # (left, bottom, right, top) in UTM

    # Transform AOI corners from WGS84 to scene UTM CRS
    aoi_utm = transform_bounds("EPSG:4326", scene_crs, *AOI_WGS84)

    # Clip AOI to scene bounds (intersection)
    left   = max(aoi_utm[0], scene_bounds.left)
    bottom = max(aoi_utm[1], scene_bounds.bottom)
    right  = min(aoi_utm[2], scene_bounds.right)
    top    = min(aoi_utm[3], scene_bounds.top)

    if left >= right or bottom >= top:
        print("  ERROR: AOI does not intersect scene extent.")
        sys.exit(1)

    clip_bounds = (left, bottom, right, top)
    coverage_pct = (
        (right - left) * (top - bottom)
        / ((aoi_utm[2] - aoi_utm[0]) * (aoi_utm[3] - aoi_utm[1]))
        * 100
    )

    # Build reference window and transform from B04 (10 m native)
    with rasterio.open(ref_path) as src:
        win = window_from_bounds(*clip_bounds, src.transform).round_lengths().round_offsets()
        ref_transform = src.window_transform(win)
        ref_shape     = (int(win.height), int(win.width))

    print(f"  AOI (WGS84)      : {AOI_WGS84}")
    print(f"  AOI (UTM)        : ({aoi_utm[0]:.0f}, {aoi_utm[1]:.0f}, "
          f"{aoi_utm[2]:.0f}, {aoi_utm[3]:.0f})")
    print(f"  Clip (UTM)       : ({left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f})")
    print(f"  AOI coverage     : {coverage_pct:.1f}%  "
          f"({'full' if coverage_pct >= 99 else 'partial — western tiles not downloaded'})")
    print(f"  Reference grid   : {ref_shape[1]} x {ref_shape[0]} px  @ 10 m  "
          f"({ref_shape[1]*10/1000:.1f} x {ref_shape[0]*10/1000:.1f} km)")

    return ref_transform, scene_crs, ref_shape, clip_bounds


# ---------------------------------------------------------------------------
# Step 3 — Load S2 bands
# ---------------------------------------------------------------------------
def load_s2_bands(scene_dir: Path,
                  clip_bounds: tuple,
                  ref_transform,
                  ref_crs,
                  ref_shape: tuple) -> dict:
    """
    Load and return all S2 bands as float32 arrays on the 10 m reference grid.
    20 m bands are resampled via bilinear interpolation.
    """
    _divider("=")
    print("  STEP 3 - Load Sentinel-2 bands")
    _divider()

    arrays = {}
    ref_h, ref_w = ref_shape

    for band in BANDS_10M + BANDS_20M:
        path = scene_dir / f"{band}.tif"
        with rasterio.open(path) as src:
            win = window_from_bounds(*clip_bounds, src.transform).round_lengths().round_offsets()
            data = src.read(
                1,
                window=win,
                out_shape=(ref_h, ref_w),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        # Mask no-data (S2 L2A uses 0 for fill)
        data = np.where(data == 0, np.nan, data)

        # Convert DN to surface reflectance (÷ 10000)
        data = data / 10_000.0

        arrays[band] = data
        native_res = 10 if band in BANDS_10M else 20
        print(f"  {band}  ({native_res} m -> 10 m)  "
              f"min={np.nanmin(data):.4f}  max={np.nanmax(data):.4f}  "
              f"nan%={np.isnan(data).mean()*100:.1f}")

    return arrays


# ---------------------------------------------------------------------------
# Step 4 — Load & reproject DEM
# ---------------------------------------------------------------------------
def load_dem(clip_bounds: tuple, ref_transform, ref_crs, ref_shape: tuple) -> np.ndarray:
    """
    Merge Copernicus GLO-30 tiles, reproject to scene CRS, resample to 10 m.
    """
    _divider("=")
    print("  STEP 4 - Load DEM (Copernicus GLO-30)")
    _divider()

    dem_files = sorted(DEM_DIR.glob("*.tif"))
    if not dem_files:
        print("  WARNING: no DEM tiles found — elevation/terrain features will be zero.")
        return np.zeros(ref_shape, dtype=np.float32)

    print(f"  DEM tiles        : {len(dem_files)}")
    for f in dem_files:
        print(f"    {f.name}")

    # Merge tiles (they share CRS = EPSG:4326)
    handles = [rasterio.open(f) for f in dem_files]
    merged, merged_transform = rio_merge(handles)
    merged = merged[0].astype(np.float32)   # shape (H, W)
    merged_crs = handles[0].crs
    for h in handles:
        h.close()

    print(f"  Merged shape     : {merged.shape}  CRS={merged_crs}")

    # Reproject merged DEM to scene CRS at 10 m resolution
    ref_h, ref_w = ref_shape
    dem_utm = np.empty((ref_h, ref_w), dtype=np.float32)

    reproject(
        source      = merged,
        destination = dem_utm,
        src_transform  = merged_transform,
        src_crs        = merged_crs,
        dst_transform  = ref_transform,
        dst_crs        = ref_crs,
        dst_nodata     = np.nan,
        resampling     = Resampling.bilinear,
    )

    valid = ~np.isnan(dem_utm)
    print(f"  Reprojected      : {ref_w} x {ref_h} px  @ 10 m")
    print(f"  Elevation range  : {np.nanmin(dem_utm):.1f} – {np.nanmax(dem_utm):.1f} m")

    return dem_utm


# ---------------------------------------------------------------------------
# Step 5 — Spectral indices
# ---------------------------------------------------------------------------
def compute_spectral_indices(b: dict) -> dict:
    """Compute 12 spectral indices from reflectance bands."""
    _divider("=")
    print("  STEP 5 - Spectral indices")
    _divider()

    idx: dict = {}

    # Core indices
    idx["NDVI"]  = _safe_divide(b["B08"] - b["B04"], b["B08"] + b["B04"])
    idx["NDBI"]  = _safe_divide(b["B11"] - b["B08"], b["B11"] + b["B08"])
    idx["NDWI"]  = _safe_divide(b["B03"] - b["B08"], b["B03"] + b["B08"])
    idx["SAVI"]  = _safe_divide((b["B08"] - b["B04"]) * 1.5,
                                 b["B08"] + b["B04"] + 0.5)
    idx["BSI"]   = _safe_divide((b["B11"] + b["B04"]) - (b["B08"] + b["B02"]),
                                (b["B11"] + b["B04"]) + (b["B08"] + b["B02"]))

    # Advanced indices
    denom_evi    = b["B08"] + 6.0*b["B04"] - 7.5*b["B02"] + 1.0
    idx["EVI"]   = _safe_divide(2.5 * (b["B08"] - b["B04"]), denom_evi)
    idx["MNDWI"] = _safe_divide(b["B03"] - b["B11"], b["B03"] + b["B11"])
    idx["UI"]    = _safe_divide(b["B12"] - b["B08"], b["B12"] + b["B08"])
    with np.errstate(invalid="ignore", divide="ignore"):
        nbi = np.where(b["B08"] > 1e-6,
                       b["B04"] * b["B11"] / b["B08"], 0.0)
    idx["NBI"]   = nbi.astype(np.float32)

    # Band ratios
    idx["RedGreen"]  = _safe_divide(b["B04"], b["B03"])
    idx["SR"]        = _safe_divide(b["B08"], b["B04"])   # Simple Ratio
    idx["SWIRRatio"] = _safe_divide(b["B11"], b["B08"])

    for name, arr in idx.items():
        print(f"  {name:<12}  min={np.nanmin(arr):8.4f}  "
              f"max={np.nanmax(arr):8.4f}  mean={np.nanmean(arr):8.4f}")

    return idx


# ---------------------------------------------------------------------------
# Step 6 — Terrain features
# ---------------------------------------------------------------------------
def compute_terrain_features(dem: np.ndarray, cellsize: float = 10.0) -> dict:
    """
    Derive slope, aspect, curvature, TRI, TPI from the DEM using numpy gradients.
    All angles in degrees; curvatures in 1/m.
    """
    _divider("=")
    print("  STEP 6 - Terrain features  (cellsize = {:.0f} m)".format(cellsize))
    _divider()

    # Fill NaN with interpolated values so gradients work across gaps
    dem_filled = dem.copy()
    nan_mask   = np.isnan(dem_filled)
    if nan_mask.any():
        # Simple fill: replace NaN with nearest-neighbour mean of non-NaN median
        fill_val = float(np.nanmedian(dem_filled))
        dem_filled[nan_mask] = fill_val

    # First-order derivatives  (axis-0 = rows = North-South)
    dz_dy, dz_dx = np.gradient(dem_filled, cellsize, cellsize)

    # ── Slope ──────────────────────────────────────────────────────────────
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope     = np.degrees(slope_rad).astype(np.float32)

    # ── Aspect (decomposed to avoid 0°/360° wrap) ─────────────────────────
    aspect_rad = np.arctan2(-dz_dx, dz_dy)   # North = 0°, clockwise
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    # ── Curvatures (second derivatives) ───────────────────────────────────
    d2z_dy2  = np.gradient(dz_dy, cellsize, axis=0)
    d2z_dx2  = np.gradient(dz_dx, cellsize, axis=1)
    d2z_dxdy = np.gradient(dz_dy, cellsize, axis=1)

    p = dz_dx**2 + dz_dy**2
    p_safe = np.where(p > 1e-8, p, 1e-8)

    plan_curv = -(
        d2z_dx2 * dz_dy**2
        - 2 * d2z_dxdy * dz_dx * dz_dy
        + d2z_dy2 * dz_dx**2
    ) / (p_safe * np.sqrt(1 + p_safe))

    prof_curv = -(
        d2z_dx2 * dz_dx**2
        + 2 * d2z_dxdy * dz_dx * dz_dy
        + d2z_dy2 * dz_dy**2
    ) / (p_safe * (1 + p_safe)**1.5)

    plan_curv = np.where(p > 1e-8, plan_curv, 0.0).astype(np.float32)
    prof_curv = np.where(p > 1e-8, prof_curv, 0.0).astype(np.float32)

    # ── TRI — Terrain Ruggedness Index (mean abs difference from 3x3 mean) ─
    dem_mean3 = uniform_filter(dem_filled, size=3)
    tri       = np.abs(dem_filled - dem_mean3).astype(np.float32)

    # ── TPI — Topographic Position Index (deviation from 11x11 mean) ──────
    dem_mean11 = uniform_filter(dem_filled, size=11)
    tpi        = (dem_filled - dem_mean11).astype(np.float32)

    # Restore NaN mask
    for arr in (slope, aspect_sin, aspect_cos, plan_curv, prof_curv, tri, tpi):
        arr[nan_mask] = np.nan

    terrain = {
        "Elevation":  dem.astype(np.float32),
        "Slope":      slope,
        "Aspect_sin": aspect_sin,
        "Aspect_cos": aspect_cos,
        "PlanCurv":   plan_curv,
        "ProfCurv":   prof_curv,
        "TRI":        tri,
        "TPI":        tpi,
    }

    for name, arr in terrain.items():
        print(f"  {name:<12}  min={np.nanmin(arr):9.3f}  "
              f"max={np.nanmax(arr):9.3f}  mean={np.nanmean(arr):9.3f}")

    return terrain


# ---------------------------------------------------------------------------
# Step 7 — Texture features
# ---------------------------------------------------------------------------
def compute_texture_features(b: dict, idx: dict) -> dict:
    _divider("=")
    print(f"  STEP 7 - Texture features  (window = {TEX_WIN}x{TEX_WIN} px = "
          f"{TEX_WIN*10} m)")
    _divider()

    ndvi_mean, ndvi_std = _local_stats(np.nan_to_num(idx["NDVI"]), TEX_WIN)
    nir_mean,  nir_std  = _local_stats(np.nan_to_num(b["B08"]),    TEX_WIN)
    red_mean,  red_std  = _local_stats(np.nan_to_num(b["B04"]),    TEX_WIN)

    texture = {
        "NDVI_std": ndvi_std,
        "NIR_mean": nir_mean,
        "NIR_std":  nir_std,
    }

    for name, arr in texture.items():
        print(f"  {name:<12}  min={np.nanmin(arr):8.4f}  "
              f"max={np.nanmax(arr):8.4f}  mean={np.nanmean(arr):8.4f}")

    return texture


# ---------------------------------------------------------------------------
# Step 8 — Stack and save
# ---------------------------------------------------------------------------
def save_feature_stack(b: dict, idx: dict, terrain: dict, texture: dict,
                       ref_transform, ref_crs) -> np.ndarray:
    _divider("=")
    print("  STEP 8 - Build & save feature stack")
    _divider()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ordered layers matching BAND_NAMES (29 total)
    layers = [
        b["B02"], b["B03"], b["B04"], b["B08"], b["B11"], b["B12"],
        idx["NDVI"], idx["NDBI"], idx["NDWI"], idx["SAVI"], idx["BSI"],
        idx["EVI"], idx["MNDWI"], idx["UI"], idx["NBI"],
        idx["RedGreen"], idx["SR"], idx["SWIRRatio"],
        terrain["Elevation"], terrain["Slope"],
        terrain["Aspect_sin"], terrain["Aspect_cos"],
        terrain["PlanCurv"], terrain["ProfCurv"],
        terrain["TRI"], terrain["TPI"],
        texture["NDVI_std"], texture["NIR_mean"], texture["NIR_std"],
    ]

    assert len(layers) == len(BAND_NAMES), \
        f"Layer count mismatch: {len(layers)} layers vs {len(BAND_NAMES)} names"

    stack = np.stack(layers, axis=0)   # (29, H, W)
    n_bands, h, w = stack.shape

    print(f"  Bands            : {n_bands}")
    print(f"  Spatial size     : {w} x {h} px  ({w*10/1000:.1f} x {h*10/1000:.1f} km)")
    print(f"  Output           : {OUT_STACK}")

    with rasterio.open(
        OUT_STACK, "w",
        driver="GTiff",
        height=h, width=w,
        count=n_bands,
        dtype="float32",
        crs=ref_crs,
        transform=ref_transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for i, (arr, name) in enumerate(zip(layers, BAND_NAMES), 1):
            dst.write(arr, i)
            dst.update_tags(i, name=name)

    size_mb = OUT_STACK.stat().st_size / 1_048_576
    print(f"  File size        : {size_mb:.1f} MB  (deflate compressed)")

    return stack


# ---------------------------------------------------------------------------
# Step 9 — Summary statistics
# ---------------------------------------------------------------------------
def print_statistics(stack: np.ndarray) -> None:
    _divider("=")
    print("  FEATURE STACK STATISTICS")
    _divider()
    print(f"  {'Band':<14}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    _divider()
    for i, name in enumerate(BAND_NAMES):
        band = stack[i]
        vmin  = float(np.nanmin(band))
        vmax  = float(np.nanmax(band))
        vmean = float(np.nanmean(band))
        vstd  = float(np.nanstd(band))
        print(f"  {name:<14}  {vmin:>10.4f}  {vmax:>10.4f}  "
              f"{vmean:>10.4f}  {vstd:>10.4f}")


# ---------------------------------------------------------------------------
# Step 10 — False colour composite
# ---------------------------------------------------------------------------
def save_false_color(b: dict) -> None:
    _divider("=")
    print("  STEP 10 - False colour composite (NIR / Red / Green)")
    _divider()

    nir = b["B08"]
    red = b["B04"]
    grn = b["B03"]

    def stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
        """Percentile linear stretch to [0, 1]."""
        p_lo, p_hi = np.nanpercentile(arr, [lo, hi])
        out = (arr - p_lo) / max(p_hi - p_lo, 1e-6)
        return np.clip(out, 0, 1)

    rgb = np.stack([stretch(nir), stretch(red), stretch(grn)], axis=-1)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.imshow(rgb)
    ax.set_title(
        "Phoenix, AZ — Sentinel-2 False Colour (NIR / Red / Green)\n"
        "Vegetation = red/magenta  |  Urban = cyan/grey  |  Bare soil = tan",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_VIS, bbox_inches="tight")
    plt.close()

    print(f"  Saved -> {OUT_VIS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    _divider("=")
    print("  Phoenix Land Use Classification — Feature Engineering")
    _divider("=")
    print()

    scene_dir = find_best_scene()
    print()

    ref_transform, ref_crs, ref_shape, clip_bounds = build_reference_grid(scene_dir)
    print()

    bands   = load_s2_bands(scene_dir, clip_bounds, ref_transform, ref_crs, ref_shape)
    print()

    dem     = load_dem(clip_bounds, ref_transform, ref_crs, ref_shape)
    print()

    indices = compute_spectral_indices(bands)
    print()

    terrain = compute_terrain_features(dem, cellsize=10.0)
    print()

    texture = compute_texture_features(bands, indices)
    print()

    stack   = save_feature_stack(bands, indices, terrain, texture,
                                 ref_transform, ref_crs)
    print()

    print_statistics(stack)
    print()

    save_false_color(bands)

    print()
    _divider("=")
    print("  ALL STEPS COMPLETE")
    _divider("=")
    print(f"  Feature stack  : {OUT_STACK}")
    print(f"  Bands          : {len(BAND_NAMES)}")
    print(f"  Visualization  : {OUT_VIS}")
    _divider("=")


if __name__ == "__main__":
    main()
