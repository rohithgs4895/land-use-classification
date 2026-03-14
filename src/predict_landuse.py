"""
Full-scene land use prediction for the Phoenix study area.

Loads best_model.pkl, applies it to every pixel in feature_stack.tif,
and generates a classified raster, colour map, vector polygons, and
area statistics.

Class ID mapping (LabelEncoder alphabetical sort)
--------------------------------------------------
  0  Agricultural   irrigated crops / farms
  1  Bare_Soil      undeveloped desert, sandy washes
  2  Industrial     warehouses, airpark, factories
  3  Urban          buildings, roads, commercial
  4  Vegetation     parks, golf courses, preserves
  5  Water          lakes, canals, river

Outputs
-------
  outputs/landuse_prediction.tif   uint8 class-ID raster (nodata=255)
  outputs/landuse_colored.png      RGB colour map with legend
  outputs/landuse_polygons.shp     dissolved class polygons (ArcGIS ready)
  outputs/landuse_summary.csv      area per class in km² and %

Usage
-----
    conda run -n arcgis-dl python src/predict_landuse.py
"""

import pickle
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(r"C:\Projects\land-use-classification")
STACK_PATH    = PROJECT_DIR / "outputs" / "feature_stack.tif"
MODELS_DIR    = PROJECT_DIR / "models"
OUTPUTS_DIR   = PROJECT_DIR / "outputs"

BEST_MODEL    = MODELS_DIR  / "best_model.pkl"
SCALER_PATH   = MODELS_DIR  / "scaler.pkl"

PRED_TIF      = OUTPUTS_DIR / "landuse_prediction.tif"
COLORED_PNG   = OUTPUTS_DIR / "landuse_colored.png"
POLY_SHP      = OUTPUTS_DIR / "landuse_polygons.shp"
SUMMARY_CSV   = OUTPUTS_DIR / "landuse_summary.csv"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
CLASS_META = {
    # class_id: (name,          colour_RGB,        display_colour)
    0: ("Agricultural", (200, 190,  60), "#C8BE3C"),
    1: ("Bare_Soil",    (180, 150, 100), "#B49664"),
    2: ("Industrial",   (150,  50, 180), "#9632B4"),
    3: ("Urban",        (220,  50,  50), "#DC3232"),
    4: ("Vegetation",   ( 50, 160,  50), "#32A032"),
    5: ("Water",        ( 50, 100, 220), "#3264DC"),
}
NODATA_VAL  = 255
CHUNK_ROWS  = 512   # process this many rows at a time

# Latitude above which Water predictions are reclassified to Bare_Soil.
# All confirmed open water (Tempe Town Lake, Salt River) lies below 33.45°N.
LAT_WATER_THRESHOLD = 33.55   # degrees N


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _divider(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.1f} s"


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Step 1 — Load model and scaler
# ---------------------------------------------------------------------------
def load_model_and_scaler():
    _divider("=")
    print("  STEP 1 - Load model and scaler")
    _divider()

    model  = _load_pickle(BEST_MODEL)
    scaler = _load_pickle(SCALER_PATH)

    print(f"  Model  : {type(model).__name__}  ({BEST_MODEL.stat().st_size/1e6:.1f} MB)")
    print(f"  Scaler : {type(scaler).__name__}  ({SCALER_PATH.stat().st_size/1e3:.1f} KB)")
    print(f"  Classes: {list(CLASS_META[i][0] for i in range(6))}")

    return model, scaler


# ---------------------------------------------------------------------------
# Step 2 — Run pixel-wise prediction (chunked)
# ---------------------------------------------------------------------------
def predict_full_scene(model, scaler) -> None:
    _divider("=")
    print("  STEP 2 - Predict full scene  (chunked, {}-row blocks)".format(CHUNK_ROWS))
    _divider()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(STACK_PATH) as src:
        profile = src.profile.copy()
        h, w    = src.shape
        n_bands = src.count

    profile.update(
        driver   = "GTiff",
        count    = 1,
        dtype    = "uint8",
        nodata   = NODATA_VAL,
        compress = "lzw",
        tiled    = True,
        blockxsize=256,
        blockysize=256,
        photometric="MINISBLACK",
    )

    total_px   = h * w
    pred_px    = 0
    nodata_px  = 0
    t0         = time.time()

    print(f"  Raster : {w} x {h} px  =  {total_px/1e6:.1f} M pixels")
    print(f"  Bands  : {n_bands}")
    print()

    with rasterio.open(STACK_PATH) as src, \
         rasterio.open(PRED_TIF, "w", **profile) as dst:

        n_chunks = (h + CHUNK_ROWS - 1) // CHUNK_ROWS

        for chunk_idx, row_start in enumerate(range(0, h, CHUNK_ROWS)):
            row_end   = min(row_start + CHUNK_ROWS, h)
            chunk_h   = row_end - row_start
            win       = rasterio.windows.Window(0, row_start, w, chunk_h)

            # Read all bands for this chunk: (n_bands, chunk_h, w)
            chunk     = src.read(window=win).astype(np.float32)

            # Reshape to pixels: (chunk_h * w, n_bands)
            pixels    = chunk.reshape(n_bands, -1).T

            # Identify valid pixels (no NaN in any band)
            valid     = ~np.isnan(pixels).any(axis=1)

            pred_row  = np.full(chunk_h * w, NODATA_VAL, dtype=np.uint8)

            if valid.any():
                X_valid     = scaler.transform(pixels[valid])
                labels      = model.predict(X_valid).astype(np.uint8)
                pred_row[valid] = labels

            pred_px   += int(valid.sum())
            nodata_px += int((~valid).sum())

            dst.write(pred_row.reshape(1, chunk_h, w), window=win)

            pct = (chunk_idx + 1) / n_chunks * 100
            print(f"  Chunk {chunk_idx+1:3d}/{n_chunks}  "
                  f"rows {row_start:5d}-{row_end:5d}  "
                  f"({pct:5.1f}%)  elapsed: {_elapsed(t0)}",
                  end="\r")

    print()
    print()
    print(f"  Total pixels  : {total_px:,}")
    print(f"  Predicted     : {pred_px:,}  ({pred_px/total_px*100:.2f}%)")
    print(f"  No-data       : {nodata_px:,}  ({nodata_px/total_px*100:.2f}%)")
    print(f"  Elapsed       : {_elapsed(t0)}")
    print(f"  Saved -> {PRED_TIF.name}")


# ---------------------------------------------------------------------------
# Step 2.5 — Geographic water constraint
# ---------------------------------------------------------------------------
def apply_water_geographic_constraint() -> int:
    """
    Post-process: reclassify Water (class 5) pixels north of LAT_WATER_THRESHOLD
    to Bare_Soil (class 1).  All confirmed open water in the Phoenix study area
    (Tempe Town Lake, Salt River) lies below 33.45°N — well south of the threshold.
    """
    _divider("=")
    print(f"  STEP 2.5 — Geographic water constraint  (lat > {LAT_WATER_THRESHOLD}°N → Bare_Soil)")
    _divider()

    WATER_ID   = 5
    BARE_ID    = 1

    with rasterio.open(PRED_TIF) as src:
        pred      = src.read(1)
        transform = src.transform
        crs       = src.crs

    # Convert the latitude threshold to the raster CRS (UTM Zone 12N).
    # UTM northing depends only on latitude; use the study-area central meridian.
    _, ys = warp_transform("EPSG:4326", crs, [-111.9], [LAT_WATER_THRESHOLD])
    north_y = ys[0]

    # Affine: y = transform.f + row * transform.e  (transform.e < 0 for north-up)
    # → row = (y − transform.f) / transform.e
    threshold_row = int((north_y - transform.f) / transform.e)
    threshold_row = max(0, min(pred.shape[0], threshold_row))

    # Rows 0 … threshold_row-1 are north of the latitude threshold.
    north_mask = np.zeros(pred.shape, dtype=bool)
    north_mask[:threshold_row, :] = True
    to_fix = north_mask & (pred == WATER_ID)
    n_fixed = int(to_fix.sum())
    pred[to_fix] = BARE_ID

    with rasterio.open(PRED_TIF, "r+") as dst:
        dst.write(pred, 1)

    print(f"  Lat threshold    : {LAT_WATER_THRESHOLD}°N  →  raster row {threshold_row}")
    print(f"  Water→Bare_Soil  : {n_fixed:,} pixels reclassified")
    return n_fixed


# ---------------------------------------------------------------------------
# Step 3 — Colour map with legend
# ---------------------------------------------------------------------------
def make_colored_map() -> None:
    _divider("=")
    print("  STEP 3 - Colour map with legend")
    _divider()

    with rasterio.open(PRED_TIF) as src:
        pred = src.read(1)    # (H, W) uint8

    h, w = pred.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)

    for cid, (name, colour, _) in CLASS_META.items():
        mask = pred == cid
        rgb[mask] = colour

    # Nodata pixels → light grey
    rgb[pred == NODATA_VAL] = (200, 200, 200)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.imshow(rgb, interpolation="none")
    ax.set_title(
        "Phoenix Metro — Land Use / Land Cover Classification\n"
        "Sentinel-2 L2A  |  29 features  |  Histogram Gradient Boosting  |  "
        "94.7% accuracy",
        fontsize=11, pad=10,
    )
    ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(colour) / 255, label=f"{name}")
        for cid, (name, colour, _) in sorted(CLASS_META.items())
    ]
    ax.legend(
        handles      = patches,
        loc          = "lower right",
        fontsize     = 9,
        title        = "Land Use Class",
        title_fontsize= 10,
        framealpha   = 0.9,
        edgecolor    = "grey",
    )

    plt.tight_layout()
    plt.savefig(COLORED_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved -> {COLORED_PNG.name}  ({COLORED_PNG.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 4 — Vectorise prediction raster → dissolved class polygons
# ---------------------------------------------------------------------------
def vectorize_prediction() -> None:
    _divider("=")
    print("  STEP 4 - Vectorise prediction -> dissolved class polygons")
    _divider()

    t0 = time.time()

    with rasterio.open(PRED_TIF) as src:
        pred      = src.read(1)
        transform = src.transform
        crs       = src.crs

    # Valid pixel mask (exclude nodata)
    valid_mask = (pred != NODATA_VAL).astype(np.uint8)

    print(f"  Generating shapes for {valid_mask.sum():,} valid pixels ...")
    geom_list = []
    for geom_dict, val in rio_shapes(pred, mask=valid_mask, transform=transform):
        if int(val) == NODATA_VAL:
            continue
        geom_list.append({
            "class_id":   int(val),
            "class_name": CLASS_META[int(val)][0],
            "geometry":   shape(geom_dict),
        })

    print(f"  Raw shapes       : {len(geom_list):,}")

    gdf = gpd.GeoDataFrame(geom_list, crs=crs)

    # Dissolve into one multipolygon per class, then simplify
    print("  Dissolving by class ...")
    gdf_diss = (
        gdf.dissolve(by="class_id")
           .reset_index()
    )
    gdf_diss["class_name"] = gdf_diss["class_id"].map(
        lambda i: CLASS_META[i][0]
    )

    # Simplify geometry to 20 m tolerance (2 pixels) — reduces file size
    print("  Simplifying geometry (tolerance = 20 m) ...")
    gdf_diss["geometry"] = gdf_diss["geometry"].simplify(
        tolerance=20.0, preserve_topology=True
    )

    gdf_diss.to_file(str(POLY_SHP))
    shp_mb = sum(p.stat().st_size for p in POLY_SHP.parent.glob("landuse_polygons.*")) / 1e6

    print(f"  Classes vectorised: {len(gdf_diss)}")
    print(f"  Elapsed           : {_elapsed(t0)}")
    print(f"  Saved -> {POLY_SHP.name}  ({shp_mb:.1f} MB total)")


# ---------------------------------------------------------------------------
# Step 5 — Area statistics
# ---------------------------------------------------------------------------
def area_statistics() -> pd.DataFrame:
    _divider("=")
    print("  STEP 5 - Area statistics")
    _divider()

    with rasterio.open(PRED_TIF) as src:
        pred = src.read(1)
        res  = src.res   # (pixel_width, pixel_height) in metres

    pixel_area_km2 = (res[0] * res[1]) / 1_000_000   # 10 × 10 m = 0.0001 km²

    rows = []
    total_valid = int((pred != NODATA_VAL).sum())

    for cid, (name, _, colour_hex) in CLASS_META.items():
        n_px   = int((pred == cid).sum())
        km2    = n_px * pixel_area_km2
        pct    = n_px / total_valid * 100 if total_valid else 0.0
        rows.append({
            "class_id":   cid,
            "class_name": name,
            "pixels":     n_px,
            "area_km2":   round(km2, 2),
            "pct":        round(pct, 1),
        })

    # Totals row
    total_km2 = sum(r["area_km2"] for r in rows)
    rows.append({
        "class_id":   -1,
        "class_name": "TOTAL",
        "pixels":     total_valid,
        "area_km2":   round(total_km2, 2),
        "pct":        100.0,
    })

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved -> {SUMMARY_CSV.name}")

    return df


# ---------------------------------------------------------------------------
# Step 6 — Summary report
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    _divider("=")
    print("  LAND USE SUMMARY — Phoenix Study Area")
    _divider()
    print(f"  {'Class':<16}  {'Pixels':>10}  {'Area (km²)':>12}  {'Coverage':>9}")
    _divider()
    for _, row in df.iterrows():
        if row["class_name"] == "TOTAL":
            _divider()
        sym = {
            "Urban":        "[RED   ]",
            "Vegetation":   "[GREEN ]",
            "Agricultural": "[YELLOW]",
            "Bare_Soil":    "[TAN   ]",
            "Water":        "[BLUE  ]",
            "Industrial":   "[PURPLE]",
            "TOTAL":        "        ",
        }.get(row["class_name"], "")
        print(f"  {row['class_name']:<16}  {row['pixels']:>10,}  "
              f"{row['area_km2']:>12.1f}  {row['pct']:>8.1f}%  {sym}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    _divider("=")
    print("  Phoenix Land Use — Full-Scene Prediction")
    _divider("=")
    print()

    model, scaler = load_model_and_scaler()
    print()

    predict_full_scene(model, scaler)
    print()

    apply_water_geographic_constraint()
    print()

    make_colored_map()
    print()

    vectorize_prediction()
    print()

    df = area_statistics()
    print()

    print_summary(df)

    print()
    _divider("=")
    print("  ALL OUTPUTS SAVED")
    _divider("=")
    for p in [PRED_TIF, COLORED_PNG, POLY_SHP, SUMMARY_CSV]:
        mb = p.stat().st_size / 1e6
        print(f"  {p.name:<40}  {mb:6.1f} MB")
    _divider("=")


if __name__ == "__main__":
    main()
