"""
Create labeled training samples for Phoenix land use classification.

Approach
--------
Known Phoenix-area landmarks define representative polygons for each class.
Pixel values are extracted from outputs/feature_stack.tif at every grid cell
that falls inside a polygon.  Each pixel becomes one training row.

Classes
-------
  1  Urban        roads, buildings, commercial, residential
  2  Vegetation   parks, golf courses, preserves, riparian trees
  3  Agricultural irrigated cropland, farms
  4  Bare_Soil    undeveloped desert, sandy washes
  5  Water        Tempe Town Lake, Arizona Canal, Salt River
  6  Industrial   warehouses, airpark, factories

Coverage (feature_stack.tif)
-----------------------------
  lon  [-112.08, -111.70]   lat  [33.35, 33.71]
  EPSG:32612  (UTM Zone 12N)  10 m / pixel

Outputs
-------
  data/processed/training_samples.shp      — polygon training areas (WGS84)
  data/processed/training_samples.geojson  — same, GeoJSON format
  data/features/training_features.csv      — one row per pixel, 29 features

Usage
-----
    conda run -n arcgis-dl python src/create_training_samples.py
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(r"C:\Projects\land-use-classification")
FEATURE_STACK = PROJECT_DIR / "outputs"    / "feature_stack.tif"
PROC_DIR      = PROJECT_DIR / "data"       / "processed"
FEAT_DIR      = PROJECT_DIR / "data"       / "features"
OUT_SHP       = PROC_DIR / "training_samples.shp"
OUT_GEOJSON   = PROC_DIR / "training_samples.geojson"
OUT_CSV       = FEAT_DIR / "training_features.csv"

# EVI is band index 11 (0-based) — clip to physical range
EVI_BAND_IDX = 11       # 0-based position in BAND_NAMES
EVI_CLIP     = (-1.0, 2.0)

# ---------------------------------------------------------------------------
# Training polygon definitions
# All coordinates are (lon_center, lat_center, lon_half_width, lat_half_height)
# in WGS84.  Each box ~0.5–1 km wide at 33°N latitude.
# ---------------------------------------------------------------------------
TRAINING_AREAS = [
    # ── Urban / Built-up ──────────────────────────────────────────────────
    # Dense development: Scottsdale Old Town, Tempe ASU district, Mesa Downtown,
    # and North Scottsdale (Kierland/Scottsdale Quarter retail/office).
    dict(cls="Urban",        cid=1, lon=-111.927, lat=33.494, dlon=0.006, dlat=0.004,
         desc="Old Town Scottsdale"),
    dict(cls="Urban",        cid=1, lon=-111.928, lat=33.420, dlon=0.006, dlat=0.004,
         desc="Tempe ASU / Mill Ave district"),
    dict(cls="Urban",        cid=1, lon=-111.832, lat=33.415, dlon=0.005, dlat=0.004,
         desc="Mesa Downtown"),
    dict(cls="Urban",        cid=1, lon=-111.921, lat=33.638, dlon=0.006, dlat=0.004,
         desc="N Scottsdale / Kierland"),

    # ── Vegetation ────────────────────────────────────────────────────────
    # Papago Park (desert buttes + irrigated lawns), McCormick Ranch Golf,
    # McDowell Mountain Regional Park, Scottsdale Native Plant Preserve.
    dict(cls="Vegetation",   cid=2, lon=-111.948, lat=33.459, dlon=0.005, dlat=0.004,
         desc="Papago Park"),
    dict(cls="Vegetation",   cid=2, lon=-111.876, lat=33.556, dlon=0.006, dlat=0.004,
         desc="McCormick Ranch Golf Club"),
    dict(cls="Vegetation",   cid=2, lon=-111.845, lat=33.598, dlon=0.006, dlat=0.005,
         desc="McDowell Mountain Regional Park"),
    dict(cls="Vegetation",   cid=2, lon=-111.781, lat=33.635, dlon=0.005, dlat=0.004,
         desc="Scottsdale native desert preserve"),

    # ── Agricultural ──────────────────────────────────────────────────────
    # East Mesa and Gilbert still have active irrigated farmland (citrus, alfalfa,
    # vegetables) fed by the SRP canal system.
    dict(cls="Agricultural", cid=3, lon=-111.752, lat=33.374, dlon=0.007, dlat=0.005,
         desc="East Mesa irrigated farmland"),
    dict(cls="Agricultural", cid=3, lon=-111.748, lat=33.357, dlon=0.006, dlat=0.004,
         desc="Gilbert irrigated fields"),
    dict(cls="Agricultural", cid=3, lon=-111.736, lat=33.407, dlon=0.006, dlat=0.004,
         desc="East Mesa SRP irrigation area"),

    # ── Bare Soil / Desert ────────────────────────────────────────────────
    # McDowell Mountains rocky desert, Scottsdale desert preserve scrub,
    # open Sonoran Desert, undeveloped East Mesa caliche flats.
    # Northern samples added to fix Water misclassification in the upper study area.
    dict(cls="Bare_Soil",    cid=4, lon=-111.835, lat=33.575, dlon=0.006, dlat=0.005,
         desc="McDowell Mountains rocky desert"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.802, lat=33.510, dlon=0.005, dlat=0.004,
         desc="Scottsdale desert preserve scrub"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.742, lat=33.485, dlon=0.005, dlat=0.004,
         desc="Sonoran Desert open scrub"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.752, lat=33.427, dlon=0.005, dlat=0.004,
         desc="East Mesa undeveloped caliche"),
    # Northern bare-soil additions (lat 33.58–33.68)
    dict(cls="Bare_Soil",    cid=4, lon=-111.984, lat=33.625, dlon=0.006, dlat=0.005,
         desc="Cave Creek desert scrub — north"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.921, lat=33.660, dlon=0.007, dlat=0.005,
         desc="North Scottsdale / Carefree desert"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.855, lat=33.670, dlon=0.007, dlat=0.004,
         desc="McDowell Sonoran Preserve north"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.776, lat=33.658, dlon=0.006, dlat=0.004,
         desc="Fort McDowell desert flats — north"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.940, lat=33.585, dlon=0.006, dlat=0.004,
         desc="North Phoenix desert — Cave Creek Rd"),
    dict(cls="Bare_Soil",    cid=4, lon=-111.806, lat=33.607, dlon=0.005, dlat=0.004,
         desc="North Scottsdale desert ridge"),

    # ── Water ─────────────────────────────────────────────────────────────
    # Only large permanent water bodies south of 33.55°N.
    # Arizona Canal removed — too narrow (<30 m) for clean 10 m pixel sampling;
    # mixed-pixel effects were contributing to false Water detections in the north.
    dict(cls="Water",        cid=5, lon=-111.921, lat=33.428, dlon=0.010, dlat=0.002,
         desc="Tempe Town Lake (Salt River impoundment)"),
    dict(cls="Water",        cid=5, lon=-111.860, lat=33.393, dlon=0.008, dlat=0.002,
         desc="Salt River channel west"),
    dict(cls="Water",        cid=5, lon=-111.810, lat=33.390, dlon=0.007, dlat=0.002,
         desc="Salt River channel east"),

    # ── Industrial ────────────────────────────────────────────────────────
    # Scottsdale Airpark (largest master-planned industrial park in the US),
    # Mesa industrial/warehouse corridor, east Phoenix light-industrial.
    dict(cls="Industrial",   cid=6, lon=-111.906, lat=33.619, dlon=0.006, dlat=0.004,
         desc="Scottsdale Airpark"),
    dict(cls="Industrial",   cid=6, lon=-111.860, lat=33.412, dlon=0.005, dlat=0.004,
         desc="Mesa Superstition industrial"),
    dict(cls="Industrial",   cid=6, lon=-111.940, lat=33.474, dlon=0.005, dlat=0.003,
         desc="East Phoenix industrial"),
    dict(cls="Industrial",   cid=6, lon=-111.844, lat=33.369, dlon=0.005, dlat=0.003,
         desc="Chandler industrial corridor"),
]

# Band names in the same order as feature_stack.tif
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


# ---------------------------------------------------------------------------
# Step 1 — Build and save training polygons
# ---------------------------------------------------------------------------
def build_polygons() -> gpd.GeoDataFrame:
    _divider("=")
    print("  STEP 1 - Build training polygons")
    _divider()

    rows = []
    for area in TRAINING_AREAS:
        geom = box(
            area["lon"] - area["dlon"],
            area["lat"] - area["dlat"],
            area["lon"] + area["dlon"],
            area["lat"] + area["dlat"],
        )
        rows.append({
            "class":    area["cls"],
            "class_id": area["cid"],
            "desc":     area["desc"],
            "geometry": geom,
            "width_m":  round(area["dlon"] * 2 * 111_320 * np.cos(np.radians(area["lat"])), 0),
            "height_m": round(area["dlat"] * 2 * 111_320, 0),
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(OUT_SHP))
    gdf.to_file(str(OUT_GEOJSON), driver="GeoJSON")

    print(f"  Polygons defined : {len(gdf)}")
    for cls in gdf["class"].unique():
        n = (gdf["class"] == cls).sum()
        print(f"    {cls:<16} {n} polygons")
    print(f"  Saved -> {OUT_SHP.name}")
    print(f"  Saved -> {OUT_GEOJSON.name}")

    return gdf


# ---------------------------------------------------------------------------
# Step 2 — Extract pixel features from the feature stack
# ---------------------------------------------------------------------------
def extract_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    _divider("=")
    print("  STEP 2 - Extract pixel features from feature_stack.tif")
    _divider()
    print(f"  Feature stack : {FEATURE_STACK}")
    print(f"  Bands         : {len(BAND_NAMES)}")
    print()

    with rasterio.open(FEATURE_STACK) as src:
        stack_crs    = src.crs
        stack_bounds = src.bounds
        nodata       = src.nodata

    # Reproject polygons to match the raster CRS
    gdf_utm = gdf.to_crs(stack_crs)

    all_rows = []
    skipped  = 0

    for _, poly_row in gdf_utm.iterrows():
        geom    = poly_row["geometry"]
        cls     = poly_row["class"]
        cid     = poly_row["class_id"]
        desc    = poly_row["desc"]

        # Skip polygons outside raster extent
        if not geom.intersects(
            box(stack_bounds.left, stack_bounds.bottom,
                stack_bounds.right, stack_bounds.top)
        ):
            print(f"  [skip] {cls} — '{desc}'  (outside raster bounds)")
            skipped += 1
            continue

        # Intersect polygon with raster bounds
        clipped_geom = geom.intersection(
            box(stack_bounds.left, stack_bounds.bottom,
                stack_bounds.right, stack_bounds.top)
        )
        if clipped_geom.is_empty:
            skipped += 1
            continue

        with rasterio.open(FEATURE_STACK) as src:
            data, _ = rio_mask(
                src,
                [clipped_geom.__geo_interface__],
                crop=True,
                nodata=np.nan,
                all_touched=False,
            )
        # data shape: (29, H, W)

        # Clip EVI to physical range before any other processing
        data[EVI_BAND_IDX] = np.clip(data[EVI_BAND_IDX], *EVI_CLIP)

        # Reshape to (H*W, 29) and drop pixels that are entirely NaN
        n_bands, h, w = data.shape
        pixels = data.reshape(n_bands, -1).T          # (N, 29)
        valid  = ~np.isnan(pixels).any(axis=1)
        pixels = pixels[valid]

        if pixels.shape[0] == 0:
            print(f"  [warn] {cls} — '{desc}'  0 valid pixels after masking")
            skipped += 1
            continue

        # Build per-pixel rows
        df = pd.DataFrame(pixels, columns=BAND_NAMES)
        df.insert(0, "class",    cls)
        df.insert(1, "class_id", cid)
        all_rows.append(df)

        print(f"  {cls:<16}  '{desc}'  ->  {len(df):,} pixels")

    if not all_rows:
        raise RuntimeError("No valid pixels extracted.  Check polygon coordinates.")

    features_df = pd.concat(all_rows, ignore_index=True)
    print()
    print(f"  Polygons skipped  : {skipped}")
    print(f"  Total pixel rows  : {len(features_df):,}")

    return features_df


# ---------------------------------------------------------------------------
# Step 3 — Save feature CSV
# ---------------------------------------------------------------------------
def save_features(df: pd.DataFrame) -> None:
    _divider("=")
    print("  STEP 3 - Save training_features.csv")
    _divider()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    size_kb = OUT_CSV.stat().st_size / 1024
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}  (class, class_id, {len(BAND_NAMES)} features)")
    print(f"  File    : {OUT_CSV}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Step 4 — Class distribution & feature statistics
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    _divider("=")
    print("  CLASS DISTRIBUTION")
    _divider()
    total = len(df)
    cls_order = ["Urban", "Vegetation", "Agricultural", "Bare_Soil", "Water", "Industrial"]
    print(f"  {'Class':<16}  {'Pixels':>8}  {'%':>6}  {'Polygons':>8}")
    _divider()
    for cls in cls_order:
        sub  = df[df["class"] == cls]
        n    = len(sub)
        pct  = n / total * 100
        # count unique descriptions from gdf — approximate via sampling
        print(f"  {cls:<16}  {n:>8,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total:>8,}  100.0%")

    _divider("=")
    print("  FEATURE STATISTICS (mean ± std  by class)")
    _divider()
    key_features = ["NDVI", "NDBI", "NDWI", "BSI", "Elevation", "Slope"]
    header = f"  {'Class':<16}" + "".join(f"  {f:>10}" for f in key_features)
    print(header)
    _divider()
    for cls in cls_order:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        means = "".join(f"  {sub[f].mean():>10.3f}" for f in key_features)
        print(f"  {cls:<16}{means}")

    _divider("=")
    print("  FULL FEATURE STATISTICS  (all 29 bands)")
    _divider()
    print(f"  {'Feature':<14}  {'Min':>9}  {'Max':>9}  {'Mean':>9}  {'Std':>9}")
    _divider()
    for feat in BAND_NAMES:
        col = df[feat]
        print(f"  {feat:<14}  {col.min():>9.4f}  {col.max():>9.4f}  "
              f"{col.mean():>9.4f}  {col.std():>9.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    _divider("=")
    print("  Phoenix Land Use — Create Training Samples")
    _divider("=")
    print()

    gdf      = build_polygons()
    print()

    features = extract_features(gdf)
    print()

    save_features(features)
    print()

    print_summary(features)

    print()
    _divider("=")
    print("  ALL STEPS COMPLETE")
    _divider("=")
    print(f"  Polygons (shp)   : {OUT_SHP}")
    print(f"  Polygons (json)  : {OUT_GEOJSON}")
    print(f"  Training CSV     : {OUT_CSV}")
    print(f"  Total pixels     : {len(features):,}")
    print(f"  Classes          : 6")
    print(f"  Features/pixel   : {len(BAND_NAMES)}")
    _divider("=")


if __name__ == "__main__":
    main()
