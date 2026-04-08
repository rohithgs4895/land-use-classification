"""
Hard geographic mask: reclassify misclassified northern pixels to Bare_Soil.

The probe shows that above lat 33.55 N:
  - class 4 (mapped as Vegetation) covers 1,410,797 px — northern desert
    misclassified as non-desert.  The user identifies these as the
    problematic "Water (class 4)" cyan/blue areas.
  - class 5 (Water) = 0 px — already corrected by Step 2.5.

Fix: any pixel above lat 33.55 N with class 4 OR class 5 -> class 1 (Bare_Soil).

Outputs updated in-place:
  outputs/landuse_prediction.tif
  outputs/landuse_colored.png
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as warp_transform

PROJECT_DIR = r"C:\Projects\land-use-classification"
PRED_TIF    = f"{PROJECT_DIR}/outputs/landuse_prediction.tif"
COLORED_PNG = f"{PROJECT_DIR}/outputs/landuse_colored.png"

LAT_THRESHOLD = 33.55   # degrees N

# Must match predict_landuse.py CLASS_META exactly
CLASS_META = {
    0: ("Agricultural", (200, 190,  60)),
    1: ("Bare_Soil",    (180, 150, 100)),
    2: ("Industrial",   (150,  50, 180)),
    3: ("Urban",        (220,  50,  50)),
    4: ("Vegetation",   ( 50, 160,  50)),
    5: ("Water",        ( 50, 100, 220)),
}
NODATA_VAL  = 255
BARE_SOIL   = 1   # target class for reclassification

# Classes to reclassify in the northern zone
RECLASSIFY  = {4, 5}   # class 4 (Vegetation misclassified as desert) + class 5 (Water)


def _divider(char="-", width=64):
    print(char * width)


# ---------------------------------------------------------------------------
# Step 1 — Apply geographic mask
# ---------------------------------------------------------------------------
_divider("=")
print("  Apply hard geographic mask")
_divider()
t0 = time.time()

with rasterio.open(PRED_TIF) as src:
    pred      = src.read(1)
    transform = src.transform
    crs       = src.crs
    h, w      = src.shape

# Convert lat threshold to raster CRS (UTM Zone 12N)
_, ys = warp_transform("EPSG:4326", crs, [-111.9], [LAT_THRESHOLD])
north_y      = ys[0]
thresh_row   = int((north_y - transform.f) / transform.e)
thresh_row   = max(0, min(h, thresh_row))

print(f"  Raster           : {w} x {h}")
print(f"  Lat threshold    : {LAT_THRESHOLD} N  ->  row {thresh_row}")
print()

# Build northern mask and apply reclassification
north_mask = np.zeros(pred.shape, dtype=bool)
north_mask[:thresh_row, :] = True

total_fixed = 0
for cls_id in sorted(RECLASSIFY):
    to_fix = north_mask & (pred == cls_id)
    n      = int(to_fix.sum())
    pred[to_fix] = BARE_SOIL
    name   = CLASS_META[cls_id][0]
    print(f"  class {cls_id} ({name:<12}) north of {LAT_THRESHOLD} N  ->  {n:>10,} px reclassified to Bare_Soil")
    total_fixed += n

print()
print(f"  Total reclassified : {total_fixed:,} px  ({total_fixed*0.0001:.1f} km²)")

# Write modified raster back
with rasterio.open(PRED_TIF, "r+") as dst:
    dst.write(pred, 1)
print(f"  Saved -> {PRED_TIF}")

# ---------------------------------------------------------------------------
# Step 2 — Verify northern region after fix
# ---------------------------------------------------------------------------
_divider("=")
print("  Northern region class counts (after fix)")
_divider()
north = pred[:thresh_row, :]
for v in sorted(np.unique(north)):
    if v == NODATA_VAL:
        continue
    n    = int((north == v).sum())
    name = CLASS_META.get(v, ("?",))[0]
    print(f"  class {v} ({name:<12})  {n:>10,} px  ({n/north.size*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 3 — Regenerate colour map
# ---------------------------------------------------------------------------
_divider("=")
print("  Regenerating colour map")
_divider()

rgb = np.zeros((h, w, 3), dtype=np.uint8)
for cid, (name, colour) in CLASS_META.items():
    rgb[pred == cid] = colour
rgb[pred == NODATA_VAL] = (200, 200, 200)

fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
ax.imshow(rgb, interpolation="none")
ax.set_title(
    "Phoenix Metro — Land Use / Land Cover Classification\n"
    "Sentinel-2 L2A  |  29 features  |  Histogram Gradient Boosting  |  "
    "95.8% accuracy  |  Geographic mask applied (lat > 33.55 N)",
    fontsize=10, pad=10,
)
ax.axis("off")

patches = [
    mpatches.Patch(color=np.array(colour) / 255, label=name)
    for cid, (name, colour) in sorted(CLASS_META.items())
]
ax.legend(
    handles       = patches,
    loc           = "lower right",
    fontsize      = 9,
    title         = "Land Use Class",
    title_fontsize= 10,
    framealpha    = 0.9,
    edgecolor     = "grey",
)

plt.tight_layout()
plt.savefig(COLORED_PNG, bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved -> {COLORED_PNG}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_divider("=")
print("  UPDATED AREA SUMMARY")
_divider()
print(f"  {'Class':<16}  {'Pixels':>10}  {'Area (km²)':>12}  {'Coverage':>9}")
_divider()
total_valid = int((pred != NODATA_VAL).sum())
for cid, (name, colour) in sorted(CLASS_META.items()):
    n_px = int((pred == cid).sum())
    km2  = n_px * 0.0001
    pct  = n_px / total_valid * 100 if total_valid else 0
    print(f"  {name:<16}  {n_px:>10,}  {km2:>12.1f}  {pct:>8.1f}%")
_divider()
print(f"  {'TOTAL':<16}  {total_valid:>10,}  {total_valid*0.0001:>12.1f}  {'100.0%':>9}")

elapsed = time.time() - t0
print(f"\n  Done in {elapsed:.1f} s")
