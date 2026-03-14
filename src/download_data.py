"""
Download Sentinel-2 L2A imagery and SRTM DEM for the Phoenix, AZ region.

Sources
-------
  Sentinel-2 L2A   Microsoft Planetary Computer (STAC API, no account needed)
  SRTM DEM         Microsoft Planetary Computer — cop-dem-glo-30 collection

Area of interest
----------------
  Phoenix, AZ bounding box: -112.5, 33.2, -111.7, 33.7  (~500 km²)

Sentinel-2 bands downloaded
---------------------------
  B02  Blue      10 m
  B03  Green     10 m
  B04  Red       10 m
  B08  NIR       10 m
  B11  SWIR      20 m
  B12  SWIR2     20 m
  SCL  Scene Classification Layer  (cloud mask)

Outputs
-------
  data/raw/sentinel2/   One sub-folder per scene (scene date)
  data/raw/dem/         SRTM elevation tiles

Usage
-----
    conda run -n arcgis-dl python src/download_data.py
"""

import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import planetary_computer
import pystac_client
import rasterio
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\Projects\land-use-classification")
S2_DIR      = PROJECT_DIR / "data" / "raw" / "sentinel2"
DEM_DIR     = PROJECT_DIR / "data" / "raw" / "dem"

# Phoenix, AZ  (west, south, east, north)
AOI_BBOX    = [-112.5, 33.2, -111.7, 33.7]

DATE_RANGE  = "2023-06-01/2023-08-31"
MAX_CLOUD   = 5        # percent
MAX_SCENES  = 3        # download the N clearest scenes

S2_BANDS    = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _divider(char: str = "-", width: int = 60) -> None:
    print(char * width)


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.1f} MB"


def _download(url: str, dest: Path, label: str) -> None:
    """Stream-download url to dest with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"    [skip] {dest.name}  (already exists)")
        return

    signed = planetary_computer.sign(url)
    resp = requests.get(signed, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):   # 1 MB chunks
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"    {label}: {pct:5.1f}%  ({_mb(downloaded)} / {_mb(total)})",
                      end="\r")
    print(f"    {label}: 100.0%  ({_mb(downloaded)})          ")


# ---------------------------------------------------------------------------
# Step 1 — Search Sentinel-2 scenes
# ---------------------------------------------------------------------------
def search_sentinel2() -> list:
    _divider("=")
    print("  STEP 1 - Search Sentinel-2 L2A scenes")
    _divider()
    print(f"  Catalog  : {PC_STAC_URL}")
    print(f"  AOI      : {AOI_BBOX}  (Phoenix, AZ)")
    print(f"  Dates    : {DATE_RANGE}")
    print(f"  Max cloud: {MAX_CLOUD}%")
    print()

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=AOI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )

    items = list(search.items())

    print(f"  Scenes found: {len(items)}")
    print()
    print(f"  {'#':<4} {'Date':<12} {'Cloud%':>7}  {'Scene ID'}")
    _divider()
    for i, item in enumerate(items, 1):
        date  = item.datetime.strftime("%Y-%m-%d") if item.datetime else "N/A"
        cloud = item.properties.get("eo:cloud_cover", "N/A")
        print(f"  {i:<4} {date:<12} {cloud:>6.2f}%   {item.id}")

    return items


# ---------------------------------------------------------------------------
# Step 2 — Download Sentinel-2 bands
# ---------------------------------------------------------------------------
def download_sentinel2(items: list) -> list:
    _divider("=")
    print("  STEP 2 - Download Sentinel-2 bands")
    _divider()
    print(f"  Bands    : {', '.join(S2_BANDS)}")
    print(f"  Scenes   : {min(MAX_SCENES, len(items))} (lowest cloud cover)")
    print()

    downloaded_dirs = []

    for item in items[:MAX_SCENES]:
        date  = item.datetime.strftime("%Y%m%d") if item.datetime else "unknown"
        cloud = item.properties.get("eo:cloud_cover", 0)
        scene_dir = S2_DIR / f"{date}_{item.id[-6:]}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Scene: {date}  cloud={cloud:.2f}%  -> {scene_dir.name}/")

        for band in S2_BANDS:
            if band not in item.assets:
                print(f"    [warn] band {band} not in assets, skipping")
                continue
            asset = item.assets[band]
            dest  = scene_dir / f"{band}.tif"
            _download(asset.href, dest, band)

        # Save scene metadata
        meta_path = scene_dir / "metadata.txt"
        if not meta_path.exists():
            with open(meta_path, "w") as fh:
                fh.write(f"scene_id    : {item.id}\n")
                fh.write(f"date        : {date}\n")
                fh.write(f"cloud_cover : {cloud:.2f}%\n")
                fh.write(f"bbox        : {item.bbox}\n")
                fh.write(f"epsg        : {item.properties.get('proj:epsg', 'N/A')}\n")
                fh.write(f"downloaded  : {datetime.now().isoformat()}\n")

        downloaded_dirs.append(scene_dir)
        print()

    return downloaded_dirs


# ---------------------------------------------------------------------------
# Step 3 — Download DEM (Copernicus GLO-30)
# ---------------------------------------------------------------------------
def download_dem() -> None:
    _divider("=")
    print("  STEP 3 - Download DEM  (Copernicus GLO-30, ~30 m)")
    _divider()
    print(f"  Collection : cop-dem-glo-30")
    print(f"  AOI        : {AOI_BBOX}")
    print()

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=AOI_BBOX,
    )

    dem_items = list(search.items())
    print(f"  DEM tiles found: {len(dem_items)}")

    DEM_DIR.mkdir(parents=True, exist_ok=True)
    for item in dem_items:
        asset = item.assets.get("data")
        if not asset:
            continue
        dest = DEM_DIR / f"{item.id}.tif"
        print(f"  Tile: {item.id}")
        _download(asset.href, dest, "DEM")

    print()


# ---------------------------------------------------------------------------
# Step 4 — Summary report
# ---------------------------------------------------------------------------
def summary_report(items: list, downloaded_dirs: list) -> None:
    _divider("=")
    print("  DOWNLOAD SUMMARY")
    _divider()
    print(f"  Total scenes available (< {MAX_CLOUD}% cloud): {len(items)}")
    print(f"  Scenes downloaded      : {len(downloaded_dirs)}")
    print()

    total_bytes = 0
    for scene_dir in downloaded_dirs:
        scene_bytes = sum(f.stat().st_size for f in scene_dir.glob("*.tif"))
        total_bytes += scene_bytes
        date = scene_dir.name[:8]
        print(f"  {date}  {scene_dir.name:40s}  {_mb(scene_bytes):>8}")

    dem_bytes = sum(f.stat().st_size for f in DEM_DIR.glob("*.tif")) if DEM_DIR.exists() else 0
    total_bytes += dem_bytes
    print(f"  {'DEM tiles':<48s}  {_mb(dem_bytes):>8}")

    print()
    print(f"  Total downloaded : {_mb(total_bytes)}")
    print(f"  S2 output dir    : {S2_DIR}")
    print(f"  DEM output dir   : {DEM_DIR}")
    _divider("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    _divider("=")
    print("  Land Use Classification - Data Download")
    print("  Phoenix, AZ  |  Sentinel-2 L2A  |  Copernicus DEM")
    _divider("=")
    print()

    try:
        items = search_sentinel2()

        if not items:
            print("  No scenes found matching criteria.  Exiting.")
            sys.exit(0)

        print()
        downloaded_dirs = download_sentinel2(items)
        download_dem()
        summary_report(items, downloaded_dirs)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not reach Planetary Computer. Check internet connection.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        raise


if __name__ == "__main__":
    main()
