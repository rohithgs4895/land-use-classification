"""
publish_to_arcgis.py
--------------------
Publishes land-use classification outputs to ArcGIS Online as hosted items.

Publishes:
  outputs/landuse_polygons.shp  → hosted feature layer  "LUC_Phoenix_LandUse_Polygons"
  outputs/landuse_summary.csv   → hosted table           "LUC_Phoenix_Summary"

The shapefile is reprojected from EPSG:32612 (UTM Zone 12N) → EPSG:4326 (WGS84)
before zipping, as ArcGIS Online hosted feature layers require geographic coordinates.

Credentials are read from .env:
  ARCGIS_USERNAME=...
  ARCGIS_PASSWORD=...
"""

import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUTS = Path(__file__).parent.parent / "outputs"

ARCGIS_URL = "https://www.arcgis.com"

# ── Item definitions ──────────────────────────────────────────────────────────

POLYGON_LAYER = {
    "shapefile":   "landuse_polygons",
    "title":       "LUC_Phoenix_LandUse_Polygons",
    "tags":        "land use, classification, Phoenix, Sentinel-2, machine learning, random forest",
    "description": (
        "Land-use classification polygons for the Phoenix, AZ metro area derived from "
        "Sentinel-2 multispectral imagery (August 2023) using a Random Forest classifier. "
        "Six classes: Agricultural, Bare Soil, Industrial, Urban, Vegetation, Water. "
        "Source: Sentinel-2 via Microsoft Planetary Computer + Copernicus DEM."
    ),
    "snippet": "ML-derived land-use polygons for Phoenix AZ from Sentinel-2 imagery (2023).",
}

SUMMARY_TABLE = {
    "csv":         "landuse_summary.csv",
    "title":       "LUC_Phoenix_Summary",
    "tags":        "land use, Phoenix, summary statistics, classification",
    "description": (
        "Summary statistics for the Phoenix land-use classification. "
        "Includes pixel count, area (km²), and percentage cover per class. "
        "Total study area: ~1,395 km²."
    ),
    "snippet": "Area and percentage coverage per land-use class — Phoenix AZ (2023).",
}

SHP_EXTENSIONS = [".shp", ".dbf", ".shx", ".prj", ".cpg"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def reproject_shapefile(src_path: Path, dst_path: Path, target_crs: str = "EPSG:4326"):
    """Reproject a shapefile and write to dst_path (same base name, new folder)."""
    import geopandas as gpd
    gdf = gpd.read_file(src_path)
    if str(gdf.crs) != target_crs:
        print(f"    Reprojecting {gdf.crs} -> {target_crs}...")
        gdf = gdf.to_crs(target_crs)
    else:
        print(f"    CRS already {target_crs}, no reprojection needed.")
    gdf.to_file(dst_path)
    return gdf


def zip_shapefile(base_dir: Path, base_name: str, zip_path: Path) -> Path:
    """Zip all shapefile components from base_dir into zip_path."""
    included = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in SHP_EXTENSIONS:
            src = base_dir / f"{base_name}{ext}"
            if src.exists():
                zf.write(src, arcname=f"{base_name}{ext}")
                included.append(f"{base_name}{ext}")
    size_kb = zip_path.stat().st_size / 1024
    print(f"    Zipped: {zip_path.name}  ({size_kb:.1f} KB)  [{', '.join(included)}]")
    return zip_path


def delete_existing(gis, title: str, item_type_keyword: str = ""):
    """Delete any existing ArcGIS Online item with the same title."""
    results = gis.content.search(query=f'title:"{title}"')
    for item in results:
        if item.title == title:
            print(f"    Removing existing item: {item.id} ({item.type})")
            try:
                item.delete()
            except Exception as e:
                print(f"    Warning: could not delete {item.id}: {e}")


# ── Publish functions ─────────────────────────────────────────────────────────

def publish_polygon_layer(gis) -> dict:
    """Reproject, zip, upload, and publish the land-use polygon shapefile."""
    print("\n  [1/2] Publishing polygon feature layer: LUC_Phoenix_LandUse_Polygons")

    src_shp  = OUTPUTS / "landuse_polygons.shp"
    wgs84_dir = OUTPUTS / "_wgs84"
    wgs84_dir.mkdir(exist_ok=True)
    wgs84_shp = wgs84_dir / "landuse_polygons.shp"
    zip_path  = OUTPUTS / "landuse_polygons.zip"

    # Step 1 — Reproject to WGS84
    reproject_shapefile(src_shp, wgs84_shp)

    # Step 2 — Zip WGS84 components
    zip_shapefile(wgs84_dir, "landuse_polygons", zip_path)

    # Step 3 — Remove any pre-existing item
    delete_existing(gis, POLYGON_LAYER["title"])

    # Step 4 — Upload as Shapefile item (use Folder.add() per arcgis 2.3+ API)
    print(f"    Uploading {zip_path.name}...", end=" ", flush=True)
    root_folder = gis.content.folders.get()
    shp_item = root_folder.add(
        item_properties={
            "title":       POLYGON_LAYER["title"],
            "type":        "Shapefile",
            "tags":        POLYGON_LAYER["tags"],
            "description": POLYGON_LAYER["description"],
            "snippet":     POLYGON_LAYER["snippet"],
        },
        file=str(zip_path),
    ).result()
    print(f"uploaded (id: {shp_item.id})")

    # Step 5 — Publish as hosted feature layer
    print(f"    Publishing as hosted feature layer...", end=" ", flush=True)
    fl_item = shp_item.publish(overwrite=True)
    print("done")

    # Step 6 — Share publicly
    fl_item.share(everyone=True)
    print(f"    Shared publicly.")

    portal_url = f"https://www.arcgis.com/home/item.html?id={fl_item.id}"
    print(f"    Item ID  : {fl_item.id}")
    print(f"    Layer URL: {fl_item.url}")
    print(f"    Portal   : {portal_url}")

    return {
        "title":       fl_item.title,
        "type":        fl_item.type,
        "item_id":     fl_item.id,
        "service_url": fl_item.url,
        "portal_url":  portal_url,
    }


def publish_summary_table(gis) -> dict:
    """Upload landuse_summary.csv and publish as a hosted table."""
    print("\n  [2/2] Publishing summary table: LUC_Phoenix_Summary")

    csv_path = OUTPUTS / SUMMARY_TABLE["csv"]

    # Step 1 — Remove any pre-existing item
    delete_existing(gis, SUMMARY_TABLE["title"])

    # Step 2 — Upload CSV item (use Folder.add() per arcgis 2.3+ API)
    print(f"    Uploading {csv_path.name}...", end=" ", flush=True)
    root_folder = gis.content.folders.get()
    csv_item = root_folder.add(
        item_properties={
            "title":       SUMMARY_TABLE["title"],
            "type":        "CSV",
            "tags":        SUMMARY_TABLE["tags"],
            "description": SUMMARY_TABLE["description"],
            "snippet":     SUMMARY_TABLE["snippet"],
        },
        file=str(csv_path),
    ).result()
    print(f"uploaded (id: {csv_item.id})")

    # Step 3 — Publish as hosted table (no overwrite — this is a fresh item)
    print(f"    Publishing as hosted table...", end=" ", flush=True)
    table_item = csv_item.publish(
        publish_parameters={
            "name":            SUMMARY_TABLE["title"],
            "locationType":    "none",
            "columnDelimiter": ",",
            "qualifier":       '"',
        },
    )
    print("done")

    # Step 4 — Share publicly
    table_item.share(everyone=True)
    print(f"    Shared publicly.")

    portal_url = f"https://www.arcgis.com/home/item.html?id={table_item.id}"
    print(f"    Item ID  : {table_item.id}")
    print(f"    Portal   : {portal_url}")

    return {
        "title":       table_item.title,
        "type":        table_item.type,
        "item_id":     table_item.id,
        "service_url": getattr(table_item, "url", "N/A"),
        "portal_url":  portal_url,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list):
    print("\n" + "=" * 65)
    print("PUBLISH COMPLETE")
    print("=" * 65)
    for r in results:
        print(f"\n  {r['title']}  [{r['type']}]")
        print(f"    Item ID : {r['item_id']}")
        if r.get("service_url") and r["service_url"] != "N/A":
            print(f"    REST URL: {r['service_url']}")
        print(f"    Portal  : {r['portal_url']}")
    print()


def save_item_ids(results: list):
    """Write item IDs to arcgis/published_layers.txt for later reference."""
    ref_dir  = Path(__file__).parent.parent / "arcgis"
    ref_dir.mkdir(exist_ok=True)
    ref_file = ref_dir / "published_layers.txt"
    with open(ref_file, "w") as f:
        f.write("Land-Use Classification — Published ArcGIS Items\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"Title    : {r['title']}\n")
            f.write(f"Type     : {r['type']}\n")
            f.write(f"Item ID  : {r['item_id']}\n")
            if r.get("service_url") and r["service_url"] != "N/A":
                f.write(f"REST URL : {r['service_url']}\n")
            f.write(f"Portal   : {r['portal_url']}\n\n")
    print(f"Item IDs saved -> {ref_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Land-Use Classification — ArcGIS Online Publisher")
    print("=" * 65)

    # Validate output files exist
    for path in [OUTPUTS / "landuse_polygons.shp", OUTPUTS / "landuse_summary.csv"]:
        if not path.exists():
            print(f"ERROR: Missing output file: {path}")
            sys.exit(1)

    # Import arcgis
    try:
        from arcgis.gis import GIS
    except ImportError:
        print("ERROR: arcgis package not installed. Run: pip install arcgis")
        sys.exit(1)

    # Credentials from .env
    username = os.getenv("ARCGIS_USERNAME", "").strip()
    password = os.getenv("ARCGIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: ARCGIS_USERNAME and ARCGIS_PASSWORD must be set in .env")
        sys.exit(1)

    # Connect
    print(f"\nConnecting to ArcGIS Online as '{username}'...")
    try:
        gis = GIS(ARCGIS_URL, username, password)
        user = gis.properties.user
        print(f"Connected : {user.fullName} ({user.username})")
        print(f"Org       : {gis.properties.name}")
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        sys.exit(1)

    # Publish
    results = []
    for publish_fn in [publish_polygon_layer, publish_summary_table]:
        try:
            results.append(publish_fn(gis))
        except Exception as e:
            fn_name = publish_fn.__name__
            print(f"\n  ERROR in {fn_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "title": fn_name, "type": "FAILED",
                "item_id": "FAILED", "service_url": str(e), "portal_url": "",
            })

    print_summary(results)
    save_item_ids(results)


if __name__ == "__main__":
    main()
