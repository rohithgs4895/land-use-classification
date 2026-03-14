# Geospatial Land Use Classification — Phoenix Metro

Automated **Land Use / Land Cover (LULC)** classification of the Phoenix, Arizona metropolitan area using multispectral Sentinel-2 satellite imagery, terrain analysis, and ensemble machine learning. The pipeline covers a **1,394 km²** study area at **10 m/pixel** resolution and achieves **95.81% overall accuracy**.

## 🌐 Live Dashboard
[![ArcGIS Dashboard](https://img.shields.io/badge/ArcGIS-Live%20Dashboard-0079C1?style=for-the-badge&logo=arcgis&logoColor=white)](https://www.arcgis.com/apps/dashboards/4935411dbec94ceb93a4bfa4506ac5e6)

**Live Demo:** https://www.arcgis.com/apps/dashboards/4935411dbec94ceb93a4bfa4506ac5e6

---

## Purpose

This project demonstrates an end-to-end geospatial ML pipeline — from raw satellite imagery download through feature engineering, model training, full-scene inference, and ArcGIS-ready output generation — targeting urban planning, environmental monitoring, and remote sensing research applications.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Satellite imagery | Sentinel-2 L2A — ESA / Microsoft Planetary Computer |
| Terrain data | Copernicus GLO-30 DEM (~30 m, resampled to 10 m) |
| Geospatial I/O | Rasterio · GeoPandas · Shapely · PyProj |
| Machine learning | Scikit-learn — Random Forest · HistGradientBoosting |
| GIS integration | ArcGIS Pro / ArcPy |
| Visualisation | Matplotlib |
| Environment | Conda (`arcgis-dl`) · Python 3.11 |

---

## Model Performance

| Metric | Random Forest | **Gradient Boosting** |
|--------|--------------|----------------------|
| Overall Accuracy | 93.56% | **95.81%** |
| Weighted F1 | 93.56% | **95.82%** |
| Cohen's Kappa | 0.916 | **0.946** |

**Best model:** HistGradientBoostingClassifier (saved as `models/best_model.pkl`)

### Per-Class Performance — Gradient Boosting (test set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Agricultural | 94.8% | 93.6% | 94.2% |
| Bare_Soil | 95.8% | 95.3% | 95.5% |
| Industrial | 94.2% | 96.4% | 95.3% |
| Urban | 91.6% | 95.6% | 93.6% |
| Vegetation | 98.1% | 96.7% | 97.4% |
| Water | 95.4% | 95.6% | 95.5% |

---

## Study Area

| Parameter | Value |
|-----------|-------|
| Location | Phoenix Metro, Arizona, USA |
| Bounding box | lon `[-112.08, -111.70]` · lat `[33.35, 33.71]` |
| CRS | EPSG:32612 (UTM Zone 12N) |
| Raster grid | 3,517 × 3,966 px @ 10 m/pixel |
| Total area | **1,394 km²** |
| Imagery date | June – August 2023 · < 5% cloud cover |
| Training pixels | 265,284 (28 hand-drawn polygons) |

---

## Six Land Use Classes

| ID | Class | Description | Map Colour |
|----|-------|-------------|-----------|
| 0 | **Agricultural** | Irrigated crops, citrus farms, alfalfa fields | Yellow |
| 1 | **Bare_Soil** | Sonoran desert, sandy washes, caliche flats | Tan |
| 2 | **Industrial** | Warehouses, airpark, light-industrial corridors | Purple |
| 3 | **Urban** | Roads, buildings, commercial, residential | Red |
| 4 | **Vegetation** | Parks, golf courses, riparian zones, preserves | Green |
| 5 | **Water** | Tempe Town Lake, Salt River, canals | Blue |

### Area Coverage — Phoenix Study Area

| Class | Area (km²) | Coverage |
|-------|-----------|----------|
| Bare_Soil | 711.3 | 51.0% |
| Urban | 199.5 | 14.3% |
| Water | 221.0 | 15.8% |
| Industrial | 117.7 | 8.4% |
| Vegetation | 86.8 | 6.2% |
| Agricultural | 58.6 | 4.2% |

---

## 29 Engineered Features

### Raw Spectral Bands (6)
Sentinel-2 surface reflectance (DN ÷ 10,000):
`B02` `B03` `B04` `B08` `B11` `B12`
Blue · Green · Red · NIR · SWIR-1 · SWIR-2

### Spectral Indices (9)

| Index | Formula | Target |
|-------|---------|--------|
| NDVI | (NIR−Red) / (NIR+Red) | Vegetation |
| NDBI | (SWIR1−NIR) / (SWIR1+NIR) | Built-up |
| NDWI | (Green−NIR) / (Green+NIR) | Water |
| SAVI | 1.5·(NIR−Red) / (NIR+Red+0.5) | Soil-adjusted veg |
| BSI | (SWIR1+Red−NIR−Blue) / (SWIR1+Red+NIR+Blue) | Bare soil |
| EVI | 2.5·(NIR−Red) / (NIR+6·Red−7.5·Blue+1) | Enhanced veg |
| MNDWI | (Green−SWIR1) / (Green+SWIR1) | Modified water |
| UI | (SWIR2−NIR) / (SWIR2+NIR) | Urban index |
| NBI | Red · SWIR1 / NIR | New built-up |

### Band Ratios (3)
`RedGreen` · `SR` (Simple Ratio) · `SWIRRatio`

### Terrain Features from GLO-30 DEM (8)
`Elevation` · `Slope` · `Aspect_sin` · `Aspect_cos` · `PlanCurv` · `ProfCurv` · `TRI` · `TPI`

### Texture Features — 7 × 7 local window (3)
`NDVI_std` · `NIR_mean` · `NIR_std`

---

## Project Structure

```
land-use-classification/
├── data/
│   ├── raw/                        Sentinel-2 scenes + DEM tiles  [gitignored]
│   ├── processed/                  Training polygon shapefiles + GeoJSON
│   └── features/                   training_features.csv (265k rows × 31 cols)
├── models/
│   ├── best_model.pkl              HistGradientBoostingClassifier  [gitignored]
│   ├── random_forest.pkl           RandomForestClassifier (324 MB)  [gitignored]
│   ├── gradient_boosting.pkl       HistGradientBoostingClassifier  [gitignored]
│   └── scaler.pkl                  Fitted StandardScaler  [gitignored]
├── outputs/
│   ├── feature_stack.tif           29-band feature raster (2 GB)  [gitignored]
│   ├── landuse_fixed_final.tif     Final classified raster  [gitignored]
│   ├── landuse_colored.png         RGB colour map with legend
│   ├── landuse_polygons.shp/       Dissolved class polygons (ArcGIS-ready)
│   ├── landuse_summary.csv         Area statistics per class
│   ├── confusion_matrix_*.png      Normalised confusion matrices
│   ├── feature_importance.png      Top-20 feature importances
│   └── phoenix_false_color.png     Sentinel-2 false colour composite
├── src/
│   ├── download_data.py            STAC search + Sentinel-2 / DEM download
│   ├── feature_engineering.py      Build 29-band feature_stack.tif
│   ├── create_training_samples.py  Define training polygons + extract pixels
│   ├── train_classifier.py         Train RF + GB, evaluate, save best model
│   ├── predict_landuse.py          Full-scene prediction + all outputs
│   └── apply_water_mask.py         Post-processing geographic correction mask
├── notebooks/                      Jupyter notebooks (exploration)
├── arcgis/                         ArcGIS Pro / ArcPy integration scripts
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and Create Environment

```bash
git clone https://github.com/rohithgs4895/land-use-classification.git
cd land-use-classification

conda create -n arcgis-dl python=3.11
conda activate arcgis-dl
pip install -r requirements.txt
```

> **ArcGIS Pro users:** use the existing `arcgis-dl` conda clone directly.

### 2. Planetary Computer Access

A free account is required to download Sentinel-2 imagery:

```bash
pip install planetary-computer
planetarycomputer authenticate
```

---

## Run the Pipeline

Execute scripts in order from the project root:

```bash
# 1. Download Sentinel-2 imagery and Copernicus DEM
conda run -n arcgis-dl python src/download_data.py

# 2. Build 29-band feature stack (~2 GB GeoTIFF, 5–10 min)
conda run -n arcgis-dl python src/feature_engineering.py

# 3. Define training polygons and extract labeled pixel features
conda run -n arcgis-dl python src/create_training_samples.py

# 4. Train Random Forest + Gradient Boosting, evaluate, save best model
conda run -n arcgis-dl python src/train_classifier.py

# 5. Full-scene prediction + coloured map + vector polygons + area stats
conda run -n arcgis-dl python src/predict_landuse.py

# 6. Apply geographic correction mask (post-processing)
conda run -n arcgis-dl python src/apply_water_mask.py
```

| Script | Approx. Runtime |
|--------|----------------|
| download_data.py | 10–30 min (network) |
| feature_engineering.py | 5–10 min |
| create_training_samples.py | 2–3 min |
| train_classifier.py | ~1 min |
| predict_landuse.py | ~40 min |
| apply_water_mask.py | < 5 s |

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/landuse_fixed_final.tif` | Final corrected raster — uint8, NODATA=255 |
| `outputs/landuse_colored.png` | Publication-ready colour map with legend |
| `outputs/landuse_polygons.shp` | 6 dissolved class polygons, ArcGIS-ready |
| `outputs/landuse_summary.csv` | Area (km²) and coverage (%) per class |

---

## Future Enhancements

- **Temporal compositing** — multi-date Sentinel-2 mosaics to reduce cloud and shadow artefacts
- **Deep learning segmentation** — U-Net or SegFormer for end-to-end pixel classification
- **Additional classes** — split Urban into residential / commercial / road; add Riparian
- **Ground truth validation** — cross-reference with NLCD 2021 or field survey data
- **Confidence mapping** — per-pixel probability rasters for uncertainty quantification
- **Change detection** — annual LULC comparisons to track urban sprawl and desert loss
- **ArcGIS Online** — publish classified raster and vector polygons as hosted tile layers
- **Expanded AOI** — scale to the full Phoenix–Mesa–Scottsdale MSA (~14,000 km²)

---

## References

- [ESA Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/)
- [Scikit-learn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GeoPandas Documentation](https://geopandas.org/)
