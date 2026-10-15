# Drone Landcover Segmentation Dataset Guide

## 1. Overview & Geographic Context

This dataset comprises high-resolution aerial and Unmanned Aerial Vehicle (UAV) imagery captured over **Butwal Ward No. 6**, located in the Rupandehi District of Lumbini Province, Nepal. The dataset was collected to facilitate automated remote sensing and semantic segmentation of urban and peri-urban landcover in developing Himalayan and Terai landscapes.

| Parameter | Specification |
|---|---|
| **Location** | Butwal Ward No. 6, Rupandehi, Nepal |
| **Platform** | Unmanned Aerial Vehicle (UAV) / Drone Survey |
| **Product** | High-Resolution Orthomosaics (.tif) & Vector Polygons (.shp) |
| **Spatial Resolution / GSD** | ~3 to 5 cm / pixel |
| **Coordinate Reference System** | WGS 84 / UTM Zone 44N (EPSG: 32644) |
| **Spectral Bands** | Red, Green, Blue (RGB) / Multi-spectral bands |
| **Patch Dimensions** | 512 x 512 pixels (or 513 x 513 pixels for DeepLabV3 alignment) |

---

## 2. Landcover Class Taxonomy

The segmentation task categorizes every pixel into one of three primary landcover classes:

| Class ID | Class Name | Hex Color | RGB | Description |
|:---:|:---|:---:|:---:|:---|
| **0** | **Background / Non-target** | `#000000` | `(0, 0, 0)` | Roads, pedestrian walkways, barren soil, parking lots, water features, and unclassified terrain. |
| **1** | **Buildings / Built-Up** | `#FF0000` | `(255, 0, 0)` | Residential buildings, commercial establishments, rooftops (tin, concrete, tiled), and infrastructure. |
| **2** | **Woodland / Vegetation** | `#00FF00` | `(0, 255, 0)` | Tree canopies, dense vegetation, urban forestry, and agricultural plots. |

---

## 3. Recommended Directory Structure

To ensure full compatibility with the automated training and inference pipeline, arrange your data according to this structure:

```text
data/
├── README.md                          # This documentation file
├── verify_dataset.py                  # Dataset integrity and statistics validator
├── generate_sample_data.py            # Generates synthetic sample tiles for testing
├── annotation image example.png       # Ground truth polygon annotation preview
├── correspondig image and mask.png    # Paired aerial patch and segmentation mask
├── gis data screenshot example.png    # ArcGIS vector digitizing interface
│
├── raw/                               # (Optional) Raw un-tiled GIS files (ignored by git)
│   ├── orthomosaics/                  # Large GeoTIFF files (e.g. Mission_1.tif)
│   └── shapefiles/                    # ESRI Shapefiles (.shp, .shx, .dbf)
│
├── sample/                            # Out-of-the-box demo patches for fast testing
│   ├── images/
│   │   ├── train/                     # Sample training images
│   │   ├── val/                       # Sample validation images
│   │   └── test/                      # Sample testing images
│   └── masks/
│       ├── train/                     # Corresponding training masks
│       ├── val/                       # Corresponding validation masks
│       └── test/                      # Corresponding testing masks
│
└── patches/                           # Full tiled dataset (512x512)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/
```

---

## 4. Visual Reference & Examples

### 4.1 Vector Annotation in GIS
Ground truth polygons were digitized using ArcGIS over the raw drone orthomosaic:
![GIS Data Screenshot](file:///d:/Projects/Projects/Drone-Image-Based-Landcover-Segmentation/data/gis%20data%20screenshot%20example.png)

### 4.2 Polygon Delineation Detail
Detailed roof and canopy boundaries traced over complex urban terrain:
![Annotation Image Example](file:///d:/Projects/Projects/Drone-Image-Based-Landcover-Segmentation/data/annotation%20image%20example.png)

### 4.3 Image Patch and Raster Mask
Extracted 512x512 pixel patch alongside its corresponding rasterized ground truth mask:
![Corresponding Image and Mask](file:///d:/Projects/Projects/Drone-Image-Based-Landcover-Segmentation/data/correspondig%20image%20and%20mask.png)

---

## 5. Dataset Preparation & Processing Pipeline

### Step 1: Shapefile Rasterization
Using `utils.py` or `rasterio`, shapefile polygons (`.shp`) are rasterized into aligned single-band or multi-band GeoTIFFs where pixel values match the Class IDs:
```python
from utils import rasterize_masks
mask_array = rasterize_masks(image_meta, intersecting_masks, height, width)
```

### Step 2: Sliding Window Patch Extraction
Gigapixel orthomosaics cannot be fed directly into deep neural networks. They are sliced into 512x512 tiles with a 256-pixel stride (50% overlap for boundary continuity):
```bash
python preprocess.py --mode extract --patch-size 512 --stride 256
```

### Step 3: Filtering Low-Information Patches
Patches where background/unclassified pixels exceed 50% or contain pure nodata borders are pruned to prevent class imbalance:
```bash
python preprocess.py --mode clean --images-dir data/patches/images --masks-dir data/patches/masks --empty-threshold 0.50
```

### Step 4: Partitioning Splits
Divide clean tiles into 70% Train, 15% Validation, and 15% Test sets:
```bash
python preprocess.py --mode split --dataset-dir data/patches --output-dir data/splits
```

---

## 6. Dataset Validation

Run the built-in dataset validator to verify that all image-mask pairs are properly matched, dimensions are consistent, and check class frequency distributions:

```bash
# Validate sample data
python data/verify_dataset.py --data-dir data/sample

# Validate full dataset
python data/verify_dataset.py --data-dir data/patches
```
