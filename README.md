# NDVI-Guided RGB Segmentation

NDVI-guided pseudo-label generation and deep learning segmentation pipeline for **RGB-only land-cover mapping at urban scale**.

This project explores how vegetation indices (NDVI) can be used to generate pseudo-labels from multispectral imagery, which are then used to train and evaluate multiple semantic segmentation models using RGB inputs only.

---

## 📌 Project Motivation

High-resolution RGB imagery is widely available, while multispectral data (with NIR) is often limited or costly.  
This project investigates whether **NDVI-guided pseudo-labels** can supervise deep learning models that operate **only on RGB**, enabling scalable urban land-cover mapping.

---

## 🧠 Land-Cover Classes

The pipeline predicts four semantic classes:

| Class ID | Class Name |
|--------|------------|
| 0 | Water |
| 1 | Impervious / Bare |
| 2 | Sparse Vegetation |
| 3 | Dense Vegetation |

---

## 🧱 Repository Structure

```text
ndvi-guided-rgb-segmentation/
│
├── notebooks/
│   ├── Label_Generation_NDVI_NDWI.ipynb
│   ├── Tiling_512x512.ipynb
│   ├── Training_Models.ipynb
│   └── Results_&_Comparison.ipynb
│
├── ndvi-data-utils/
│   └── labeling utilities & helper functions
│
├── NDVI_NDWI_experimental_full_pipeline.ipynb
├── README.md
└── .gitignore
