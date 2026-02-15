# 🌿 Amazonia
## AI-Powered Digital Excavation for the Amazon

Ghost Canopy is a geospatial AI platform designed to surface hidden anthropogenic patterns beneath dense Amazon rainforest canopy using satellite imagery and multimodal anomaly detection.

The project combines remote sensing, self-supervised computer vision, and interactive web mapping into a full-stack prototype.

---

# 🚀 What This Project Demonstrates

Ghost Canopy includes:

- A complete AI anomaly detection pipeline
- A working Streamlit prototype (full AI capability)
- A production-oriented Django web application
- An asynchronous job system
- Interactive geospatial decision support workflow

---

# 🧠 AI Pipeline (Fully Implemented Prototype)

The AI prototype performs:

### 1️⃣ Satellite Retrieval
- Sentinel-2 L2A imagery via STAC (Microsoft Planetary Computer)
- Cloud filtering and best-scene selection
- Raster stacking using `stackstac`

### 2️⃣ Spectral Anomaly (NDVI)
- NDVI-based vegetation anomaly detection
- Highlights unusual canopy patterns that may correlate with anthropogenic soil variation (e.g., Terra Preta)

### 3️⃣ Structural Anomaly (DINOv2)
- Self-supervised Vision Transformer (`dinov2_vits14`)
- Patch embedding extraction
- Cosine anomaly scoring
- Surfaces subtle geometric patterns beneath canopy

### 4️⃣ Fusion Layer
- Weighted fusion of spectral + structural signals
- Smoothed confidence surface

### 5️⃣ Hotspot Extraction & Ranking
- Thresholding + connected components
- Polygon generation
- Ranked candidate site list (CSV + GeoJSON)

### Outputs
- `fused_score.tif`
- `fused_hotspots.geojson`
- `candidates.csv`

The Streamlit dashboard visualizes:
- Confidence heatmap overlay
- Ranked candidate sites
- Hotspot polygons
- Interactive blending controls

---
<img width="1915" height="972" alt="image" src="https://github.com/user-attachments/assets/bb618fe8-091d-4647-8811-7cf88e4a0b32" />

# 🌐 Django Web Application (Production-Oriented Architecture)

In parallel to the AI prototype, a Django-based web application was built to represent the scalable deployment architecture.

## ✅ Implemented Features

### Interactive AOI Selection
- Leaflet map centered on the Amazon
- Leaflet Draw (polygon + rectangle)
- Single AOI constraint
- Analyze button enabled only when AOI exists

### Asynchronous Analyze Workflow
When the user clicks **Analyze**:

- Django receives POST request
- Unique `job_id` generated
- `media/<job_id>/` folder created
- `payload.json` and `status.json` written
- Background thread launched
- Browser remains responsive

This simulates a production async job system without Celery.

### Live Status Polling
- `/status/<job_id>/` endpoint
- Frontend polls every 1.5 seconds
- Status states:
  - queued
  - running
  - done
  - error
- Page auto-refreshes on completion

### Result Rendering
After job completion:
- `meta.json` parsed
- Hotspots rendered as map markers
- Map zooms to AOI bounding box
- Clicking a hotspot displays:
  - Priority
  - Anomaly score
  - Explanation list

A disclaimer is included to ensure responsible interpretation (“Decision support only”).

---

# 📌 Current Integration Status

- ✅ NDVI anomaly detection integrated into Django workflow  
- 🚧 DINOv2 structural anomaly implemented in AI prototype  
- 🚧 Fusion layer implemented in prototype  
- 🔄 Full DINO + fusion integration into Django backend planned but not completed within hackathon timeframe  

The Streamlit interface demonstrates the full AI capability, while the Django project represents the intended scalable production architecture.

---


---

# 🛠 Tech Stack

- Python
- Sentinel-2 (STAC / Planetary Computer)
- Rasterio / Stackstac
- NumPy / SciPy
- DINOv2 (PyTorch)
- GeoPandas / Shapely
- Streamlit + Folium
- Django + Leaflet

---

# 🎯 Vision

Ghost Canopy aims to reduce the archaeological and environmental verification gap by:

- Surfacing hidden signals at scale
- Ranking candidate regions
- Providing interactive decision support
- Enabling scalable backend deployment

This repository demonstrates both:
- A validated AI proof-of-concept
- A production-oriented web architecture
