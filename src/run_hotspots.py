import os
import io
import streamlit as st
import numpy as np
import rasterio
import folium
import geopandas as gpd
from streamlit_folium import st_folium
from rasterio.warp import transform_bounds
from PIL import Image

st.set_page_config(layout="wide")
st.title("🌿 Ghost Canopy — Digital Excavation")

RGB_PATH = "data/rgb.tif"
HEAT_PATH = "outputs/ghost_heatmap.tif"
HOTSPOTS_PATH = "outputs/hotspots.geojson"

if not os.path.exists(RGB_PATH):
    st.error(f"Missing {RGB_PATH}. Run: python src/fetch_sentinel_tile.py")
    st.stop()

if not os.path.exists(HEAT_PATH):
    st.error(f"Missing {HEAT_PATH}. Run: python src/run_hotspots.py")
    st.stop()

def to_png_bytes(img_uint8_hwc):
    """HWC uint8 -> PNG bytes"""
    img = Image.fromarray(img_uint8_hwc)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def read_rgb_png_and_bounds(path):
    with rasterio.open(path) as src:
        rgb = src.read()  # (3,H,W)
        left, bottom, right, top = src.bounds
        crs = src.crs

    # Convert bounds to lat/lon for folium
    minlon, minlat, maxlon, maxlat = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    bounds_latlon = [(minlat, minlon), (maxlat, maxlon)]

    # Ensure uint8 HWC
    rgb = np.transpose(rgb, (1, 2, 0))  # HWC
    rgb = rgb.astype(np.uint8)

    return to_png_bytes(rgb), bounds_latlon

def heat_to_png_bytes(path):
    with rasterio.open(path) as src:
        heat = src.read(1).astype(np.float32)

    heat = np.nan_to_num(heat, nan=0.0)

    lo = np.percentile(heat, 2)
    hi = np.percentile(heat, 98)
    heat = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1)

    # Inferno-like colormap (manual RGB)
    # 0..1 -> (R,G,B)
    r = (255 * heat).astype(np.uint8)
    g = (150 * (heat ** 0.7)).astype(np.uint8)
    b = (50 * (1 - heat)).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)  # HWC
    return to_png_bytes(rgb)

rgb_png, bounds_latlon = read_rgb_png_and_bounds(RGB_PATH)
heat_png = heat_to_png_bytes(HEAT_PATH)

# UI slider
alpha = st.slider("Ghost Intensity", 0.0, 1.0, 0.6)

# Folium map
center = [(bounds_latlon[0][0] + bounds_latlon[1][0]) / 2,
          (bounds_latlon[0][1] + bounds_latlon[1][1]) / 2]

m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

# RGB overlay
folium.raster_layers.ImageOverlay(
    image=rgb_png,
    bounds=bounds_latlon,
    opacity=1.0,
    interactive=False,
    cross_origin=False,
    zindex=1,
).add_to(m)

# Heat overlay
folium.raster_layers.ImageOverlay(
    image=heat_png,
    bounds=bounds_latlon,
    opacity=alpha,
    interactive=False,
    cross_origin=False,
    zindex=2,
).add_to(m)

# Hotspots overlay (optional)
if os.path.exists(HOTSPOTS_PATH):
    try:
        gdf = gpd.read_file(HOTSPOTS_PATH)
        # Reproject to lat/lon for folium
        gdf = gdf.to_crs("EPSG:4326")
        folium.GeoJson(
            gdf,
            name="Hotspots",
            style_function=lambda x: {"color": "cyan", "weight": 2, "fillOpacity": 0.15},
        ).add_to(m)
    except Exception as e:
        st.warning(f"Could not load hotspots: {e}")

folium.LayerControl().add_to(m)

st_folium(m, width=1100, height=650)

st.markdown("### 🔍 Interpretation")
st.markdown("""
- **Ghost heatmap** = NDVI-derived anomaly proxy (possible Terra Preta indicator)
- **Cyan polygons** = detected hotspots
- Slider blends raw imagery ↔ ghost canopy
""")
