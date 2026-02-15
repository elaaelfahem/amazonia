import os, io, base64
import streamlit as st
import numpy as np
import rasterio
import folium
from streamlit_folium import st_folium
from rasterio.warp import transform_bounds
from PIL import Image
import geopandas as gpd
import pandas as pd

st.set_page_config(layout="wide")
st.title("🌿 Ghost Canopy — Digital Excavation")

RGB_PATH = "data/rgb.tif"

SPECTRAL_PATH = "outputs/ghost_heatmap.tif"
SPECTRAL_HOTSPOTS = "outputs/hotspots.geojson"

DINO_PATH = "outputs/dinov2_structural_score.tif"
DINO_HOTSPOTS = "outputs/dinov2_hotspots.geojson"

FUSED_PATH = "outputs/fused_score.tif"
FUSED_HOTSPOTS = "outputs/fused_hotspots.geojson"
CANDIDATES_CSV = "outputs/candidates.csv"

if not os.path.exists(RGB_PATH):
    st.error("Missing data/rgb.tif — run: python src/fetch_sentinel_tile.py")
    st.stop()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def to_png_data_url(img_uint8_hwc: np.ndarray) -> str:
    im = Image.fromarray(img_uint8_hwc)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return png_bytes_to_data_url(buf.getvalue())

@st.cache_data(show_spinner=False)
def read_bounds_latlon(path):
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        src_crs = src.crs
    minlon, minlat, maxlon, maxlat = transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top)
    return [(minlat, minlon), (maxlat, maxlon)]

@st.cache_data(show_spinner=False)
def read_rgb_data_url(path):
    with rasterio.open(path) as src:
        rgb = src.read().astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    return to_png_data_url(rgb)

@st.cache_data(show_spinner=False)
def read_score(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr

def norm01(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

@st.cache_data(show_spinner=False)
def score_to_heat_data_url_cached(score_01: np.ndarray):
    r = (255 * score_01).astype(np.uint8)
    g = (160 * (score_01 ** 0.7)).astype(np.uint8)
    b = (40  * (1 - score_01)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return to_png_data_url(rgb)

@st.cache_data(show_spinner=False)
def load_hotspots(path):
    return gpd.read_file(path).to_crs("EPSG:4326")

@st.cache_data(show_spinner=False)
def load_candidates(path):
    return pd.read_csv(path)

# ---- Sidebar: candidates panel (Phase 3) ----
st.sidebar.header("Phase 3 — Candidates")

if os.path.exists(CANDIDATES_CSV):
    cand = load_candidates(CANDIDATES_CSV)
    cand_show = cand.copy()

    # Nice formatting
    cand_show["area_m2"] = cand_show["area_m2"].round(0).astype("int64")
    cand_show["mean_score"] = cand_show["mean_score"].round(3)
    cand_show["max_score"] = cand_show["max_score"].round(3)
    cand_show["centroid_lat"] = cand_show["centroid_lat"].round(6)
    cand_show["centroid_lon"] = cand_show["centroid_lon"].round(6)

    topk = st.sidebar.slider("Show top K sites", 3, 30, 10)
    cand_top = cand_show.head(topk)

    st.sidebar.dataframe(
        cand_top[["site_id", "max_score", "mean_score", "area_m2", "centroid_lat", "centroid_lon"]],
        use_container_width=True,
        height=360
    )

    selected_site = st.sidebar.selectbox("Zoom to site", ["(none)"] + cand_top["site_id"].tolist())
else:
    st.sidebar.info("No candidates yet. Run: python src/run_candidates.py")
    selected_site = "(none)"

# ---- Main UI controls ----
overlay_mode = st.selectbox(
    "Overlay",
    ["Spectral (NDVI)", "DINOv2 Structural", "Fused (Confidence)"]
)
alpha = st.slider("Ghost Intensity", 0.0, 1.0, 0.6)

# ---- bounds + base ----
bounds = read_bounds_latlon(RGB_PATH)
rgb_url = read_rgb_data_url(RGB_PATH)

# ---- pick overlay ----
hotspots_gdf = None

if overlay_mode == "Spectral (NDVI)":
    if not os.path.exists(SPECTRAL_PATH):
        st.error("Missing outputs/ghost_heatmap.tif — run: python src/run_hotspots.py")
        st.stop()
    heat_url = score_to_heat_data_url_cached(norm01(read_score(SPECTRAL_PATH)))
    if os.path.exists(SPECTRAL_HOTSPOTS):
        hotspots_gdf = load_hotspots(SPECTRAL_HOTSPOTS)

elif overlay_mode == "DINOv2 Structural":
    if not os.path.exists(DINO_PATH):
        st.error("Missing outputs/dinov2_structural_score.tif — run: python src/run_dinov2_structural_fast.py")
        st.stop()
    heat_url = score_to_heat_data_url_cached(norm01(read_score(DINO_PATH)))
    if os.path.exists(DINO_HOTSPOTS):
        hotspots_gdf = load_hotspots(DINO_HOTSPOTS)

else:  # Fused
    if not os.path.exists(FUSED_PATH):
        st.error("Missing outputs/fused_score.tif — run: python src/run_candidates.py")
        st.stop()
    heat_url = score_to_heat_data_url_cached(norm01(read_score(FUSED_PATH)))
    if os.path.exists(FUSED_HOTSPOTS):
        hotspots_gdf = load_hotspots(FUSED_HOTSPOTS)

# ---- Build map ----
# Default fit to AOI, but if user selected a candidate, zoom to its bbox
fit_bounds = bounds

if selected_site != "(none)" and os.path.exists(CANDIDATES_CSV):
    row = load_candidates(CANDIDATES_CSV)
    row = row[row["site_id"] == selected_site]
    if len(row) == 1:
        # bbox in projected CRS is in CSV (minx/miny/maxx/maxy). We need lat/lon bounds for folium.
        # easiest: read fused hotspots geojson which is already latlon and match site_id
        if os.path.exists(FUSED_HOTSPOTS):
            gdf_f = load_hotspots(FUSED_HOTSPOTS)
            hit = gdf_f[gdf_f["site_id"] == selected_site]
            if len(hit) == 1:
                minx, miny, maxx, maxy = hit.total_bounds  # lon/lat
                fit_bounds = [(miny, minx), (maxy, maxx)]

center = [(fit_bounds[0][0] + fit_bounds[1][0]) / 2, (fit_bounds[0][1] + fit_bounds[1][1]) / 2]
m = folium.Map(location=center, zoom_start=14, control_scale=True)

folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="CartoDB",
    name="Carto Light",
).add_to(m)

m.fit_bounds(fit_bounds)

folium.raster_layers.ImageOverlay(
    image=rgb_url,
    bounds=bounds,          # keep overlays at AOI bounds
    opacity=1.0,
    name="RGB",
    zindex=1
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=heat_url,
    bounds=bounds,          # keep overlays at AOI bounds
    opacity=alpha,
    name="Ghost Overlay",
    zindex=2
).add_to(m)

if hotspots_gdf is not None and len(hotspots_gdf) > 0:
    folium.GeoJson(
        hotspots_gdf,
        name="Hotspots",
        style_function=lambda x: {"color": "cyan", "weight": 2, "fillOpacity": 0.15},
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=1100, height=650)

st.markdown("### ✅ Phase 3 Output")
st.markdown("""
- **Fused (Confidence)** turns your two signals into one ranked decision layer.
- **Candidates** are the top hotspot regions, exported to CSV for verification workflows.
""")
