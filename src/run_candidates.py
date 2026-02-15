
import os
import csv
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, label
import geopandas as gpd
from shapely.geometry import MultiPoint
from rasterio.transform import xy as pix2xy
from pyproj import Transformer

SPECTRAL_PATH = "outputs/ghost_heatmap.tif"
DINO_PATH = "outputs/dinov2_structural_score.tif"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FUSED_TIF = os.path.join(OUT_DIR, "fused_score.tif")
FUSED_HOTSPOTS = os.path.join(OUT_DIR, "fused_hotspots.geojson")
CANDIDATES_CSV = os.path.join(OUT_DIR, "candidates.csv")

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return arr, profile, transform, crs, nodata

def write_single(path, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)

def norm01(x):
    x = np.nan_to_num(x, nan=0.0)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0, 1).astype(np.float32)

def main():
    if not os.path.exists(SPECTRAL_PATH):
        raise SystemExit("Missing outputs/ghost_heatmap.tif (run: python src/run_hotspots.py)")
    if not os.path.exists(DINO_PATH):
        raise SystemExit("Missing outputs/dinov2_structural_score.tif (run: python src/run_dinov2_structural_fast.py)")

    s, prof_s, transform, crs, _ = read_band(SPECTRAL_PATH)
    d, prof_d, transform_d, crs_d, _ = read_band(DINO_PATH)

    # Basic safety: shapes should match (they do in your pipeline)
    if s.shape != d.shape:
        raise SystemExit(f"Shape mismatch spectral {s.shape} vs dino {d.shape}")

    # Normalize each signal, then fuse
    s01 = norm01(s)
    d01 = norm01(d)

    fused = norm01(0.6 * s01 + 0.4 * d01)

    # Smooth fused for nicer clustering
    fused_s = gaussian_filter(fused, sigma=3)

    # Save fused raster
    write_single(FUSED_TIF, fused_s, prof_s)
    print("Saved:", FUSED_TIF)

    # Hotspots: top 3%
    thr = float(np.percentile(fused_s, 97))
    mask = fused_s > thr
    labeled, n = label(mask)

    # For lat/lon export
    transformer = None
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception:
        transformer = None

    polys = []
    rows = []

    # Pixel area in CRS units (EPSG:3857 => meters)
    # pixel width/height from transform
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    pixel_area = px_w * px_h

    for rid in range(1, n + 1):
        ys, xs = np.where(labeled == rid)
        if len(xs) < 100:   # ignore tiny blobs
            continue

        # Polygon hull in map coords
        pts = [transform * (int(x), int(y)) for x, y in zip(xs, ys)]
        hull = MultiPoint(pts).convex_hull
        polys.append(hull)

        # Region scores
        region_vals = fused_s[ys, xs]
        mean_score = float(np.mean(region_vals))
        max_score = float(np.max(region_vals))

        # Area (approx from pixel count)
        area_m2 = float(len(xs) * pixel_area)

        # Centroid in map coords
        cx, cy = hull.centroid.x, hull.centroid.y

        # Convert centroid to lat/lon if possible
        if transformer is not None:
            lon, lat = transformer.transform(cx, cy)
        else:
            lon, lat = (None, None)

        # Bounds in map coords
        minx, miny, maxx, maxy = hull.bounds

        rows.append({
            "site_id": f"SITE_{rid:03d}",
            "mean_score": mean_score,
            "max_score": max_score,
            "area_m2": area_m2,
            "centroid_lon": lon,
            "centroid_lat": lat,
            "minx": float(minx),
            "miny": float(miny),
            "maxx": float(maxx),
            "maxy": float(maxy),
        })

    if not polys:
        print("No hotspots found (try lowering percentile threshold).")
        # Still write empty outputs
        gdf_empty = gpd.GeoDataFrame({"site_id": [], "mean_score": [], "max_score": [], "area_m2": []},
                                     geometry=[], crs=crs)
        gdf_empty.to_file(FUSED_HOTSPOTS, driver="GeoJSON")
        with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "site_id","mean_score","max_score","area_m2","centroid_lon","centroid_lat","minx","miny","maxx","maxy"
            ])
            w.writeheader()
        print("Saved:", FUSED_HOTSPOTS)
        print("Saved:", CANDIDATES_CSV)
        return

    # Rank: max_score desc, then area desc
    rows_sorted = sorted(rows, key=lambda r: (r["max_score"], r["area_m2"]), reverse=True)

    # Save GeoJSON with attributes
    gdf = gpd.GeoDataFrame(rows_sorted, geometry=polys, crs=crs)
    gdf.to_file(FUSED_HOTSPOTS, driver="GeoJSON")
    print("Saved:", FUSED_HOTSPOTS)

    # Save CSV
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "site_id","mean_score","max_score","area_m2","centroid_lon","centroid_lat","minx","miny","maxx","maxy"
        ])
        w.writeheader()
        w.writerows(rows_sorted)
    print("Saved:", CANDIDATES_CSV)

    print(f"Candidates: {len(rows_sorted)} | Threshold (97th pct): {thr:.3f}")

if __name__ == "__main__":
    main()
