import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, label
import geopandas as gpd
from shapely.geometry import MultiPoint

IN_PATH = "outputs/dinov2_structural_score.tif"
OUT_GEO = "outputs/dinov2_hotspots.geojson"

def main():
    with rasterio.open(IN_PATH) as src:
        score = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    score = gaussian_filter(np.nan_to_num(score, nan=0.0), sigma=3)

    thr = np.percentile(score, 97)
    mask = score > thr
    labeled, n = label(mask)

    polys = []
    for rid in range(1, n + 1):
        ys, xs = np.where(labeled == rid)
        if len(xs) < 80:
            continue
        pts = [transform * (int(x), int(y)) for x, y in zip(xs, ys)]
        polys.append(MultiPoint(pts).convex_hull)

    gdf = gpd.GeoDataFrame({"thr": [float(thr)] * len(polys)}, geometry=polys, crs=crs)
    gdf.to_file(OUT_GEO, driver="GeoJSON")

    print("Saved:", OUT_GEO)

if __name__ == "__main__":
    main()
