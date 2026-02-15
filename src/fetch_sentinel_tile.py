import os
import numpy as np
import rasterio

from pystac_client import Client
import planetary_computer as pc
import stackstac

# =========================================================
# CONFIG: small Amazon bbox (minLon, minLat, maxLon, maxLat)
# =========================================================
BBOX = [-60.2, -3.2, -60.0, -3.0]          # later: user-click AOI
DATE_RANGE = "2026-01-01/2026-02-14"
MAX_CLOUD = 30

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def choose_best_item(items):
    # lowest cloud, then newest
    def key(it):
        cloud = it.properties.get("eo:cloud_cover", 999)
        dt = it.datetime
        return (cloud, -dt.timestamp())
    return sorted(items, key=key)[0]

def write_singleband_geotiff(path, arr_2d, transform, crs):
    h, w = arr_2d.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",   # save as float32 on disk (fine)
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_2d.astype(np.float32), 1)

def write_rgb_geotiff(path, rgb_3hw_uint8, transform, crs):
    _, h, w = rgb_3hw_uint8.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb_3hw_uint8.astype(np.uint8))

def robust_to_uint8(x):
    x = x.astype(np.float64)

    # treat zeros as nodata for display scaling
    x = np.where(x == 0.0, np.nan, x)

    if np.all(np.isnan(x)):
        return np.zeros_like(x, dtype=np.uint8)

    lo, hi = np.nanpercentile(x, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-10:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo + 1e-10)
    y = np.clip(y, 0, 1)
    y = np.nan_to_num(y, nan=0.0)
    return (y * 255).astype(np.uint8)

def main():
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=50,
    )

    items = list(search.items())
    if not items:
        raise SystemExit("No Sentinel-2 items found. Try wider DATE_RANGE or higher MAX_CLOUD.")

    best = choose_best_item(items)
    best = pc.sign(best)

    print("Chosen item datetime:", best.datetime)
    print("Cloud cover:", best.properties.get("eo:cloud_cover"))
    print("Item ID:", best.id)

    assets = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

    # ✅ WORKAROUND: use float64 so stackstac accepts fill_value
    data = stackstac.stack(
        [best],
        assets=assets,
        resolution=10,
        bounds_latlon=BBOX,
        epsg=3857,            # meters-based CRS
        chunksize=2048,
        dtype="float64",      # ✅ key change
        fill_value=0.0,       # ✅ float fill value
        rescale=False,
    ).compute()

    band_data = data[0].values  # (4, H, W)
    b02, b03, b04, b08 = band_data

    zero_ratio = float(np.mean(b04 == 0.0))
    print(f"Zero ratio (Red band): {zero_ratio:.3f}")
    if zero_ratio > 0.95:
        raise SystemExit(
            "Almost everything is NoData (zeros). Try:\n"
            "- slightly larger BBOX (e.g. [-60.25,-3.25,-59.95,-2.95])\n"
            "- wider DATE_RANGE\n"
            "- or different Amazon location"
        )

    crs = data.crs
    transform = data.transform

    # Save red + nir as float32 GeoTIFFs
    write_singleband_geotiff(os.path.join(OUT_DIR, "red.tif"), b04, transform, crs)
    write_singleband_geotiff(os.path.join(OUT_DIR, "nir.tif"), b08, transform, crs)

    # Save RGB for visualization
    rgb_u8 = np.stack(
        [robust_to_uint8(b04), robust_to_uint8(b03), robust_to_uint8(b02)],
        axis=0
    )
    write_rgb_geotiff(os.path.join(OUT_DIR, "rgb.tif"), rgb_u8, transform, crs)

    print("\nSaved files:")
    print(" - data/rgb.tif")
    print(" - data/red.tif")
    print(" - data/nir.tif")

if __name__ == "__main__":
    main()
