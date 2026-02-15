import os
import numpy as np
import rasterio

IN_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def read_single(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile

def write_single(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def main():
    red, profile = read_single(os.path.join(IN_DIR, "red.tif"))
    nir, _ = read_single(os.path.join(IN_DIR, "nir.tif"))

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)

    # Mask nodata zeros (from fill_value=0.0)
    ndvi[(red == 0) | (nir == 0)] = np.nan

    # Normalize NDVI into 0..1 using robust percentiles
    lo, hi = np.nanpercentile(ndvi, [2, 98])
    ndvi_norm = (ndvi - lo) / (hi - lo + 1e-6)
    ndvi_norm = np.clip(ndvi_norm, 0, 1)

    # Spectral anomaly proxy (lower NDVI => higher anomaly)
    spectral_score = 1.0 - ndvi_norm

    write_single(os.path.join(OUT_DIR, "ndvi.tif"), ndvi, profile)
    write_single(os.path.join(OUT_DIR, "spectral_score.tif"), spectral_score, profile)

    print("Saved:")
    print(" - outputs/ndvi.tif")
    print(" - outputs/spectral_score.tif")

if __name__ == "__main__":
    main()
