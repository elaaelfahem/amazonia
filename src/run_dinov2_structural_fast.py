import os
import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torchvision import transforms

IN_RGB = "data/rgb.tif"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vits14"   # fast model
MAX_SIDE = 512                 # ✅ tuned downsample target (try 512 if you want even faster)

# Optional CPU tuning (can help on Windows)
if DEVICE == "cpu":
    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

# Cache model in process (so if you import this in Streamlit later, it won't reload)
_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is None:
        print("Loading DINOv2:", MODEL_NAME, "| Device:", DEVICE)
        _MODEL = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        _MODEL.eval().to(DEVICE)
    return _MODEL

def read_rgb(path):
    with rasterio.open(path) as src:
        rgb = src.read().astype(np.uint8)   # (3,H,W)
        profile = src.profile
    rgb = np.transpose(rgb, (1, 2, 0))      # (H,W,3)
    return rgb, profile

def write_single(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0, 1)

def downsample_max_side(rgb_hwc, max_side):
    H, W, _ = rgb_hwc.shape
    scale = min(1.0, max_side / float(max(H, W)))
    if scale >= 1.0:
        return rgb_hwc, 1.0

    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    # Resize with torch (fast + good quality)
    x = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0).float()  # 1,3,H,W
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = x[0].permute(1, 2, 0).byte().cpu().numpy()  # H,W,3 uint8
    return out, scale

def main():
    t0 = time.time()

    rgb, profile = read_rgb(IN_RGB)
    H, W, _ = rgb.shape

    rgb_small, scale = downsample_max_side(rgb, MAX_SIDE)
    h2, w2, _ = rgb_small.shape
    print(f"Input: {H}x{W}  ->  Downsampled: {h2}x{w2}  (scale={scale:.3f})")

    model = load_model()

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    x = tfm(rgb_small).unsqueeze(0).to(DEVICE)  # (1,3,h2,w2)

    patch = 14
    new_h = (h2 // patch) * patch
    new_w = (w2 // patch) * patch
    x = x[:, :, :new_h, :new_w]

    # ✅ faster inference context than no_grad
    with torch.inference_mode():
        feats = model.forward_features(x)["x_norm_patchtokens"]  # (1, N, D)

        f = feats[0]                 # (N,D)
        f = F.normalize(f, dim=1)
        mu = f.mean(dim=0, keepdim=True)

        # anomaly = 1 - cosine similarity to mean
        cos = (f * mu).sum(dim=1)
        anomaly = (1.0 - cos).clamp(min=0)

    gh = new_h // patch
    gw = new_w // patch

    anomaly_map = anomaly.view(gh, gw)[None, None, :, :].float().cpu()

    # Upsample to small image size
    an_small = F.interpolate(
        anomaly_map, size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    # Place into full downsampled frame (h2,w2)
    small_full = np.zeros((h2, w2), dtype=np.float32)
    small_full[:new_h, :new_w] = an_small
    small_full = normalize01(small_full)

    # Upsample back to original resolution for geotiff alignment
    t = torch.from_numpy(small_full)[None, None, :, :].float()
    up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    up = normalize01(up)

    out_path = os.path.join(OUT_DIR, "dinov2_structural_score.tif")
    write_single(out_path, up, profile)

    print("Saved:", out_path)
    print(f"Total time: {time.time() - t0:.2f}s  | Device: {DEVICE}  | MAX_SIDE: {MAX_SIDE}")

if __name__ == "__main__":
    main()
