import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import requests
from PIL import Image
from datetime import datetime

# ==============================
# 1. MODEL EDSR
# ==============================
class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=2):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)

# ==============================
# 2. KONFIGURASI
# ==============================
ESP32_URL = "http://<ESP32-Cam_IP_Address>/image-svga.jpg"
MODEL_PATH = "~/Model_ML_Enhance/edsr_x2_svga_to_uxga_300epoch.pth"
SAVE_DIR = "~/Outputs"
INTERVAL_SEC = 30

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 3. LOAD MODEL
# ==============================
model = EDSR(scale=2).to(device).eval()

state_dict = torch.load(MODEL_PATH, map_location=device)
model_dict = model.state_dict()
mapped_dict = {
    k_model: v_pth for (k_pth, v_pth), (k_model, _) in zip(state_dict.items(), model_dict.items())
}
model.load_state_dict(mapped_dict)

print("EDSR model loaded")

# ==============================
# 4. LOOP OTOMATIS
# ==============================
counter = 1

while True:
    try:
        # --- Ambil gambar dari ESP32 ---
        response = requests.get(ESP32_URL, timeout=10)
        lr_img = Image.open(BytesIO(response.content)).convert("RGB")
        lr_np = np.array(lr_img)

        # --- Preprocess ---
        input_tensor = (
            torch.from_numpy(lr_np)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(device)
        )

        # --- Inferensi ---
        with torch.no_grad():
            output = model(input_tensor)

        # --- Postprocess ---
        sr_np = (
            output.squeeze()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)

        # --- Simpan hasil ---
        filename = f"edsr_{counter:04d}.png"
        save_path = os.path.join(SAVE_DIR, filename)
        Image.fromarray(sr_np).save(save_path)

        print(f"Saved: {filename}")

        counter += 1
        time.sleep(INTERVAL_SEC)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
