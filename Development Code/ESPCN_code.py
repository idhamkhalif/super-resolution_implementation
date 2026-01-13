import os
import time
import torch
import torch.nn as nn
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime

# ==============================
# 1. ARSITEKTUR ESPCN (x2)
# ==============================
class ESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return self.pixel_shuffle(x)

# ==============================
# 2. KONFIGURASI
# ==============================
ESP32_URL = "http://<ESP32-Cam_IP_Address>/image-svga.jp"
MODEL_PATH = "~/Model_ML_Enhance/espcn_x2_svga_to_uxga_300epoch.pth"
SAVE_DIR = "~/Outputs/"
INTERVAL_SEC = 30

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# 3. LOAD MODEL (ROBUST)
# ==============================
model = ESPCN(scale_factor=2).to(DEVICE).eval()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "params" in ckpt:
        ckpt = ckpt["params"]

model.load_state_dict(ckpt, strict=False)

print("ESPCN model loaded")

# ==============================
# 4. LOOP OTOMATIS (30 DETIK)
# ==============================
counter = 1

while True:
    try:
        # --- Ambil gambar LR dari ESP32 ---
        resp = requests.get(ESP32_URL, timeout=10)
        lr_img = Image.open(BytesIO(resp.content)).convert("RGB")
        lr_np = np.array(lr_img)

        # --- Preprocess ---
        input_tensor = (
            torch.from_numpy(lr_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # --- Inferensi ---
        with torch.no_grad():
            sr_tensor = model(input_tensor)

        # --- Postprocess ---
        sr_np = (
            sr_tensor.squeeze(0)
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )
        sr_np = (sr_np * 255).astype(np.uint8)

        # --- Simpan hasil ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sespcn_{timestamp}.png"
        save_path = os.path.join(SAVE_DIR, filename)

        Image.fromarray(sr_np).save(save_path)

        print(f"Saved: {filename}")

        counter += 1
        time.sleep(INTERVAL_SEC)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
