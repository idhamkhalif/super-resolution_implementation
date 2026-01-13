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
# 1. MODEL REAL-ESRGAN (x2)
# ==============================
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels * 3, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x + x3 * 0.2


class RRDB(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + self.rdb3(self.rdb2(self.rdb1(x))) * 0.2


class RealESRGAN(nn.Module):
    def __init__(self, scale=2, num_blocks=6, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.head(x)
        fea = fea + self.body(fea)
        return self.tail(fea)

# ==============================
# 2. KONFIGURASI
# ==============================
ESP32_URL = "http://<ESP32-Cam_IP_Address>/image-svga.jpg"
MODEL_PATH = "~/Model_ML_Enhance/realesrgan_x2_svga_to_uxga_500epoch.pth"
SAVE_DIR = "~/Outputs"
INTERVAL_SEC = 30

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# 3. LOAD MODEL
# ==============================
model = RealESRGAN(scale=2, num_blocks=6, channels=64).to(DEVICE).eval()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state_dict)

print("Real-ESRGAN model loaded")

# ==============================
# 4. LOOP OTOMATIS (30 DETIK)
# ==============================
counter = 1

while True:
    try:
        # --- Ambil gambar dari ESP32 ---
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
        filename = f"srealesrgan_{timestamp}.png"
        save_path = os.path.join(SAVE_DIR, filename)

        Image.fromarray(sr_np).save(save_path)

        print(f"Saved: {filename}")

        counter += 1
        time.sleep(INTERVAL_SEC)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
