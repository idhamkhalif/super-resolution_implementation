!pip install psutil

import psutil
import time
import os

from google.colab import drive
drive.mount('/content/drive')

# Jalankan di cell TERPISAH sebelum training
import time
from IPython.display import display, Javascript

def keep_alive():
    while True:
        display(Javascript('''
            console.log("Keeping alive...");
        '''))
        time.sleep(60)

import threading
threading.Thread(target=keep_alive).start()

"""# EDSR"""

import os, time, glob
import psutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

TRAIN_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR'
TRAIN_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR'

VAL_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR-Validation'
VAL_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR-Validation'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=128, scale=2):
        self.lr_images = sorted(glob.glob(os.path.join(lr_dir, '*')))
        self.hr_images = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        try:
            lr = Image.open(self.lr_images[idx]).convert('RGB')
            hr = Image.open(self.hr_images[idx]).convert('RGB')
        except Exception as e:
            print(f"Skip corrupted file: {self.lr_images[idx]}")
            return self.__getitem__((idx + 1) % len(self.lr_images))

        w, h = lr.size
        ps = self.patch_size
        x = np.random.randint(0, w - ps)
        y = np.random.randint(0, h - ps)

        lr_patch = lr.crop((x, y, x+ps, y+ps))
        hr_patch = hr.crop((x*2, y*2, (x+ps)*2, (y+ps)*2))

        return self.transform(lr_patch), self.transform(hr_patch)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, scale=2, num_blocks=16, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)

EDSR_CKPT_DIR = "YOUR-GDRIVE PATH FOR CHECKPOINT"
EDSR_CKPT_PATH = os.path.join(EDSR_CKPT_DIR, "edsr_last.pth")

#EDSR_CKPT_DIR = "/content/drive/MyDrive/Model_ML_Enhance"
#EDSR_CKPT_PATH = os.path.join(EDSR_CKPT_DIR, "edsr_x2_svga_to_uxga_100epoch.pth")

os.makedirs(EDSR_CKPT_DIR, exist_ok=True)

import torch

ckpt = torch.load(EDSR_CKPT_PATH)

print("Data Tipe:", type(ckpt))
if isinstance(ckpt, dict):
    print("Keys available:", ckpt.keys())
else:
    print("This file is not a dictionary, but directly a state_dict.")

import time

def train_edsr(epochs=100, batch_size=4, lr_rate=1e-4, resume=True):
    dataset = SRDataset(TRAIN_LR, TRAIN_HR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = EDSR().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler('cuda')

    history = {'loss': [], 'psnr': [], 'ssim': []}
    start_epoch = 0

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if resume and os.path.exists(EDSR_CKPT_PATH):
        ckpt = torch.load(
            EDSR_CKPT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"[EDSR] Resume from epoch {start_epoch}")

    # =========================
    # START TOTAL TIME
    # =========================
    total_start_time = time.time()

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            sr_np = sr.detach().cpu().numpy()
            hr_np = hr_img.detach().cpu().numpy()

            for i in range(sr_np.shape[0]):
                total_psnr += psnr(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    data_range=1.0
                )
                total_ssim += ssim(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    channel_axis=2,
                    data_range=1.0
                )

        n = len(loader.dataset)
        history['loss'].append(total_loss / len(loader))
        history['psnr'].append(total_psnr / n)
        history['ssim'].append(total_ssim / n)

        print(f"[EDSR] Epoch {epoch+1}/{epochs} | "
              f"Loss {history['loss'][-1]:.4f} | "
              f"PSNR {history['psnr'][-1]:.2f} | "
              f"SSIM {history['ssim'][-1]:.4f}")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history
        }, EDSR_CKPT_PATH)

    # =========================
    # END TOTAL TIME
    # =========================
    total_time = time.time() - total_start_time
    print(f"\n Total training time: {total_time/60:.2f} minutes")

    return model, history

def evaluate_cpu(model, lr_dir, hr_dir):
    model.eval()
    cpu_usage, infer_time, psnr_list, ssim_list = [], [], [], []
    process = psutil.Process(os.getpid())

    with torch.no_grad():
        for lr_p, hr_p in zip(sorted(glob.glob(lr_dir+'/*')),
                              sorted(glob.glob(hr_dir+'/*'))):

            lr = transforms.ToTensor()(Image.open(lr_p).convert('RGB')).unsqueeze(0).to(DEVICE)
            hr = np.array(Image.open(hr_p).convert('RGB')) / 255.0

            cpu_before = process.cpu_percent(interval=None)
            start = time.time()
            sr = model(lr)
            torch.cuda.synchronize() if DEVICE=='cuda' else None
            end = time.time()
            cpu_after = process.cpu_percent(interval=None)

            infer_time.append((end-start)*1000)
            cpu_usage.append(abs(cpu_after-cpu_before))

            sr_np = sr.squeeze(0).cpu().numpy().transpose(1,2,0)
            psnr_list.append(psnr(hr, sr_np, data_range=1.0))
            ssim_list.append(ssim(hr, sr_np, channel_axis=2, data_range=1.0))

    return {
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Inference Time (ms)': np.mean(infer_time),
        'CPU Usage (%)': np.mean(cpu_usage)
    }

def show_sr(model, lr_path, hr_path):
    model.eval()
    lr = Image.open(lr_path).convert('RGB')
    hr = Image.open(hr_path).convert('RGB')

    with torch.no_grad():
        sr = model(transforms.ToTensor()(lr).unsqueeze(0).to(DEVICE))

    sr = sr.squeeze(0).cpu().numpy().transpose(1,2,0)

    plt.figure(figsize=(15,5))
    for i, (img, title) in enumerate(zip([lr, sr, hr],
                                         ['LR','EDSR','HR'])):
        plt.subplot(1,3,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

def show_roi(model, lr_path, hr_path, x=100, y=100, size=64):
    model.eval()
    lr = Image.open(lr_path).convert('RGB')
    hr = Image.open(hr_path).convert('RGB')

    with torch.no_grad():
        sr = model(transforms.ToTensor()(lr).unsqueeze(0).to(DEVICE))

    sr = Image.fromarray((sr.squeeze(0).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

    lr_roi = lr.crop((x,y,x+size,y+size))
    sr_roi = sr.crop((x*2,y*2,(x+size)*2,(y+size)*2))
    hr_roi = hr.crop((x*2,y*2,(x+size)*2,(y+size)*2))

    plt.figure(figsize=(12,4))
    for i, (img, title) in enumerate(zip([lr_roi, sr_roi, hr_roi],
                                         ['LR ROI','EDSR ROI','HR ROI'])):
        plt.subplot(1,3,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

model_edsr, history_edsr = train_edsr(
    epochs=100,
    batch_size=4,
    lr_rate=1e-4,
    resume=True
)

metrics = evaluate_cpu(model_edsr, VAL_LR, VAL_HR)
metrics

show_sr(model_edsr,
        f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
        f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi(model_edsr,
         f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
         f"{VAL_HR}/YOUR_IMAGE_NAME.jpg",
         x=300, y=200)

show_sr(model_edsr,
        f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
        f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi(model_edsr,
         f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
         f"{VAL_HR}/iYOUR_IMAGE_NAME.jpg",
         x=310, y=150)

SAVE_DIR = '/../Model_ML_Enhance'
#os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 500

MODEL_PATH = f'{SAVE_DIR}/edsr_x2_svga_to_uxga_{EPOCHS}epoch.pth'
torch.save(model_edsr.state_dict(), MODEL_PATH)

print("✅ Model saved to:", MODEL_PATH)

import json

METRIC_PATH = f'{SAVE_DIR}/edsr_x2_svga_to_uxga_{EPOCHS}epoch_metrics.json'

with open(METRIC_PATH, 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to:", METRIC_PATH)

"""# Real-ESRGAN"""

import os, time, glob, json
import psutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

TRAIN_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR'
TRAIN_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR'

VAL_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR-Validation'
VAL_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR-Validation'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=128, scale=2):
        self.lr_images = sorted(glob.glob(os.path.join(lr_dir, '*')))
        self.hr_images = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        try:
          lr = Image.open(self.lr_images[idx]).convert('RGB')
          hr = Image.open(self.hr_images[idx]).convert('RGB')
        except Exception as e:
          print(f"Skip corrupted file: {self.lr_images[idx]}")
          return self.__getitem__((idx + 1) % len(self.lr_images))

        w, h = lr.size
        ps = self.patch_size

        x = np.random.randint(0, w - ps)
        y = np.random.randint(0, h - ps)

        lr_patch = lr.crop((x, y, x + ps, y + ps))
        hr_patch = hr.crop((x * 2, y * 2,
                            (x + ps) * 2, (y + ps) * 2))

        return self.transform(lr_patch), self.transform(hr_patch)

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels*2, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels*3, channels, 3, 1, 1)
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
            nn.Conv2d(channels, channels*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.head(x)
        body = self.body(fea)
        fea = fea + body
        return self.tail(fea)

CKPT_DIR = "YOUR-GDRIVE PATH FOR CHECKPOINT"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, 'realesrgan_last.pth')

# @title
import time

def train_realesrgan(epochs=500, batch_size=4, lr_rate=1e-4, resume=True):
    dataset = SRDataset(TRAIN_LR, TRAIN_HR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = RealESRGAN().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler('cuda')

    history = {'loss': [], 'psnr': [], 'ssim': []}
    start_epoch = 0

    # =========================
    # LOAD CHECKPOINT (optional)
    # =========================
    if resume and os.path.exists(CKPT_PATH):
        ckpt = torch.load(
            CKPT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"Resume training from epoch {start_epoch}")

    # =========================
    # START TOTAL TIME
    # =========================
    total_start_time = time.time()

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            sr_np = sr.detach().cpu().numpy()
            hr_np = hr_img.detach().cpu().numpy()

            for i in range(sr_np.shape[0]):
                total_psnr += psnr(
                    hr_np[i].transpose(1, 2, 0),
                    sr_np[i].transpose(1, 2, 0),
                    data_range=1.0
                )
                total_ssim += ssim(
                    hr_np[i].transpose(1, 2, 0),
                    sr_np[i].transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=1.0
                )

        n = len(loader.dataset)
        history['loss'].append(total_loss / len(loader))
        history['psnr'].append(total_psnr / n)
        history['ssim'].append(total_ssim / n)

        print(f"[Real-ESRGAN] Epoch {epoch+1}/{epochs} | "
              f"Loss {history['loss'][-1]:.4f} | "
              f"PSNR {history['psnr'][-1]:.2f} | "
              f"SSIM {history['ssim'][-1]:.4f}")

        # =========================
        # SAVE CHECKPOINT
        # =========================
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history
        }, CKPT_PATH)

    # =========================
    # END TOTAL TIME
    # =========================
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    return model, history

# @title
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_realesrgan(epochs=500, batch_size=4, lr_rate=1e-4, resume=True):
    dataset = SRDataset(TRAIN_LR, TRAIN_HR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = RealESRGAN().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler('cuda')

    history = {
        'loss': [],
        'psnr': [],
        'ssim': [],
        'epoch_time': []   # ⬅️ waktu per epoch
    }

    start_epoch = 0
    total_elapsed_time = 0.0  # ⬅️ waktu total kumulatif (detik)

    # =========================
    # LOAD CHECKPOINT (optional)
    # =========================
    if resume and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        total_elapsed_time = ckpt.get('total_elapsed_time', 0.0)  # ⬅️ aman untuk ckpt lama
        print(f"Resume training from epoch {start_epoch}")
        print(f"Previous training time: {total_elapsed_time/60:.2f} minutes")

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            sr_np = sr.detach().cpu().numpy()
            hr_np = hr_img.detach().cpu().numpy()

            for i in range(sr_np.shape[0]):
                total_psnr += psnr(
                    hr_np[i].transpose(1, 2, 0),
                    sr_np[i].transpose(1, 2, 0),
                    data_range=1.0
                )
                total_ssim += ssim(
                    hr_np[i].transpose(1, 2, 0),
                    sr_np[i].transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=1.0
                )

        n = len(loader.dataset)
        history['loss'].append(total_loss / len(loader))
        history['psnr'].append(total_psnr / n)
        history['ssim'].append(total_ssim / n)

        epoch_time = time.time() - epoch_start_time
        total_elapsed_time += epoch_time
        history['epoch_time'].append(epoch_time)

        print(
            f"[Real-ESRGAN] Epoch {epoch+1}/{epochs} | "
            f"Loss {history['loss'][-1]:.4f} | "
            f"PSNR {history['psnr'][-1]:.2f} | "
            f"SSIM {history['ssim'][-1]:.4f} | "
            f"Time {epoch_time:.2f}s"
        )

        # =========================
        # SAVE CHECKPOINT
        # =========================
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history,
            'total_elapsed_time': total_elapsed_time  # SAVE TIME
        }, CKPT_PATH)

    print(f"\n Total training time: {total_elapsed_time/60:.2f} minutes")

    return model, history

def evaluate_cpu(model, lr_dir, hr_dir):
    model.eval()
    cpu_usage, infer_time, psnr_list, ssim_list = [], [], [], []
    process = psutil.Process(os.getpid())

    with torch.no_grad():
        for lr_p, hr_p in zip(sorted(glob.glob(lr_dir+'/*')),
                              sorted(glob.glob(hr_dir+'/*'))):

            lr = transforms.ToTensor()(Image.open(lr_p).convert('RGB')).unsqueeze(0).to(DEVICE)
            hr = np.array(Image.open(hr_p).convert('RGB')) / 255.0

            cpu_before = process.cpu_percent(interval=None)
            start = time.time()
            sr = model(lr)
            torch.cuda.synchronize() if DEVICE=='cuda' else None
            end = time.time()
            cpu_after = process.cpu_percent(interval=None)

            infer_time.append((end-start)*1000)
            cpu_usage.append(abs(cpu_after-cpu_before))

            sr_np = sr.squeeze(0).cpu().numpy().transpose(1,2,0)
            psnr_list.append(psnr(hr, sr_np, data_range=1.0))
            ssim_list.append(ssim(hr, sr_np, channel_axis=2, data_range=1.0))

    return {
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Inference Time (ms)': np.mean(infer_time),
        'CPU Usage (%)': np.mean(cpu_usage)
    }

def show_roi(model, lr_path, hr_path, x=100, y=100, size=64):
    model.eval()
    lr = Image.open(lr_path).convert('RGB')
    hr = Image.open(hr_path).convert('RGB')

    with torch.no_grad():
        sr = model(transforms.ToTensor()(lr).unsqueeze(0).to(DEVICE))

    sr = Image.fromarray((sr.squeeze(0).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

    lr_roi = lr.crop((x,y,x+size,y+size))
    sr_roi = sr.crop((x*2,y*2,(x+size)*2,(y+size)*2))
    hr_roi = hr.crop((x*2,y*2,(x+size)*2,(y+size)*2))

    plt.figure(figsize=(12,4))
    for i, (img, title) in enumerate(zip([lr_roi, sr_roi, hr_roi],
                                         ['LR ROI','Real-ESRGAN ROI','HR ROI'])):
        plt.subplot(1,3,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()


model_realesrgan, history = train_realesrgan(
    epochs=500,
    batch_size=4,
    lr_rate=1e-4,
    resume=True
)

metrics_r = evaluate_cpu(model_realesrgan, VAL_LR, VAL_HR)
metrics_r

show_sr(model_realesrgan,
        f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
        f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi(model_realesrgan,
         f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
         f"{VAL_HR}/YOUR_IMAGE_NAME.jpg",
         x=300, y=200)

show_sr(model_realesrgan,
        f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
        f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi(model_realesrgan,
         f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
         f"{VAL_HR}/YOUR_IMAGE_NAME.jpg",
         x=310, y=150)

SAVE_DIR = '/.../Model_ML_Enhance'
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 500
MODEL_PATH = f'{SAVE_DIR}/realesrgan_x2_svga_to_uxga_{EPOCHS}epoch.pth'
torch.save(model_realesrgan.state_dict(), MODEL_PATH)

print("Real-ESRGAN model saved to:", MODEL_PATH)

import json

METRIC_PATH = f'{SAVE_DIR}/realsrgan_x2_svga_to_uxga_{EPOCHS}epoch_metrics.json'

with open(METRIC_PATH, 'w') as f:
    json.dump(metrics_r, f, indent=4)

print("Metrics saved to:", METRIC_PATH)

"""# ESPCN"""

import os, time, glob, json
import psutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

TRAIN_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR'
TRAIN_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR'

VAL_LR = 'YOUR_G-DRIVE_PATH_FOR DATASET-LR-Validation'
VAL_HR = 'YOUR_G-DRIVE_PATH_FOR DATASET-HR-Validation'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=128, scale=2):
        self.lr_images = sorted(glob.glob(os.path.join(lr_dir, '*')))
        self.hr_images = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_images[idx]).convert('RGB')
        hr = Image.open(self.hr_images[idx]).convert('RGB')

        w, h = lr.size
        ps = self.patch_size
        x = np.random.randint(0, w - ps)
        y = np.random.randint(0, h - ps)

        lr_patch = lr.crop((x, y, x+ps, y+ps))
        hr_patch = hr.crop((x*2, y*2, (x+ps)*2, (y+ps)*2))

        return self.transform(lr_patch), self.transform(hr_patch)

class ESPCN(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.features(x)

ESPCN_CKPT_DIR = "YOUR-GDRIVE PATH FOR CHECKPOINT"
os.makedirs(ESPCN_CKPT_DIR, exist_ok=True)
ESPCN_CKPT_PATH = os.path.join(ESPCN_CKPT_DIR, "espcn_ckpt.pth")

# @title
import time

def train_espcn(epochs=500, batch_size=4, lr_rate=1e-4, resume=True):
    dataset = SRDataset(TRAIN_LR, TRAIN_HR)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   
        pin_memory=True
    )

    model = ESPCN(scale=2).to(DEVICE)
    criterion = nn.MSELoss()   # ESPCN
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler('cuda')

    history = {'loss': [], 'psnr': [], 'ssim': []}
    start_epoch = 0

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if resume and os.path.exists(ESPCN_CKPT_PATH):
        ckpt = torch.load(
            ESPCN_CKPT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"[ESPCN] Resume from epoch {start_epoch}")

    # =========================
    # START TOTAL TIME
    # =========================
    total_start_time = time.time()

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            sr_np = sr.detach().cpu().numpy()
            hr_np = hr_img.detach().cpu().numpy()

            for i in range(sr_np.shape[0]):
                total_psnr += psnr(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    data_range=1.0
                )
                total_ssim += ssim(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    channel_axis=2,
                    data_range=1.0
                )

        n = len(loader.dataset)
        history['loss'].append(total_loss / len(loader))
        history['psnr'].append(total_psnr / n)
        history['ssim'].append(total_ssim / n)

        print(f"[ESPCN] Epoch {epoch+1}/{epochs} | "
              f"Loss {history['loss'][-1]:.4f} | "
              f"PSNR {history['psnr'][-1]:.2f} | "
              f"SSIM {history['ssim'][-1]:.4f}")

        # =========================
        # SAVE CHECKPOINT
        # =========================
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history
        }, ESPCN_CKPT_PATH)

    # =========================
    # END TOTAL TIME
    # =========================
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    return model, history

# @title
import time

def train_espcn(epochs=500, batch_size=4, lr_rate=1e-4, resume=True):
    dataset = SRDataset(TRAIN_LR, TRAIN_HR)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # WAJIB untuk Colab + Drive
        pin_memory=True
    )

    model = ESPCN(scale=2).to(DEVICE)
    criterion = nn.MSELoss()   # ESPCN asli
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler('cuda')

    history = {'loss': [], 'psnr': [], 'ssim': []}
    start_epoch = 0
    elapsed_time = 0.0   # ⬅️ TAMBAHAN (aman)

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if resume and os.path.exists(ESPCN_CKPT_PATH):
        ckpt = torch.load(
            ESPCN_CKPT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1

        # ⬅️ AMAN jika checkpoint lama belum punya time
        elapsed_time = ckpt.get('elapsed_time', 0.0)

        print(f"[ESPCN] Resume from epoch {start_epoch}")

    # =========================
    # START TOTAL TIME
    # =========================
    total_start_time = time.time()

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()   # ⬅️ TAMBAHAN

        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0

        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            sr_np = sr.detach().cpu().numpy()
            hr_np = hr_img.detach().cpu().numpy()

            for i in range(sr_np.shape[0]):
                total_psnr += psnr(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    data_range=1.0
                )
                total_ssim += ssim(
                    hr_np[i].transpose(1,2,0),
                    sr_np[i].transpose(1,2,0),
                    channel_axis=2,
                    data_range=1.0
                )

        epoch_time = time.time() - epoch_start_time   
        elapsed_time += epoch_time                    

        n = len(loader.dataset)
        history['loss'].append(total_loss / len(loader))
        history['psnr'].append(total_psnr / n)
        history['ssim'].append(total_ssim / n)

        print(
            f"[ESPCN] Epoch {epoch+1}/{epochs} | "
            f"Loss {history['loss'][-1]:.4f} | "
            f"PSNR {history['psnr'][-1]:.2f} | "
            f"SSIM {history['ssim'][-1]:.4f} | "
            f"Time {epoch_time/60:.2f} min"
        )

        # =========================
        # SAVE CHECKPOINT (+ TIME)
        # =========================
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history,
            'elapsed_time': elapsed_time  
        }, ESPCN_CKPT_PATH)

    # =========================
    # END TOTAL TIME
    # =========================
    total_time = time.time() - total_start_time + elapsed_time
    print(f"\n Total training time: {total_time/60:.2f} minutes")

    return model, history

def evaluate_cpu(model, lr_dir, hr_dir):
    model.eval()
    cpu_usage, infer_time, psnr_list, ssim_list = [], [], [], []
    process = psutil.Process(os.getpid())

    with torch.no_grad():
        for lr_p, hr_p in zip(sorted(glob.glob(lr_dir+'/*')),
                              sorted(glob.glob(hr_dir+'/*'))):

            lr = transforms.ToTensor()(Image.open(lr_p).convert('RGB')).unsqueeze(0).to(DEVICE)
            hr = np.array(Image.open(hr_p).convert('RGB')) / 255.0

            cpu_before = process.cpu_percent(interval=None)
            start = time.time()
            sr = model(lr)
            torch.cuda.synchronize() if DEVICE=='cuda' else None
            end = time.time()
            cpu_after = process.cpu_percent(interval=None)

            infer_time.append((end-start)*1000)
            cpu_usage.append(abs(cpu_after-cpu_before))

            sr_np = sr.squeeze(0).cpu().numpy().transpose(1,2,0)
            psnr_list.append(psnr(hr, sr_np, data_range=1.0))
            ssim_list.append(ssim(hr, sr_np, channel_axis=2, data_range=1.0))

    return {
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Inference Time (ms)': np.mean(infer_time),
        'CPU Usage (%)': np.mean(cpu_usage)
    }

def show_roi_espcn(model, lr_path, hr_path, x=100, y=100, size=64):
    model.eval()
    lr = Image.open(lr_path).convert('RGB')
    hr = Image.open(hr_path).convert('RGB')

    with torch.no_grad():
        sr = model(transforms.ToTensor()(lr).unsqueeze(0).to(DEVICE))

    sr = Image.fromarray((sr.squeeze(0).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

    lr_roi = lr.crop((x,y,x+size,y+size))
    sr_roi = sr.crop((x*2,y*2,(x+size)*2,(y+size)*2))
    hr_roi = hr.crop((x*2,y*2,(x+size)*2,(y+size)*2))

    plt.figure(figsize=(12,4))
    for i, (img, title) in enumerate(zip([lr_roi, sr_roi, hr_roi],
                                         ['LR ROI','ESPCN ROI','HR ROI'])):
        plt.subplot(1,3,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

model_espcn, history_espcn = train_espcn(
    epochs=500,
    batch_size=4,
    lr_rate=1e-4,
    resume=True   # True = lanjut dari checkpoint jika ada
)

metrics_espcn = evaluate_cpu(
    model_espcn,
    VAL_LR,
    VAL_HR
)

print(metrics_espcn)

# ============================
# Fungsi show_sr (full image)
# ============================
def show_sr_espcn(model, lr_path, hr_path):
    """
    Displays LR, SR (ESPCN results), and HR images side by side.
    The code structure is adapted from the original code.
    """
    model.eval()
    lr = Image.open(lr_path).convert('RGB')
    hr = Image.open(hr_path).convert('RGB')

    with torch.no_grad():
        lr_tensor = transforms.ToTensor()(lr).unsqueeze(0).to(DEVICE)
        sr_tensor = model(lr_tensor)

    sr = Image.fromarray((sr_tensor.squeeze(0).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

    plt.figure(figsize=(12,4))
    for i, (img, title) in enumerate(zip([lr, sr, hr], ['LR','ESPCN','HR'])):
        plt.subplot(1,3,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

show_sr_espcn(model_espcn,
              f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
              f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi_espcn(model_espcn,
               f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
               f"{VAL_HR}/YOUR_IMAGE_NAME.jpg",
               x=300, y=200)

show_sr_espcn(model_espcn,
          f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
          f"{VAL_HR}/YOUR_IMAGE_NAME.jpg")

show_roi_espcn(model_espcn,
          f"{VAL_LR}/YOUR_IMAGE_NAME.jpg",
          f"{VAL_HR}/YOUR_IMAGE_NAME.jpg",
          x=310, y=150)

EPOCHS = 500
SAVE_DIR = '/.../Model_ML_Enhance'
#os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = f'{SAVE_DIR}/espcn_x2_svga_to_uxga_{EPOCHS}epoch.pth'
torch.save(model_espcn.state_dict(), MODEL_PATH)

print("ESPCN model saved to:", MODEL_PATH)

import json

METRIC_PATH = f'{SAVE_DIR}/espcn_x2_svga_to_uxga_{EPOCHS}epoch_metrics.json'

with open(METRIC_PATH, 'w') as f:
    json.dump(metrics_espcn, f, indent=4)

print("✅ Metrics saved to:", METRIC_PATH)