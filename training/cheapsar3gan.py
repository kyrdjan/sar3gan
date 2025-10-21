# ============================================================
# complete_training_fir_64x256.py — 64x64 → 256x256 version (final fixed)
# ============================================================

import os, sys, shutil, zipfile, math
from io import BytesIO
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from torchinfo import summary
from pytorch_fid import fid_score
import numpy as np

# Add the folder containing torch_utils to the Python path
sys.path.append(os.path.abspath("."))
from torch_utils.ops import upfirdn2d

# ============================================================
# FIR Kernel and Resamplers
# ============================================================
def CreateLowpassKernel(weights, inplace=False):
    kernel = np.array([weights]) if inplace else np.convolve(weights, [1, 1]).reshape(1, -1)
    kernel = torch.Tensor(kernel.T @ kernel)
    return kernel / torch.sum(kernel)

class InterpolativeUpsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super().__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter))
    def forward(self, x): return upfirdn2d.upsample2d(x, self.Kernel)

class InterpolativeDownsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super().__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter))
    def forward(self, x): return upfirdn2d.downsample2d(x, self.Kernel)

InterpolativeUpsampler = InterpolativeUpsamplerCUDA
InterpolativeDownsampler = InterpolativeDownsamplerCUDA

# ============================================================
# Dataset Loader (paired ZIPs)
# ============================================================
class ImagePairZipDataset(Dataset):
    def __init__(self, hr_zip_path, lr_zip_path, transform_hr=None, transform_lr=None):
        self.hr_zip = zipfile.ZipFile(hr_zip_path, 'r')
        self.lr_zip = zipfile.ZipFile(lr_zip_path, 'r')
        self.transform_hr, self.transform_lr = transform_hr, transform_lr
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        self.hr_images = sorted([f for f in self.hr_zip.namelist() if f.lower().endswith(valid_exts)])
        self.lr_images = sorted([f for f in self.lr_zip.namelist() if f.lower().endswith(valid_exts)])
        if not self.hr_images or not self.lr_images:
            raise ValueError("No valid images found in the zip files!")
    def __len__(self): return min(len(self.hr_images), len(self.lr_images))
    def __getitem__(self, idx):
        hr_img = Image.open(BytesIO(self.hr_zip.read(self.hr_images[idx]))).convert("RGB")
        lr_img = Image.open(BytesIO(self.lr_zip.read(self.lr_images[idx]))).convert("RGB")
        if self.transform_hr: hr_img = self.transform_hr(hr_img)
        if self.transform_lr: lr_img = self.transform_lr(lr_img)
        return lr_img, hr_img

# ============================================================
# Utility
# ============================================================
def denormalize(t): return ((t + 1) / 2).clamp(0, 1)

def save_with_labels(tensors, labels, filename, target_size=(256, 256)):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imgs = []
    for i, t in enumerate(tensors):
        t = denormalize(t)
        mode = "nearest" if labels[i].upper() == "LR" else "bilinear"
        t = F.interpolate(t.unsqueeze(0), size=target_size, mode=mode, align_corners=False).squeeze(0)
        imgs.append(t)
    tmp = filename.replace(".png", "_grid.png")
    save_image(torch.stack(imgs), tmp, nrow=len(imgs))
    img = Image.open(tmp).convert("RGB")
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 18)
    except: font = ImageFont.load_default()
    w, _ = img.size
    step = w // len(tensors)
    for i, label in enumerate(labels):
        draw.text((i * step + 5, 5), label, fill="white", font=font)
    img.save(filename)
    os.remove(tmp)

# ============================================================
# Network Building Blocks
# ============================================================
def MSRInitializer(layer, gain=1.0):
    if hasattr(layer, "weight"):
        nn.init.normal_(layer.weight, mean=0.0,
                        std=gain / math.sqrt(layer.weight.data.size(1) * layer.weight[0][0].numel()))
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.layer = MSRInitializer(nn.Conv2d(in_ch, out_ch, k, padding=(k-1)//2))
    def forward(self, x): return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1, self.conv2 = Convolution(ch, ch), Convolution(ch, ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query, self.key = [MSRInitializer(nn.Conv2d(ch, ch // 8, 1)) for _ in range(2)]
        self.value = MSRInitializer(nn.Conv2d(ch, ch, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        att = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out

class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, Filter=[1,2,1]):
        super().__init__()
        self.resampler = InterpolativeUpsampler(Filter)
        self.proj = Convolution(in_ch, out_ch, 1)
    def forward(self, x): return self.proj(self.resampler(x))

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, Filter=[1,2,1]):
        super().__init__()
        self.resampler = InterpolativeDownsampler(Filter)
        self.proj = Convolution(in_ch, out_ch, 1)
    def forward(self, x): return self.proj(self.resampler(x))

# ============================================================
# Generator & Discriminator (ends at 256×256)
# ============================================================
class GeneratorStage(nn.Module):
    def __init__(self, in_ch, out_ch, *, num_blocks=2, ResamplingFilter=None):
        super().__init__()
        self.transition = UpsampleLayer(in_ch, out_ch, ResamplingFilter) if ResamplingFilter else Convolution(in_ch, out_ch)
        self.blocks = nn.ModuleList([ResidualBlock(out_ch) for _ in range(num_blocks)])
        self.attn = SelfAttention(out_ch)
    def forward(self, x):
        x = self.transition(x)
        for b in self.blocks: x = b(x)
        return self.attn(x)

class DiscriminatorStage(nn.Module):
    def __init__(self, in_ch, out_ch, *, num_blocks=2, ResamplingFilter=None):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(in_ch) for _ in range(num_blocks)])
        self.down = DownsampleLayer(in_ch, out_ch, ResamplingFilter) if ResamplingFilter else None
        self.proj = Convolution(in_ch, out_ch, 1) if not ResamplingFilter else None
        self.attn = SelfAttention(out_ch)
    def forward(self, x):
        for b in self.blocks: x = b(x)
        x = self.down(x) if self.down else self.proj(x)
        return self.attn(x)

class Generator(nn.Module):
    def __init__(self, ResamplingFilter=[1,2,1]):
        super().__init__()
        self.stage1 = GeneratorStage(3, 64, num_blocks=2, ResamplingFilter=None)   # 64×64
        self.stage2 = GeneratorStage(64, 128, num_blocks=2, ResamplingFilter=ResamplingFilter)  # →128×128
        self.stage3 = GeneratorStage(128, 256, num_blocks=2, ResamplingFilter=ResamplingFilter) # →256×256
        self.agg = Convolution(256, 3, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.tanh(self.agg(x))

class Discriminator(nn.Module):
    def __init__(self, ResamplingFilter=[1,2,1]):
        super().__init__()
        self.extract = Convolution(3, 64, 1)
        self.stage1 = DiscriminatorStage(64, 128, num_blocks=2, ResamplingFilter=ResamplingFilter)
        self.stage2 = DiscriminatorStage(128, 256, num_blocks=2, ResamplingFilter=ResamplingFilter)
        self.stage3 = DiscriminatorStage(256, 512, num_blocks=2, ResamplingFilter=None)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = MSRInitializer(nn.Linear(512, 1))
    def forward(self, x):
        x = self.extract(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(self.gap(x).view(x.size(0), -1)).view(-1)

# ============================================================
# Adversarial Training
# ============================================================
class AdversarialTraining:
    def __init__(self, G, D, device="cuda"):
        self.G, self.D, self.device = G, D, device
    @staticmethod
    def ZeroCenteredGradientPenalty(samples, logits):
        gradients, = torch.autograd.grad(outputs=logits.sum(), inputs=samples, create_graph=True)
        return gradients.pow(2).reshape(gradients.size(0), -1).sum(1)
    def GeneratorLoss(self, Lr, Hr):
        fake = self.G(Lr)
        return F.softplus(-self.D(fake)).mean(), fake
    def DiscriminatorLoss(self, Lr, Hr, Gamma=10.0):
        Hr = Hr.detach().requires_grad_(True)
        fake = self.G(Lr).detach().requires_grad_(True)
        real_logits, fake_logits = self.D(Hr), self.D(fake)
        d_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        r1 = self.ZeroCenteredGradientPenalty(Hr, real_logits).mean()
        r2 = self.ZeroCenteredGradientPenalty(fake, fake_logits).mean()
        return d_loss, d_loss + (Gamma / 2.0) * (r1 + r2), r1, r2

# ============================================================
# Training Script
# ============================================================
class Logger:
    def __init__(self, path):
        self.terminal, self.log = sys.stdout, open(path, "a", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()

def evaluate_fid(G, dataloader, device, step, save_dir="fid_eval", num_images=500):
    real_dir, fake_dir = os.path.join(save_dir,"real"), os.path.join(save_dir,"fake")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(real_dir); os.makedirs(fake_dir)
    G.eval(); count = 0
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            fake = G(lr)
            for i in range(lr.size(0)):
                if count >= num_images: break
                save_image(denormalize(hr[i]), f"{real_dir}/{count}.png")
                save_image(denormalize(fake[i]), f"{fake_dir}/{count}.png")
                count += 1
            if count >= num_images: break
    fid = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=2, device=device, dims=2048)
    print(f"Step {step}: FID = {fid:.2f}")
    return fid

def main():
    save_dir = "training_snapshots_64to256"
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "log.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_lr = transforms.Compose([
        transforms.Resize((64,64)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_hr = transforms.Compose([
        transforms.Resize((256,256)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImagePairZipDataset(
        "datasets/hr256with10k.zip",
        "datasets/64with10k.zip",
        transform_hr=transform_hr, transform_lr=transform_lr
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    G, D = Generator().to(device), Discriminator().to(device)
    print("Generator Summary:"); summary(G, input_size=(1,3,64,64), device=device)
    print("Discriminator Summary:"); summary(D, input_size=(1,3,256,256), device=device)

    trainer = AdversarialTraining(G,D,device)
    g_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0,0.99))
    d_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0,0.99))
    g_sched = optim.lr_scheduler.StepLR(g_opt, step_size=1000, gamma=0.5)
    d_sched = optim.lr_scheduler.StepLR(d_opt, step_size=1000, gamma=0.5)

    step, best_fid, epochs = 0, float("inf"), 10
    for epoch in range(epochs):
        for lr_imgs, hr_imgs in tqdm(loader, ncols=120):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            d_opt.zero_grad()
            d_loss, d_total, r1, r2 = trainer.DiscriminatorLoss(lr_imgs, hr_imgs)
            d_total.backward(); d_opt.step()
            g_opt.zero_grad()
            g_loss, fake = trainer.GeneratorLoss(lr_imgs, hr_imgs)
            g_loss.backward(); g_opt.step()
            g_sched.step(); d_sched.step(); step += 1

            if step % 50 == 0:
                save_with_labels([lr_imgs[0], fake[0], hr_imgs[0]],
                                 ["LR","SR","HR"], f"{save_dir}/step_{step:06d}.png")
                print(f"Step {step:06d} | G:{g_loss.item():.4f} | D:{d_loss.item():.4f} | R1:{r1.item():.4f} | R2:{r2.item():.4f}")
            if step % 300 == 0:
                fid = evaluate_fid(G, loader, device, step)
                if fid < best_fid:
                    best_fid = fid
                    torch.save(G.state_dict(), os.path.join(save_dir, "best_generator.pth"))
                    print(f">> New best FID: {fid:.2f} at step {step}")

    torch.save(G.state_dict(), os.path.join(save_dir, "final_generator.pth"))
    torch.save(D.state_dict(), os.path.join(save_dir, "final_discriminator.pth"))
    print(">> Training complete.")

if __name__ == "__main__":
    main()
