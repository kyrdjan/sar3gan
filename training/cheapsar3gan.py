import os
import shutil
from tqdm import tqdm
import zipfile
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from torchinfo import summary
from pytorch_fid import fid_score
import sys

# -----------------------
# Dataset
# -----------------------
class ImagePairZipDataset(Dataset):
    def __init__(self, hr_zip_path, lr_zip_path, transform_hr=None, transform_lr=None):
        self.hr_zip = zipfile.ZipFile(hr_zip_path, 'r')
        self.lr_zip = zipfile.ZipFile(lr_zip_path, 'r')
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        self.hr_images = sorted([f for f in self.hr_zip.namelist() if f.lower().endswith(valid_exts)])
        self.lr_images = sorted([f for f in self.lr_zip.namelist() if f.lower().endswith(valid_exts)])
        if len(self.hr_images) == 0 or len(self.lr_images) == 0:
            raise ValueError("No valid images found in the zip files!")

    def __len__(self):
        return min(len(self.hr_images), len(self.lr_images))

    def __getitem__(self, idx):
        hr_data = self.hr_zip.read(self.hr_images[idx])
        lr_data = self.lr_zip.read(self.lr_images[idx])
        hr_img = Image.open(BytesIO(hr_data)).convert("RGB")
        lr_img = Image.open(BytesIO(lr_data)).convert("RGB")
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        return lr_img, hr_img

# -----------------------
# Save Images Utility
# -----------------------
def denormalize(t):
    """Convert [-1,1] → [0,1] for saving"""
    t = (t + 1) / 2
    return t.clamp(0, 1)

def save_with_labels(tensors, labels, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    resized_tensors = []
    for i, t in enumerate(tensors):
        t = denormalize(t)
        c, h, w = t.shape
        if labels[i] == "LR":
            # Force LR to be shown pixelated
            t = F.interpolate(t.unsqueeze(0), size=(256, 256), mode="nearest").squeeze(0)
        resized_tensors.append(t)

    grid_file = filename.replace(".png", "_grid.png")
    save_image(torch.stack(resized_tensors, 0), grid_file, nrow=len(resized_tensors), normalize=False)

    # Add labels
    img = Image.open(grid_file).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    w, h = img.size
    step = w // len(tensors)
    for i, label in enumerate(labels):
        draw.text((i * step + 5, 5), label, fill="white", font=font)

    img.save(filename)
    os.remove(grid_file)

# -----------------------
# Lightweight GAN Architecture
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h*w)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, -1, h*w)
        out = torch.bmm(v, attn.permute(0,2,1)).view(b, c, h, w)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.resblocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(4)])  # bumped to 4
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv_mid = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.attn = SelfAttention(base_channels)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.final = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(x)
        x = self.upsample1(x)
        x = self.conv_mid(x)
        if x.shape[2] == 64:
            x = self.attn(x)
        x = self.upsample2(x)
        x = self.final(x)
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.resblocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(2)])
        self.attn = SelfAttention(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels*2, 1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.resblocks(x)
        if x.shape[2] == 64:
            x = self.attn(x)
        x = self.lrelu(self.conv2(x))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

# -----------------------
# Training Helper
# -----------------------
class AdversarialTraining:
    def __init__(self, G, D, device="cuda"):
        self.G = G
        self.D = D
        self.device = device

    def AccumulateDiscriminatorGradients(self, LrImages, RealSamples, Gamma=10.0):
        fake = self.G(LrImages).detach()
        real_logits = self.D(RealSamples)
        fake_logits = self.D(fake)
        d_loss = (F.softplus(fake_logits) + F.softplus(-real_logits)).mean()

        real_samples = RealSamples.requires_grad_(True)
        grad_real = torch.autograd.grad(
            outputs=self.D(real_samples).sum(), inputs=real_samples,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        r1_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()

        fake_samples = fake.requires_grad_(True)
        grad_fake = torch.autograd.grad(
            outputs=self.D(fake_samples).sum(), inputs=fake_samples,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        r2_penalty = grad_fake.pow(2).reshape(grad_fake.size(0), -1).sum(1).mean()

        d_loss_total = d_loss + Gamma*(r1_penalty + r2_penalty)
        d_loss_total.backward()
        return d_loss.detach(), d_loss_total.detach(), r1_penalty.detach(), r2_penalty.detach()

    def AccumulateGeneratorGradients(self, LrImages):
        fake = self.G(LrImages)
        g_loss = F.softplus(-self.D(fake)).mean()
        g_loss.backward()
        return g_loss.detach(), fake

# -----------------------
# FID Evaluation
# -----------------------
def evaluate_fid(G, dataloader, device, step, save_dir="fid_eval", num_images=500):
    real_dir = os.path.join(save_dir, "real")
    fake_dir = os.path.join(save_dir, "fake")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    G.eval()
    count = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            fake_imgs = G(lr_imgs)
            for i in range(lr_imgs.size(0)):
                if count >= num_images: break
                save_image(denormalize(hr_imgs[i]), os.path.join(real_dir, f"{count}.png"), normalize=False)
                save_image(denormalize(fake_imgs[i]), os.path.join(fake_dir, f"{count}.png"), normalize=False)
                count += 1
            if count >= num_images: break
    fid = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=2, device=device, dims=2048)
    print(f"Step {step}: FID = {fid:.2f}")
    return fid

# -----------------------
# Logger
# -----------------------
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# -----------------------
# Training Script
# -----------------------
def main():
    save_dir = "training_snapshots"
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "log.txt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms for LR and HR
    transform_lr = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    transform_hr = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

    train_dataset = ImagePairZipDataset("datasets/64with10k.zip", "datasets/hr256with10k.zip", transform_hr=transform_hr, transform_lr=transform_lr)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    G = Generator().to(device)
    D = Discriminator().to(device)

    print("Generator Summary:")
    summary(G, input_size=(1,3,64,64), device=device)
    print("Discriminator Summary:")
    summary(D, input_size=(1,3,256,256), device=device)

    trainer = AdversarialTraining(G, D, device)
    g_optim = optim.Adam(G.parameters(), lr=1e-4, betas=(0,0.99))
    d_optim = optim.Adam(D.parameters(), lr=1e-4, betas=(0,0.99))
    g_scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=1000, gamma=0.5)
    d_scheduler = optim.lr_scheduler.StepLR(d_optim, step_size=1000, gamma=0.5)

    step = 0
    best_fid = float("inf")
    epochs = 10

    for epoch in range(epochs):
        for lr_imgs, hr_imgs in tqdm(train_loader, ncols=120):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            d_optim.zero_grad()
            trainer.AccumulateDiscriminatorGradients(lr_imgs, hr_imgs)
            d_optim.step()

            g_optim.zero_grad()
            trainer.AccumulateGeneratorGradients(lr_imgs)
            g_optim.step()

            g_scheduler.step()
            d_scheduler.step()
            step += 1

            if step % 500 == 0:
                with torch.no_grad():
                    fake = G(lr_imgs[:1])
                    save_with_labels([lr_imgs[0].cpu(), fake[0].cpu(), hr_imgs[0].cpu()],
                                     ["LR","Fake","HR"], f"{save_dir}/step{step}.png")

            if step % 2000 == 0 and step > 0:
                fid = evaluate_fid(G, train_loader, device, step)
                if fid < best_fid:
                    best_fid = fid
                    torch.save(G.state_dict(), os.path.join(save_dir,"best_generator.pth"))

    print("=== Training Finished ===")

if __name__ == "__main__":
    main()
