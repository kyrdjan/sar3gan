import os

import zipfile
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import sys, shutil, zipfile, math, copy, time
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


sys.path.append(os.path.abspath("."))  # for torch_utils.ops
from torch_utils.ops import upfirdn2d, bias_act

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


# ======================
# Dataset (same as train)
# ======================
class ImagePairZipDataset(Dataset):
    def __init__(self, hr_zip_path, lr_zip_path, transform_hr, transform_lr):
        self.hr_zip = zipfile.ZipFile(hr_zip_path, 'r')
        self.lr_zip = zipfile.ZipFile(lr_zip_path, 'r')
        self.hr_files = sorted(self.hr_zip.namelist())
        self.lr_files = sorted(self.lr_zip.namelist())
        self.t_hr = transform_hr
        self.t_lr = transform_lr

    def __len__(self):
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        hr = Image.open(BytesIO(self.hr_zip.read(self.hr_files[idx]))).convert("RGB")
        lr = Image.open(BytesIO(self.lr_zip.read(self.lr_files[idx]))).convert("RGB")
        return self.t_lr(lr), self.t_hr(hr)

# ======================
# Generator (import yours)
# ======================

class GenerativeBasis(nn.Module):
    """
    Kept for experimentation, but NOT used as the generator's first-stage
    transition in the default SR path. For conditional SR we preserve spatial
    structure and therefore prefer a convolutional transition.
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # Basis spatial resolution chosen to match later upsampling (64x64 basis)
        self.Basis = nn.Parameter(torch.empty(output_channels, 64, 64).normal_(0, 1))

        self.Project = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.LinearLayer = MSRInitializer(
            nn.Linear(64, output_channels, bias=False)
        )

    def forward(self, x):
        # x: [N, C_in, H, W] -> features -> coefficients -> scale basis
        features = self.Project(x).view(x.size(0), -1)   # [N, 64]
        coeffs = self.LinearLayer(features)              # [N, Out]
        out = self.Basis.view(1, -1, 64, 64) * coeffs.view(x.size(0), -1, 1, 1)
        return out

class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super().__init__()
        self.DataType = DataType

        # CHANGE: for SR we want to keep spatial structure at the first stage.
        # If ResamplingFilter is None (first stage), use a normal Conv transition.
        # Otherwise use UpsampleLayer.
        if ResamplingFilter is None:
            # spatial-preserving transition
            self.Transition = Convolution(InputChannels, OutputChannels, k=3)
        else:
            self.Transition = UpsampleLayer(InputChannels, OutputChannels)

        self.Blocks = nn.ModuleList([
            ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)
            for _ in range(NumberOfBlocks)
        ])

        self.Attention = SelfAttention(OutputChannels)

    def forward(self, x):
        x = x.to(self.DataType)
        x = self.Transition(x)
        for block in self.Blocks:
            x = block(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage,
                 ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=3,
                 KernelSize=3, ResamplingFilter=[1,2,1]):
        super().__init__()

        VarianceScalingParameter = sum(BlocksPerStage)

        # First stage uses GenerativeBasis (resample=None)
        in_ch = 3 if ConditionDimension is None else 3 + ConditionEmbeddingDimension
        MainLayers = [
            GeneratorStage(in_ch, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0],
                           ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None)
        ]
        # Remaining stages use Upsample
        MainLayers += [
            GeneratorStage(WidthPerStage[i], WidthPerStage[i + 1], CardinalityPerStage[i + 1],
                           BlocksPerStage[i + 1], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter)
            for i in range(len(WidthPerStage) - 1)
        ]

        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, k=1)

        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(
                nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False)
            )

    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            cond = self.EmbeddingLayer(y).view(y.shape[0], -1, 1, 1)
            x = torch.cat([x, cond.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
        for Layer in self.MainLayers:
            x = Layer(x)
        return torch.tanh(self.AggregationLayer(x))
    
def MSRInitializer(layer, gain=1.0):
    if hasattr(layer, "weight"):
        nn.init.normal_(layer.weight, mean=0.0,
                        std=gain / math.sqrt(layer.weight.data.size(1) * layer.weight[0][0].numel()))
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = (k - 1) // 2
        self.layer = MSRInitializer(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, groups=groups)
        )
    def forward(self, x): 
    # check for Nan/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf detected in Convolution input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.layer(x)

# ============================================================
# Residual Block
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality=4, ExpansionFactor=2, KernelSize=3, VarianceScalingParameter=1.0):
        super().__init__()
        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        assert ExpandedChannels % Cardinality == 0

        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))

        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, k=1)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, k=KernelSize, groups=Cardinality)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, k=1)

        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)

    def forward(self, x):

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf detected in ResidualBlock input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        return x + y

# ============================================================
# Self-Attention
# ============================================================
class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch//8, 1)
        self.key   = nn.Conv2d(ch, ch//8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key(x).view(B, -1, H*W)
        att = torch.softmax(torch.bmm(q,k), dim=-1)
        v = self.value(x).view(B, -1, H*W)
        out = torch.bmm(v, att.permute(0,2,1)).view(B, C, H, W)
        return x + self.gamma * out

# ============================================================
# Upsample / Downsample Layer
# ============================================================
class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.resampler = InterpolativeUpsampler()
        self.proj = Convolution(in_ch, out_ch, 1)
    def forward(self, x): return self.proj(self.resampler(x))


class BiasedActivationReference(nn.Module):
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))
    Function = nn.LeakyReLU(0.2)

    def __init__(self, InputUnits):
        super().__init__()
        self.Bias = nn.Parameter(torch.zeros(InputUnits))

    def forward(self, x):
        y = x + self.Bias.to(x.dtype).view(1, -1, 1, 1) if x.ndim > 2 else x + self.Bias.to(x.dtype).view(1, -1)
        return BiasedActivationReference.Function(y)

BiasedActivation = BiasedActivationReference

def CreateLowpassKernel(weights, inplace=False):
    kernel = np.array([weights]) if inplace else np.convolve(weights, [1, 1]).reshape(1, -1)
    kernel = torch.Tensor(kernel.T @ kernel)
    return kernel / torch.sum(kernel)

class InterpolativeUpsampler(nn.Module):
    def __init__(self, Filter=[1,2,1]):
        super().__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter))
    def forward(self, x):
        return upfirdn2d.upsample2d(x, self.Kernel)

class InterpolativeDownsampler(nn.Module):
    def __init__(self, Filter=[1,2,1]):
        super().__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter))
    def forward(self, x):
        return upfirdn2d.downsample2d(x, self.Kernel)


def denormalize_auto(t, eps=1e-6):
    t = t.detach()
    tmin, tmax = float(t.min()), float(t.max())
    if tmax <= 1.0 + eps and tmin < -eps:
        t = t * 0.5 + 0.5
    elif tmax > 1.5:
        t = t / 255.0
    return t.clamp(0, 1)
 
# ======================
# Evaluation
# ======================
def evaluate(best_model_path, hr_zip, lr_zip, max_images=500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms (MUST match training)
    t_lr = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    t_hr = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImagePairZipDataset(hr_zip, lr_zip, t_hr, t_lr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # build generator
    G = Generator(
        WidthPerStage=[64,128,256],
        CardinalityPerStage=[4,4,4],
        BlocksPerStage=[2,2,2],
        ExpansionFactor=2
    ).to(device)

    # load EMA weights
    G.load_state_dict(torch.load(best_model_path, map_location=device))
    G.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            if i >= max_images:
                break

            lr = lr.to(device)
            hr = hr.to(device)

            sr = G(lr)

            sr = denormalize_auto(sr)
            hr = denormalize_auto(hr)

            psnr_vals.append(psnr_metric(sr, hr).item())
            ssim_vals.append(ssim_metric(sr, hr).item())

    psnr = float(np.mean(psnr_vals))
    ssim = float(np.mean(ssim_vals))

    print("==============================")
    print(f"PSNR : {psnr:.4f} dB")
    print(f"SSIM : {ssim:.6f}")
    print("==============================")

if __name__ == "__main__":
    evaluate(
        best_model_path="training_snapshots_64to256-FINAL-TRAIN-3/best_generator_ema.pth",
        hr_zip="datasets/256khr.zip",
        lr_zip="datasets/64klr.zip",
        max_images=500
    )
