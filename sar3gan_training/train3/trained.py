# ============================================================
# cheapsar3ganfullcopy_resblock.py — 64×64 → 256×256 with RaGAN loss + EMA + schedulers + ResidualBlocks
# Updated: replaced OldGenerator/OldDiscriminator with new modular Generator/Discriminator
# ============================================================
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay, self.device = decay, device
        self.shadow = {}
        self.register(model)

    def register(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach().to(self.device if self.device else p.device)

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def copy_to_model(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])

    def to(self, device):
        self.device = device
        for k in list(self.shadow.keys()):
            self.shadow[k] = self.shadow[k].to(device)
        return self

# ============================================================
# Scheduler
# ============================================================
def get_scheduler(optimizer, name="step", **kwargs):
    name = name.lower()
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 1000), gamma=kwargs.get("gamma", 0.5)
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("T_max", 1000), eta_min=kwargs.get("eta_min", 1e-6)
        )
    elif name == "warmup_cosine":
        warmup_steps, total_steps = kwargs.get("warmup_steps", 500), kwargs.get("total_steps", 10000)
        min_lr = kwargs.get("min_lr", 1e-6)
        max_lr = optimizer.param_groups[0]['lr']
        def lr_lambda(step):
            if step < warmup_steps:
                return step / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / max_lr + (1 - min_lr / max_lr) * cosine_decay
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "linear":
        total_steps = kwargs.get("total_steps", 10000)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / float(total_steps)))
    else:
        raise ValueError(f"Unknown scheduler: {name}")

# ============================================================
# FIR Kernel and Resamplers
# ============================================================
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

# ============================================================
# Biased Activation
# ============================================================
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

# ============================================================
# Convolution Layer with MSR Init
# ============================================================
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

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.resampler = InterpolativeDownsampler()
        self.proj = Convolution(in_ch, out_ch, 1)
    def forward(self, x): return self.proj(self.resampler(x))

# ============================================================
# New Generative / Discriminative Basis + Stages + Networks
# ============================================================

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

class DiscriminativeBasis(nn.Module):
    """
    Depthwise conv + linear projection for discriminator when Transition
    is 'basis' (i.e., the last stage that pools to a vector).
    """
    def __init__(self, input_channels, output_dimension):
        super().__init__()
        # Depthwise conv to transform feature maps, followed by linear
        self.Basis = MSRInitializer(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(input_channels, output_dimension, bias=False))

    def forward(self, x):
        x = self.Basis(x)                # [N, C, H, W]
        x = x.mean(dim=[2, 3])           # global average pool -> [N, C]
        return self.LinearLayer(x)       # [N, OutDim]

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
        if x.shape[-1] == 64:
            x = self.Attention(x)
        for block in self.Blocks:
            x = block(x)
        return x

class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super().__init__()
        self.DataType = DataType

        # Residual blocks operate at InputChannels resolution
        self.Blocks = nn.ModuleList([
            ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)
            for _ in range(NumberOfBlocks)
        ])

        self.Attention = SelfAttention(InputChannels)

        if ResamplingFilter is None:
            # Terminal stage: pool -> linear
            self.Transition = nn.AdaptiveAvgPool2d((1, 1))
            self.Linear = MSRInitializer(nn.Linear(InputChannels, OutputChannels, bias=False))
        else:
            # Downsample transition
            self.Transition = DownsampleLayer(InputChannels, OutputChannels)
            self.Linear = None

    def forward(self, x):
        x = x.to(self.DataType)
        if x.shape[-1] == 64: # self attention
            x = self.Attention(x)
        for block in self.Blocks:
            x = block(x)
        if isinstance(self.Transition, nn.AdaptiveAvgPool2d):
            x = self.Transition(x).view(x.size(0), -1)
            x = self.Linear(x)
        else:
            x = self.Transition(x)
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
    
class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=3, KernelSize=3, ResamplingFilter=[1,2,1]):
        super().__init__()

        VarianceScalingParameter = sum(BlocksPerStage)

        in_ch = 3 if ConditionDimension is None else 3 + ConditionEmbeddingDimension
        self.ExtractionLayer = Convolution(in_ch, WidthPerStage[0], k=1)

        MainLayers = [
            DiscriminatorStage(WidthPerStage[i], WidthPerStage[i + 1], CardinalityPerStage[i], BlocksPerStage[i],
                               ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter)
            for i in range(len(WidthPerStage) - 1)
        ]

        # ask final stage to output a single unit
        MainLayers += [
            DiscriminatorStage(WidthPerStage[-1], 1, CardinalityPerStage[-1], BlocksPerStage[-1],
                            ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None)
        ]

        self.MainLayers = nn.ModuleList(MainLayers)

        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False), gain=1 / math.sqrt(ConditionEmbeddingDimension))

    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            cond = self.EmbeddingLayer(y).view(y.shape[0], -1, 1, 1)
            x = torch.cat([x, cond.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)

        x = self.ExtractionLayer(x)
        for Layer in self.MainLayers:
            x = Layer(x)

        # final output: scalar per sample
        # returned shape [B]
        return x.view(x.size(0))

# ============================================================
# Dataset Loader (unchanged)
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
# Logger (unchanged)
# ============================================================
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================================
# Utility functions (unchanged)
# ============================================================
def denormalize_auto(t, eps=1e-6):
    t = t.detach()
    tmin, tmax = float(t.min()), float(t.max())
    if tmax <= 1.0 + eps and tmin < -eps:
        t = t * 0.5 + 0.5
    elif tmax > 1.5:
        t = t / 255.0
    return t.clamp(0, 1)

def save_with_labels(tensors, labels, filename, nrow=3, target_size=(256, 256)):
    import torchvision.transforms.functional as TF
    processed = []
    for t in tensors:
        t = denormalize_auto(t.cpu())
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if target_size is not None:
            t = F.interpolate(t.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
        processed.append(t)
    grid = torch.cat(processed, dim=2)
    grid_img = TF.to_pil_image(grid)
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    w = target_size[0]
    for i, label in enumerate(labels):
        draw.text((i * w + 10, 10), label, fill=(255, 255, 255), font=font)
    grid_img.save(filename)

# ============================================================
# Adversarial Training (R3GAN) 
# ============================================================
class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])

    def AccumulateGeneratorGradients(self, Lr, Hr, Scale=1, Preprocessor=lambda x: x):
        Fake = self.Generator(Lr)
        Hr = Hr.detach()
        FakeLogits = self.Discriminator(Preprocessor(Fake))
        RealLogits = self.Discriminator(Preprocessor(Hr))
        Relativistic = FakeLogits - RealLogits
        AdvLoss = F.softplus(-Relativistic)
        
        # Check for NaN/Inf BEFORE backward
        if torch.isnan(AdvLoss).any() or torch.isinf(AdvLoss).any():
            print("WARNING: NaN/Inf in Generator loss, skipping update")
            return torch.tensor(0.0).to(Lr.device), Fake.detach()
        
        (Scale * AdvLoss.mean()).backward()
        return AdvLoss.mean().detach(), Fake.detach()

    def AccumulateDiscriminatorGradients(self, Lr, Hr, Gamma=10.0, Scale=1, Preprocessor=lambda x: x):
        Hr = Hr.detach().requires_grad_(True)   
        with torch.no_grad():
            Fake = self.Generator(Lr)
        Fake = Fake.detach().requires_grad_(True)
        
        RealLogits = self.Discriminator(Preprocessor(Hr))
        FakeLogits = self.Discriminator(Preprocessor(Fake))
        R1 = self.ZeroCenteredGradientPenalty(Hr, RealLogits)
        R2 = self.ZeroCenteredGradientPenalty(Fake, FakeLogits)
        Relativistic = RealLogits - FakeLogits
        AdvLoss = F.softplus(-Relativistic)
        DLoss = AdvLoss + (Gamma / 2) * (R1 + R2)
        
        # Check for NaN/Inf BEFORE backward
        if torch.isnan(DLoss).any() or torch.isinf(DLoss).any():
            print("WARNING: NaN/Inf in Discriminator loss, skipping update")
            return torch.tensor(0.0).to(Lr.device), torch.tensor(0.0).to(Lr.device), torch.tensor(0.0).to(Lr.device)
        
        (Scale * DLoss.mean()).backward()
        return AdvLoss.mean().detach(), R1.mean().detach(), R2.mean().detach()

# ============================================================
# FID evaluation (unchanged)
# ============================================================
def evaluate_fid(G_model, dataloader, device, step, save_dir="fid_eval", num_images=50000):
    real_dir, fake_dir = os.path.join(save_dir, "real"), os.path.join(save_dir, "fake")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(real_dir); os.makedirs(fake_dir)
    G_model.eval()
    count = 0
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            fake = G_model(lr)
            for i in range(lr.size(0)):
                if count >= num_images: break
                save_image(denormalize_auto(hr[i]), f"{real_dir}/{count}.png")
                save_image(denormalize_auto(fake[i]), f"{fake_dir}/{count}.png")
                count += 1
            if count >= num_images: break
    fid = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=2, device=device, dims=2048)
    print(f"Step {step}: FID = {fid:.2f}")
    G_model.train()
    return fid

def save_checkpoint(G, D, g_opt, d_opt, g_sched, d_sched, ema, step, epoch, best_fid, save_dir):
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'best_fid': best_fid,  # Add this line
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'g_optimizer_state_dict': g_opt.state_dict(),
        'd_optimizer_state_dict': d_opt.state_dict(),
        'g_scheduler_state_dict': g_sched.state_dict(),
        'd_scheduler_state_dict': d_sched.state_dict(),
        'ema_shadow': ema.shadow,
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_step_{step}.pth'))
    torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
    print(f">> Checkpoint saved at step {step}")

def load_checkpoint(checkpoint_path, G, D, g_opt, d_opt, g_sched, d_sched, ema, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    g_opt.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_opt.load_state_dict(checkpoint['d_optimizer_state_dict'])
    g_sched.load_state_dict(checkpoint['g_scheduler_state_dict'])
    d_sched.load_state_dict(checkpoint['d_scheduler_state_dict'])
    ema.shadow = checkpoint['ema_shadow']
    return checkpoint['step'], checkpoint['epoch'], checkpoint.get('best_fid', float("inf"))

def load_weights_only(weights_path, G, device, start_step=0, start_epoch=0):
    """Load only model weights from old training (no optimizer/scheduler states)"""
    print(f"Loading weights only from {weights_path}...")
    G.load_state_dict(torch.load(weights_path, map_location=device))
    print(f">> Weights loaded successfully. Resuming from step {start_step}, epoch {start_epoch}")
    return start_step, start_epoch, float("inf")  # return specified step/epoch, best_fid=inf

# ============================================================
# Main training loop (uses new Generator / Discriminator)
# ============================================================
def main(resume_from=None, weights_only=None, start_step=0, start_epoch=0):
    save_dir = "training_snapshots_64to256"
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "log.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms
    transform_lr = transforms.Compose([
        transforms.Resize((64,64)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_hr = transforms.Compose([
        transforms.Resize((256,256)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # dataset (adjust paths)
    dataset = ImagePairZipDataset("datasets/256khr.zip", "datasets/64klr.zip",
                                  transform_hr=transform_hr, transform_lr=transform_lr)
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # models
    width = [64, 128, 256]
    card = [4, 4, 4]
    gen_blocks = [2, 2, 2]
    disc_blocks = [2, 2, 2]
    expansion = 2

    G = Generator(width, card, gen_blocks, ExpansionFactor=expansion).to(device)
    D = Discriminator(width, card, disc_blocks, ExpansionFactor=expansion).to(device)

    print("Generator Summary:"); summary(G, input_size=(1,3,64,64), device=device)
    print("Discriminator Summary:"); summary(D, input_size=(1,3,256,256), device=device)

    trainer = AdversarialTraining(G, D)
    ema = EMA(G, decay=0.999).to(device)

    # optimizers
    g_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
    d_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))

    # schedulers
    total_epochs = 50
    total_steps = total_epochs * len(loader)
    g_sched = get_scheduler(g_opt, name="warmup_cosine", warmup_steps=500, total_steps=total_steps, min_lr=2e-5)
    d_sched = get_scheduler(d_opt, name="warmup_cosine", warmup_steps=500, total_steps=total_steps, min_lr=2e-5)

    # training state
    best_fid = float("inf")
    
    # Resume from full checkpoint OR load weights only
    if resume_from:
        start_step, start_epoch, best_fid = load_checkpoint(
            resume_from, G, D, g_opt, d_opt, g_sched, d_sched, ema, device
        )
        print(f">> Resumed from step {start_step}, epoch {start_epoch}, best FID: {best_fid:.2f}")
    elif weights_only:
        start_step, start_epoch, best_fid = load_weights_only(weights_only, G, device, start_step, start_epoch)
        # Advance schedulers to the correct position
        for _ in range(start_step):
            g_sched.step()
            d_sched.step()
        print(f">> Loaded weights. Schedulers advanced to step {start_step}")
    
    step = start_step
    epochs = total_epochs

    # Calculate which epoch we're in based on step
    steps_per_epoch = len(loader)
    current_epoch = start_epoch if start_epoch > 0 else (start_step // steps_per_epoch)
    
    print(f">> Starting training from epoch {current_epoch}, step {step}")

    # training loop
    for epoch in range(current_epoch, epochs):
        torch.cuda.empty_cache()
        pbar = tqdm(enumerate(loader), total=len(loader), ncols=120)
        for i, (lr_imgs, hr_imgs) in pbar:
            # Skip iterations if resuming mid-epoch
            current_iter = epoch * steps_per_epoch + i
            if current_iter < start_step:
                continue
                
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Discriminator step
            d_opt.zero_grad()
            d_loss, r1, r2 = trainer.AccumulateDiscriminatorGradients(lr_imgs, hr_imgs)
            d_opt.step()

            # Generator step
            g_opt.zero_grad()
            g_loss, fake = trainer.AccumulateGeneratorGradients(lr_imgs, hr_imgs)
            g_opt.step()

            # update schedulers (per step)
            try:
                g_sched.step()
                d_sched.step()
            except Exception:
                pass

            # update EMA
            ema.update(G)

            step += 1

            # logging & snapshots
            if step % 500 == 0:
                save_with_labels([lr_imgs[0], fake[0], hr_imgs[0]], ["LR","SR","HR"], f"{save_dir}/step_{step:06d}.png")
                print(f"Step {step:06d} | G:{g_loss:.4f} | D:{d_loss:.4f} | R1:{r1:.4f} | R2:{r2:.4f}")

            # Save checkpoint periodically
            if step % 2000 == 0:
                save_checkpoint(G, D, g_opt, d_opt, g_sched, d_sched, ema, step, epoch, best_fid, save_dir)

            # evaluate FID using EMA model periodically
            if step % 1000 == 0:
                G_ema_copy = copy.deepcopy(G)
                ema.copy_to_model(G_ema_copy)
                fid = evaluate_fid(G_ema_copy, loader, device, step, save_dir=os.path.join(save_dir, "fid_eval"), num_images=50000)
                if fid < best_fid:
                    best_fid = fid
                    best_path = os.path.join(save_dir, "best_generator_ema.pth")
                    torch.save(G_ema_copy.state_dict(), best_path)
                    print(f">> New best FID: {fid:.2f} at step {step}, saved {best_path}")
                    save_with_labels([lr_imgs[0], fake[0], hr_imgs[0]], ["LR","SR","HR"], f"{save_dir}/best_sample_at_step_{step:06d}.png")
                    save_checkpoint(G, D, g_opt, d_opt, g_sched, d_sched, ema, step, epoch, best_fid, save_dir)

            pbar.set_description(f"ep{epoch+1}/{epochs} step{step} G{g_loss:.4f} D{d_loss:.4f}")

    # final save
    torch.save(G.state_dict(), os.path.join(save_dir, "final_generator.pth"))
    G_ema_final = copy.deepcopy(G)
    ema.copy_to_model(G_ema_final)
    torch.save(G_ema_final.state_dict(), os.path.join(save_dir, "final_generator_ema.pth"))
    torch.save(D.state_dict(), os.path.join(save_dir, "final_discriminator.pth"))
    save_checkpoint(G, D, g_opt, d_opt, g_sched, d_sched, ema, step, epoch, best_fid, save_dir)
    print(">> Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to full checkpoint to resume from')
    parser.add_argument('--weights-only', type=str, default=None, help='Path to old .pth with weights only')
    parser.add_argument('--start-step', type=int, default=0, help='Step number to resume from (when using --weights-only)')
    parser.add_argument('--start-epoch', type=int, default=0, help='Epoch number to resume from (when using --weights-only)')
    args = parser.parse_args()
    
    main(resume_from=args.resume, weights_only=args.weights_only, 
         start_step=args.start_step, start_epoch=args.start_epoch)