# ============================================================
# test_inference.py — SAR3GAN 64x64 → 256x256
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torchvision import transforms as T
from torchvision.utils import save_image
# ============================================================
# Basic modules (Convolution, ResidualBlock, SelfAttention)
# ============================================================
import math

def MSRInitializer(layer, gain=1.0):
    if hasattr(layer, "weight"):
        nn.init.normal_(layer.weight, mean=0.0,
                        std=gain / math.sqrt(layer.weight.data.size(1) * layer.weight[0][0].numel()))
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

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

class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = (k - 1)//2
        self.layer = MSRInitializer(nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, groups=groups))
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.layer(x)

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
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        return x + y

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch//8, 1)
        self.key   = nn.Conv2d(ch, ch//8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B,C,H,W = x.shape
        q = self.query(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key(x).view(B, -1, H*W)
        att = torch.softmax(torch.bmm(q,k), dim=-1)
        v = self.value(x).view(B, -1, H*W)
        out = torch.bmm(v, att.permute(0,2,1)).view(B,C,H,W)
        return x + self.gamma * out

# ============================================================
# Upsample
# ============================================================
import numpy as np
from torch_utils.ops import upfirdn2d  # make sure this exists

def CreateLowpassKernel(weights):
    kernel = np.convolve(weights,[1,1]).reshape(1,-1)
    kernel = torch.Tensor(kernel.T @ kernel)
    return kernel / torch.sum(kernel)

class InterpolativeUpsampler(nn.Module):
    def __init__(self, Filter=[1,2,1]):
        super().__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter))
    def forward(self, x):
        return upfirdn2d.upsample2d(x, self.Kernel)

class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.resampler = InterpolativeUpsampler()
        self.proj = Convolution(in_ch, out_ch, 1)
    def forward(self, x):
        return self.proj(self.resampler(x))

# ============================================================
# GeneratorStage & Generator
# ============================================================
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super().__init__()
        self.DataType = DataType
        if ResamplingFilter is None:
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
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=3, KernelSize=3, ResamplingFilter=[1,2,1]):
        super().__init__()
        VarianceScalingParameter = sum(BlocksPerStage)
        in_ch = 3 if ConditionDimension is None else 3 + ConditionEmbeddingDimension
        MainLayers = [GeneratorStage(in_ch, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0],
                                     ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None)]
        MainLayers += [GeneratorStage(WidthPerStage[i], WidthPerStage[i+1], CardinalityPerStage[i+1],
                                      BlocksPerStage[i+1], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter)
                       for i in range(len(WidthPerStage)-1)]
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, k=1)
        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False))
    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            cond = self.EmbeddingLayer(y).view(y.shape[0], -1, 1, 1)
            x = torch.cat([x, cond.expand(-1,-1,x.shape[2],x.shape[3])], dim=1)
        for Layer in self.MainLayers:
            x = Layer(x)
        return torch.tanh(self.AggregationLayer(x))

# ============================================================
# Inference
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', type=str, required=True, help='Path to .pth generator weights')
    parser.add_argument('--lr-image', type=str, required=True, help='Path to LR input image (64x64)')
    parser.add_argument('--save-path', type=str, default='test/baseline/output.png')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create generator (must match training)
    width = [64,128,256]
    card = [4,4,4]
    blocks = [2,2,2]
    G = Generator(width, card, blocks, ExpansionFactor=2).to(device)
    G.load_state_dict(torch.load(args.generator, map_location=device))
    G.eval()

    # load LR image
    lr_img = Image.open(args.lr_image).convert("RGB")
    transform = T.Compose([T.Resize((64,64)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    lr_tensor = transform(lr_img).unsqueeze(0).to(device)

    # run inference
    with torch.no_grad():
        sr_tensor = G(lr_tensor)

    # save output
    sr_img = (sr_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0,1)
    save_image(sr_img, args.save_path)
    print(f">> SR output saved to {args.save_path}")
