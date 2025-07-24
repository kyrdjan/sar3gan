# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Simplified MSR initializer
# def init_weights(layer, gain=1):
#     fan_in = layer.weight.size(1) * layer.weight[0][0].numel()
#     layer.weight.data.normal_(0, gain / math.sqrt(fan_in))
#     if layer.bias is not None:
#         layer.bias.data.zero_()
#     return layer

# # Basic Conv2d wrapper with MSR init
# def conv_layer(in_c, out_c, k, stride=1, padding=None, groups=1, bias=False, gain=1):
#     if padding is None:
#         padding = (k - 1) // 2
#     conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=bias)
#     return init_weights(conv, gain)

# # Biased Activation (Swish-like)
# class BiasedActivation(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
#     def forward(self, x):
#         return F.relu(x + self.bias)

# # Residual block with grouped conv
# class ResidualBlock(nn.Module):
#     def __init__(self, in_c, cardinality, expansion, k):
#         super().__init__()
#         mid_c = in_c * expansion // cardinality
#         self.block = nn.Sequential(
#             conv_layer(in_c, mid_c, 1),
#             BiasedActivation(mid_c),
#             conv_layer(mid_c, mid_c, k, groups=cardinality),
#             BiasedActivation(mid_c),
#             conv_layer(mid_c, in_c, 1, gain=0)
#         )
#     def forward(self, x):
#         return x + self.block(x)

# # Upsample + channel mapping
# class UpsampleLayer(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.adjust = conv_layer(in_c, out_c, 3) if in_c != out_c else nn.Identity()
#     def forward(self, x):
#         return self.adjust(self.upsample(x))

# # Downsample + channel mapping
# class DownsampleLayer(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.pool = nn.AvgPool2d(2)
#         self.adjust = conv_layer(in_c, out_c, 1) if in_c != out_c else nn.Identity()
#     def forward(self, x):
#         return self.adjust(self.pool(x))

# # Learnable starting tensor for generator
# class GeneratorBasis(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.input_map = conv_layer(in_c, out_c, 3)
#     def forward(self, z):
#         return self.input_map(z)

# # Global average + linear for disc output
# class DiscriminatorBasis(nn.Module):
#     def __init__(self, in_c, out_dim):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = init_weights(nn.Linear(in_c, out_dim, bias=False))
#     def forward(self, x):
#         return self.fc(self.pool(x).view(x.size(0), -1))

# # Generator stage
# class GeneratorStage(nn.Module):
#     def __init__(self, in_c, out_c, cardinality, blocks, expansion, k, upsample=False):
#         super().__init__()
#         layers = [UpsampleLayer(in_c, out_c)] if upsample else [GeneratorBasis(in_c, out_c)]
#         layers += [ResidualBlock(out_c, cardinality, expansion, k) for _ in range(blocks)]
#         self.model = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.model(x)

# # Discriminator stage
# class DiscriminatorStage(nn.Module):
#     def __init__(self, in_c, out_c, cardinality, blocks, expansion, k, downsample=False):
#         super().__init__()
#         layers = [ResidualBlock(in_c, cardinality, expansion, k) for _ in range(blocks)]
#         layers += [DownsampleLayer(in_c, out_c)] if downsample else [DiscriminatorBasis(in_c, out_c)]
#         self.model = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.model(x)

# # Full Generator
# class Generator(nn.Module):
#     def __init__(self, in_channels, widths, cards, blocks, expansion, k):
#         super().__init__()
#         self.stages = nn.ModuleList()
#         for i in range(len(widths)):
#             up = i > 0
#             in_c = in_channels if i == 0 else widths[i-1]
#             self.stages.append(GeneratorStage(in_c, widths[i], cards[i], blocks[i], expansion, k, upsample=up))
#         self.final = conv_layer(widths[-1], 3, 1)
#     def forward(self, x):
#         for stage in self.stages:
#             x = stage(x)
#         return self.final(x)

# # Full Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, widths, cards, blocks, expansion, k):
#         super().__init__()
#         self.extract = conv_layer(3, widths[0], 1)
#         self.stages = nn.ModuleList()
#         for i in range(len(widths)-1):
#             self.stages.append(DiscriminatorStage(widths[i], widths[i+1], cards[i], blocks[i], expansion, k, downsample=True))
#         self.stages.append(DiscriminatorStage(widths[-1], 1, cards[-1], blocks[-1], expansion, k))
#     def forward(self, x):
#         x = self.extract(x)
#         for stage in self.stages:
#             x = stage(x)
#         return x.view(x.size(0))
    

# # needs to understand 
# # should we reamove upsacale and downsacle to make this fix size input and oupt image but nbetter quality?




import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

