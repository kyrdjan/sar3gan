import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()

    print('MRSI')
    return Layer

class Convolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1):
        super(Convolution, self).__init__()
        
        self.Layer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Groups, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        print('Conv')
        return nn.functional.conv2d(x, self.Layer.weight.to(x.dtype), padding=self.Layer.padding, groups=self.Layer.groups)



# Self-Attention Layer (Simplified for 2D feature maps)
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        print('SA')
        B, C, H, W = x.size()
        x = x.to(self.query.weight.dtype)
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key(x).view(B, -1, H * W)                       # B x C' x N
        proj_value = self.value(x).view(B, -1, H * W)                   # B x C x N

        attention = torch.bmm(proj_query, proj_key)                     # B x N x N
        attention = torch.softmax(attention, dim=-1)                    # B x N x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # B x C x N
        out = out.view(B, C, H, W)

        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter):
        super(ResidualBlock, self).__init__()
        
        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))
        
        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, KernelSize=KernelSize, Groups=Cardinality, ActivationGain=ActivationGain)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)
        
    def forward(self, x):

        print('ResNet')
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        print('UP')

        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        print('DS')
        
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        
        return x
    
# class GenerativeBasis(nn.Module): # OLD
#     def __init__(self, InputDimension, OutputChannels):
#         super(GenerativeBasis, self).__init__()
        
#         self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
#         self.LinearLayer = MSRInitializer(nn.Linear(InputDimension, OutputChannels, bias=False))
        
#     def forward(self, x):
#         return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)


# TODO: FIX  Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 0]
class GenerativeBasis(nn.Module):  # NEW
    def __init__(self, InputChannels, OutputChannels):
        super().__init__()
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        self.Conv = Convolution(InputChannels, OutputChannels, KernelSize=1, ActivationGain=1)

    def forward(self, x):
        print('GenerativeBasis')
        print(x)
        # Assumes x is already [B, C, 4, 4]
        return self.Conv(x) * self.Basis.view(1, -1, 4, 4) # PROBLEM: RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 0]



class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, OutputDimension, bias=False))
        
    def forward(self, x):

        print('DiscriminativeBasis')
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))
    


class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super(GeneratorStage, self).__init__()

        self.DataType = DataType

        # NEW
        if ResamplingFilter is None:
            TransitionLayer = GenerativeBasis(InputChannels, OutputChannels)
            self.Layers = nn.ModuleList([
                TransitionLayer,
                *[ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)],
                SelfAttention(OutputChannels) # SELF ATTENTION IS HEEREEEEEEEEEEEEEEEEEEEEE -----------
            ])
        else:
            TransitionLayer = UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
            self.Layers = nn.ModuleList([
                TransitionLayer,
                *[ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)]
            ])

        # OLD
        # TransitionLayer = GenerativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        # self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)])
  

    def forward(self, x):

        print('GeneratorStage')
        x = x.to(self.DataType)

        for Layer in self.Layers:
            x = Layer(x)

        return x

    
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super(DiscriminatorStage, self).__init__()
        
        # NEW
        if ResamplingFilter is None:
            TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels)
            self.Layers = nn.ModuleList(
                [SelfAttention(InputChannels)] +
                [ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)] +
                [TransitionLayer]
            )
        else:
            TransitionLayer = DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
            self.Layers = nn.ModuleList([
                ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)] 
                + [TransitionLayer]
            )


        # OLD
        # TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        # self.Layers = nn.ModuleList([ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        
        
        self.DataType = DataType
        
    def forward(self, x):
        print('DiscriminatorStage')
        
        x = x.to(self.DataType)
        
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class Generator(nn.Module):
    # def __init__(self, NoiseDimension, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=3, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        VarianceScalingParameter = sum(BlocksPerStage)
        #MainLayers = [GeneratorStage(NoiseDimension + ConditionEmbeddingDimension, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0], ExpansionFactor, KernelSize, VarianceScalingParameter)]
        MainLayers = [GeneratorStage(ConditionEmbeddingDimension, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0], ExpansionFactor, KernelSize, VarianceScalingParameter)]
        MainLayers += [GeneratorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x + 1], BlocksPerStage[x + 1], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False))
        
    def forward(self, x, y=None):

        print('Generator')
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        return self.AggregationLayer(x)
    
class Discriminator(nn.Module):
    #def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=3, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [DiscriminatorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        MainLayers += [DiscriminatorStage(WidthPerStage[-1], 1 if ConditionDimension is None else ConditionEmbeddingDimension, CardinalityPerStage[-1], BlocksPerStage[-1], ExpansionFactor, KernelSize, VarianceScalingParameter)]
        
        self.ExtractionLayer = Convolution(3, WidthPerStage[0], KernelSize=1)
        self.MainLayers = nn.ModuleList(MainLayers)
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False), ActivationGain=1 / math.sqrt(ConditionEmbeddingDimension))
        
    def forward(self, x, y=None):

        print('Discriminator')
        x = self.ExtractionLayer(x.to(self.MainLayers[0].DataType))
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        x = (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
