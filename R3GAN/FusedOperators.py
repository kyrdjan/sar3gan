import torch
import torch.nn as nn
import math
from torch_utils.ops import bias_act

"""Avoid X is too large error for tensors"""
# class BiasedActivationReference(nn.Module):
#     Gain = math.sqrt(2 / (1 + 0.2 ** 2))
#     Function = nn.LeakyReLU(0.2)
    
#     def __init__(self, InputUnits):
#         super(BiasedActivationReference, self).__init__()
        
#         self.Bias = nn.Parameter(torch.empty(InputUnits))
#         self.Bias.data.zero_()
        
#     def forward(self, x):
#         y = x + self.Bias.to(x.dtype).view(1, -1, 1, 1) if len(x.shape) > 2 else x + self.Bias.to(x.dtype).view(1, -1)
#         return BiasedActivationReference.Function(y)

# class BiasedActivationCUDA(nn.Module):
#     Gain = math.sqrt(2 / (1 + 0.2 ** 2))
#     Function = 'lrelu'  
    
#     def __init__(self, InputUnits):
#         super(BiasedActivationCUDA, self).__init__()
        
#         self.Bias = nn.Parameter(torch.empty(InputUnits))
#         self.Bias.data.zero_()
        
#     def forward(self, x):
#         return bias_act.bias_act(x, self.Bias.to(x.dtype), act=BiasedActivationCUDA.Function, gain=1)

# BiasedActivation = BiasedActivationCUDA


class BiasedActivationReference(nn.Module):
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))
    Function = nn.LeakyReLU(0.2)

    def __init__(self, InputUnits):
        super().__init__()
        self.Bias = nn.Parameter(torch.zeros(InputUnits))

    def forward(self, x):
        # Add bias
        y = x + self.Bias.to(x.dtype).view(1, -1, 1, 1) if x.ndim > 2 else x + self.Bias.to(x.dtype).view(1, -1)
        # Apply LeakyReLU
        return BiasedActivationReference.Function(y)


class BiasedActivationCUDA(nn.Module):
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))
    Function = 'lrelu'
    MAX_ELEMS = 2**31 - 1  # CUDA kernel limit (~2.1B elements)

    def __init__(self, InputUnits):
        super().__init__()
        self.Bias = nn.Parameter(torch.zeros(InputUnits))

    def forward(self, x):
        bias = self.Bias.to(x.dtype)
        num_elems = x.numel()

        if num_elems > self.MAX_ELEMS:
            # 🚀 split into chunks to stay under CUDA limit, but still use fast kernel
            chunks = torch.chunk(x, math.ceil(num_elems / self.MAX_ELEMS), dim=0)
            outputs = []
            for chunk in chunks:
                outputs.append(bias_act.bias_act(chunk, bias, act=self.Function, gain=1))
            return torch.cat(outputs, dim=0)

        # Normal CUDA fast path
        return bias_act.bias_act(x, bias, act=self.Function, gain=1)


# ✅ Choose implementation
# Force PyTorch version (slower, safe):
BiasedActivation = BiasedActivationCUDA