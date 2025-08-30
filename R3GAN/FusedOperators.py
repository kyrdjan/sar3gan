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

    def __init__(self, InputUnits):
        super().__init__()
        self.Bias = nn.Parameter(torch.zeros(InputUnits))

    def forward(self, x):
        try:
            return bias_act.bias_act(
                x,
                self.Bias.to(x.dtype),
                act=BiasedActivationCUDA.Function,
                gain=1
            )
        except RuntimeError as e:
            # print(f"[Warning] Falling back to PyTorch BiasedActivationReference due to: {e}")
            # Fallback: just use reference version
            y = x + self.Bias.to(x.dtype).view(1, -1, 1, 1) if x.ndim > 2 else x + self.Bias.to(x.dtype).view(1, -1)
            return nn.functional.leaky_relu(y, negative_slope=0.2)


# Choose implementation
# If you want to **force disable CUDA plugin**, just set:
# BiasedActivation = BiasedActivationReference
# Otherwise, keep CUDA and auto-fallback:
BiasedActivation = BiasedActivationCUDA
