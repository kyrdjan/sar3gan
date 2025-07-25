# import torch
# import torch.nn as nn
# import copy
# import R3GAN.Networks

# class Generator(nn.Module):
#     def __init__(self, *args, **kw):
#         super(Generator, self).__init__()
        
#         config = copy.deepcopy(kw)
#         del config['FP16Stages'] # 16-floating point
#         del config['c_dim'] # for conditioning of the input image 
#         del config['img_resolution'] 
        
#         if kw['c_dim'] != 0:
#             config['ConditionDimension'] = kw['c_dim']
        
#         self.Model = R3GAN.Networks.Generator(*args, **config)
#         self.z_dim = kw['NoiseDimension'] # should be change of low resolution image
#         self.c_dim = kw['c_dim']
#         self.img_resolution = kw['img_resolution']
        
#         for x in kw['FP16Stages']:
#             self.Model.MainLayers[x].DataType = torch.bfloat16
        
#     def forward(self, x, c):
#         return self.Model(x, c)
    
# class Discriminator(nn.Module):
#     def __init__(self, *args, **kw):
#         super(Discriminator, self).__init__()
        
#         config = copy.deepcopy(kw)
#         del config['FP16Stages']
#         del config['c_dim']
#         del config['img_resolution']
        
#         if kw['c_dim'] != 0:
#             config['ConditionDimension'] = kw['c_dim']
        
#         self.Model = R3GAN.Networks.Discriminator(*args, **config)
        
#         for x in kw['FP16Stages']:
#             self.Model.MainLayers[x].DataType = torch.bfloat16
        
#     def forward(self, x, c):
#         return self.Model(x, c)

#--------------- FOR SAR3GAN -----------------------------------

import torch
import torch.nn as nn
import copy
import R3GAN.Networks

class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Generator(*args, **config)

        # In paired setting, z_dim becomes irrelevant — comment out
        # self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16

    def forward(self, lr_image, c=None):
        # NOTE: Assuming the Generator can take an image instead of z
        return self.Model(lr_image, c)


class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Discriminator(*args, **config)
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16
        
    def forward(self, hr_image, c=None):
        return self.Model(hr_image, c)
