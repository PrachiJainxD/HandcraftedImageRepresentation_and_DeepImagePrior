import numpy as np
import torch
import torch.nn as nn

class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()
        self.ConvBlocks = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(True)
        )
        
        self.DeconvBlocks = nn.Sequential(
            nn.Upsample(mode = 'bilinear', scale_factor = 2),
            nn.Conv2d(in_channels = 64,out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(True),
            
            nn.Upsample(mode='bilinear', scale_factor = 2),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU(True), 
            
            nn.Upsample(mode='bilinear', scale_factor =2),
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(True)          
        )

    def forward(self, x):
        x = self.ConvBlocks(x)
        x = self.DeconvBlocks(x)
        return x          