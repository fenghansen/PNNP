import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from .noise_flow import NoiseFlow
from .Unet import *
from .ResUnet import *
# from .PNNP import *

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.normal_(0.0, 0.02)
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)

if __name__ == '__main__':
    pass