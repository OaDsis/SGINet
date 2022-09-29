import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
from math import exp

class L_TV(nn.Module):
    
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()

    def forward(self,x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv