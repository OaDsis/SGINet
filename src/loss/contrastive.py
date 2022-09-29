import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
from math import exp

class Contrastive_Loss(nn.Module):
    def __init__(self, model, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Contrastive_Loss, self).__init__()
        self.vgg = model
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def forward(self, a, p, n):
        # a: resotored image
        # p: ground-truth image
        # n: input low-quality image
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        
        loss = 0.0      
        loss += self.weights[0] * self.compute_layer(a_vgg['relu1_1'], p_vgg['relu1_1'], n_vgg['relu1_1'])
        loss += self.weights[1] * self.compute_layer(a_vgg['relu2_1'], p_vgg['relu2_1'], n_vgg['relu2_1'])
        loss += self.weights[2] * self.compute_layer(a_vgg['relu3_1'], p_vgg['relu3_1'], n_vgg['relu3_1'])
        loss += self.weights[3] * self.compute_layer(a_vgg['relu4_1'], p_vgg['relu4_1'], n_vgg['relu4_1'])
        loss += self.weights[4] * self.compute_layer(a_vgg['relu5_1'], p_vgg['relu5_1'], n_vgg['relu5_1'])
        return loss
    
    def compute_layer(self, feature_a, feature_p, feature_n):
        result = self.criterion(feature_a, feature_p) / (self.criterion(feature_a, feature_n) + 1e-7)
        return result