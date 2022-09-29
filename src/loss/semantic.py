import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
from math import exp
import pdb

class Semantic_Loss(nn.Module):
    def __init__(self, model1, model2, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Semantic_Loss, self).__init__()
        self.vgg = model1
        self.psp50 = model2
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def forward(self, img1, img2, img3, img4, resotored, positive, negative):
        # a: resotored image
        # p: ground-truth image
        # n: input low-quality image
        # a_s, p_s, n_s = self.psp50(resotored), self.psp50(positive), self.psp50(negative)
        # a_vgg, p_vgg, n_vgg = self.vgg(a_s), self.vgg(p_s), self.vgg(n_s)
        
        loss = 0.0
        loss = self.compute_layer(resotored, positive, negative)
        # loss += self.weights[0] * self.compute_layer(a_vgg['relu1_1'], p_vgg['relu1_1'], n_vgg['relu1_1'])
        # loss += self.weights[1] * self.compute_layer(a_vgg['relu2_1'], p_vgg['relu2_1'], n_vgg['relu2_1'])
        # loss += self.weights[2] * self.compute_layer(a_vgg['relu3_1'], p_vgg['relu3_1'], n_vgg['relu3_1'])
        # loss += self.weights[3] * self.compute_layer(a_vgg['relu4_1'], p_vgg['relu4_1'], n_vgg['relu4_1'])
        # loss += self.weights[4] * self.compute_layer(a_vgg['relu5_1'], p_vgg['relu5_1'], n_vgg['relu5_1'])
        return loss
    
    def compute_layer(self, feature_a, feature_p, feature_n):
        result = self.criterion(feature_a, feature_p) / (self.criterion(feature_a, feature_n) + 1e-7)
        return result