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
        loss = 0.0
        loss = self.compute_layer(resotored, positive, negative)
        return loss
    
    def compute_layer(self, feature_a, feature_p, feature_n):
        result = self.criterion(feature_a, feature_p) / (self.criterion(feature_a, feature_n) + 1e-7)
        return result