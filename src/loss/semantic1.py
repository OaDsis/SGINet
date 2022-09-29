import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from math import exp
import pdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# loader使用torchvision中自带的transforms函数
loader = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])  

unloader = torchvision.transforms.ToPILImage()

class Semantic_Loss(nn.Module):
    def __init__(self, args, model1, model2, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Semantic_Loss, self).__init__()
        self.psp50 = model2
        # self.criterion = torch.nn.KLDivLoss()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.args = args

    def forward(self, resotored, positive, negative):
        # a: resotored imageconda activate pytorch0.4.0/1.4
        # p: ground-truth image
        # n: input low-quality image
        a_s, p_s, n_s = self.psp50(resotored), self.psp50(positive), self.psp50(negative)
        a_h = self.computer_his(a_s)
        p_h = self.computer_his(p_s)
        n_h = self.computer_his(n_s)
        #pdb.set_trace()
        loss = self.compute_layer(a_h, p_h, n_h)
        return loss
    
    def compute_layer(self, his_a, his_p, his_n):
        loss = self.criterion(his_a, his_p) / (self.criterion(his_a, his_n) + 1e-7)
        return loss

    def computer_his(self, segmant):
        batch, channel, _, _ = segmant.shape[0], segmant.shape[1], segmant.shape[2], segmant.shape[3]
        for i in range(batch):       #   segmant[i] = (1,3,256,256)
            # image = torch.cat([segmant[i][0].unsqueeze(0), segmant[i][8].unsqueeze(0), segmant[i][10].unsqueeze(0)], 0)
            img = segmant[i].cpu().detach().numpy().transpose((1, 2, 0))
            hist0 = cv2.calcHist([img],[0],None,[256],[0,255])
            hist8 = cv2.calcHist([img],[8],None,[256],[0,255])
            hist10 = cv2.calcHist([img],[10],None,[256],[0,255])
            hist_total = hist0 + hist8 + hist10
            hist_total = torch.from_numpy(hist_total).float().cuda()
        return hist_total

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def tensor_to_PIL(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image