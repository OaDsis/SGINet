import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from model.degnet import deg_net
import time
import model.pspnet as seg
import torch.nn as nn


parser = argparse.ArgumentParser(description="RCDNet_Test")
parser.add_argument("--model_dir", type=str, default="./syn100l_models", help='path to model files')
parser.add_argument("--data_path", type=str, default="./syn100l/test/small/rain", help='path to testing data')
parser.add_argument('--num_M', type=int, default=32, help='the number of rain maps')
parser.add_argument('--num_Z', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=17, help='the number of iterative stages in RCDNet')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="./derained/syn100l", help='path to derained results')
opt = parser.parse_args()


try:
    os.makedirs(opt.save_path)
except OSError:
    pass

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def main():
    # Build model
    print('Loading model ...\n')
    model = deg_net().cuda()
    seg_model = seg.PSPNet()
    seg_model = nn.DataParallel(seg_model)
    seg_model = seg_model.cuda()
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'model_best.pt')))
    seg_model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'pspnet_best.pt')))
    model.eval()
    seg_model.eval()
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            # input image
            O = cv2.imread(img_path)
            b, g, r = cv2.split(O)
            O = cv2.merge([r, g, b])
            O = np.expand_dims(O.transpose(2, 0, 1), 0)
            O = Variable(torch.Tensor(O))
            if opt.use_GPU:
                O = O.cuda()
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                coarse_pred = model(O, None, None, True) * 255   # 模型推导过程
                seg_pred = seg_model(coarse_pred)
                seg_gray = seg_pred.argmax(dim=1)
                seg_gray = seg_gray.unsqueeze(1).float()
                out = model(O, coarse_pred, seg_gray, False) * 255

                out = torch.clamp(out, 0., 255.)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
                print(img_name, ': ', dur_time)
            if opt.use_GPU:
                save_out = np.uint8(out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(out.data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)
            count += 1
    print('Avg. time:', time_test/count)
if __name__ == "__main__":
    main()

