import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.nn as nn
import model.vgg as vgg
import model.pspnet as seg
import pdb

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        vgg_model = vgg.VGG16()
        vgg_model = nn.DataParallel(vgg_model)
        vgg_model = vgg_model.cuda()
        seg_model = seg.PSPNet()
        seg_model = nn.DataParallel(seg_model)
        seg_model = seg_model.cuda()
        checkpoint1 = torch.load(os.path.join(args.model_dir, 'train_epoch_200_psp101.pth'))
        seg_model.load_state_dict(checkpoint1['state_dict'],strict=False)
        print_network(model)
        loss = loss.Loss(args, checkpoint, vgg_model, seg_model) if not args.test_only else None
        t = Trainer(args, loader, model, seg_model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()