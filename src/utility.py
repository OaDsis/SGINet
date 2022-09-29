import os
import math
import time
import datetime
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import imageio

class timer():      # 计时工具
    def __init__(self):
        self.acc = 0
        self.tic()
    def tic(self):
        self.t0 = time.time()
    def toc(self):
        return time.time() - self.t0
    def hold(self):
        self.acc += self.toc()
    def release(self):
        ret = self.acc
        self.acc = 0
        return ret
    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')     # 输出年月日-时分秒格式

        if args.load == '.':        # 如果没有指定加载路径，则默认以时间格式来生成文件夹
            if args.save == '.': args.save = now        # 若没有输入存储目录，则以时间来生成文件夹
            self.dir = '../experiment/' + args.save     # 如果是第一次训练，则是在训练命令里面输入的存储目录，如'../experiment/200mm'
        else:
            self.dir = '../experiment/' + args.load     # 如果load路径不为空，则设定self.dir为对应地址
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')    # 加载psnr_log.pt文件
                print('Continue from epoch {}...'.format(len(self.log)))    # 打印当前epoch

        if args.reset:      # 删除文件夹下的数据并重新训练
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):    # 判断文件夹是否存在并建立
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'   # 判断log.txt是否存在，不存在则创建
        self.log_file = open(self.dir + '/log.txt', open_type)  # 打开log.txt文件进行操作
        with open(self.dir + '/config.txt', open_type) as f:        # 在config.txt中记录option.py里面的配置
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):          # 保存各种数据
        trainer.model.save(self.dir, epoch, is_best=is_best)    # 保存模型
        trainer.loss.save(self.dir)                             # 保存loss
        trainer.loss.plot_loss(self.dir, epoch)                 # 保存loss

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))     # 保存psnr的pytorch数据
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')          # 保存优化器的pytorch数据
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):    # 写入log.txt，log是需要写入的数据
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):     # 关闭log.txt
        self.log_file.close()

    def plot_psnr(self, epoch):     # 将每个epoch计算出来的psnr打印成pdf格式文件
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):     # 保存得到的图片，根据SR（去雨图片），LR（雨图），HR（干净）的顺序保存图片
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        #postfix = ('SR','LR', 'HR')
        postfix = ('SR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            # misc.imsave('{}{}.png'.format(filename, p), ndarr)
            imageio.imwrite('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):       # 用于限定图片的像素值在0-255范围内
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round()

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):       # 计算psnr
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

# 生成优化器的函数
def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':      # 默认是Adam
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)      # 返回优化器函数

# 生成优化策略的函数
def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

