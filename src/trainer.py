import os
import math
from decimal import Decimal
import utility
import IPython
import torch
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio 
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pdb
import torch.nn as nn

class Trainer():
    def __init__(self, args, loader, my_model, seg_model1, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.S = args.stage
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model1 = seg_model1
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.optimizer.param_groups[0]['lr']
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        self.model1.eval()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            coarse_pred = self.model(lr, None, None, True) * 255
            seg_pred = self.model1(coarse_pred)
            seg_gray = seg_pred.argmax(dim=1)
            seg_gray = seg_gray.unsqueeze(1).float()
            seg_rain = self.model1(lr)
            seg_norain = self.model1(hr)
            refined_pred = self.model(lr, coarse_pred, seg_gray, False) * 255
            loss = self.loss(refined_pred, hr, lr, coarse_pred, seg_pred * 255, seg_norain * 255, seg_rain * 255)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0: 
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        print(loss)
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch# + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        self.model1.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                if self.args.test_only == True:
                    for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                        filename = filename[0]
                        no_eval = (hr.nelement() == 1)
                        if not no_eval:
                            lr, hr = self.prepare(lr, hr)
                        else:
                            lr, = self.prepare(lr)

                        coarse_pred = self.model(lr, None, None, True) * 255
                        seg_pred = self.model1(coarse_pred)
                        seg_gray = seg_pred.argmax(dim=1)
                        seg_gray = seg_gray.unsqueeze(1).float()
                        refined_pred = self.model(lr, coarse_pred, seg_gray, False) * 255

                        sr = utility.quantize(refined_pred, self.args.rgb_range)
                        save_list = [sr]
                        if not no_eval:
                            eval_acc += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=self.loader_test.dataset.benchmark
                            )

                        if self.args.save_results:
                            self.ckp.save_results(filename, save_list, scale)
                else:
                    for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                        filename = filename[0]
                        no_eval = (hr.nelement() == 1)
                        if not no_eval:
                            lr, hr = self.prepare(lr, hr)
                        else:
                            lr, = self.prepare(lr)

                        coarse_pred = self.model(lr, None, None, True) * 255
                        seg_pred = self.model1(coarse_pred)
                        seg_gray = seg_pred.argmax(dim=1)
                        seg_gray = seg_gray.unsqueeze(1).float()
                        refined_pred = self.model(lr, coarse_pred, seg_gray, False) * 255

                        sr = utility.quantize(refined_pred, self.args.rgb_range)
                        save_list = [sr]
                        if not no_eval:
                            eval_acc += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=self.loader_test.dataset.benchmark
                            )

                        if self.args.save_results:
                            self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            torch.save(self.model1.state_dict(), os.path.join("../experiment/", self.args.save, 'model', 'pspnet_latest.pt'))
            if (best[1][0] + 1 == epoch):
                torch.save(self.model1.state_dict(), os.path.join("../experiment/", self.args.save, 'model', 'pspnet_best.pt'))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs
