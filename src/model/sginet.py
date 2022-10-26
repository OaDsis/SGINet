from re import X
from readline import set_completion_display_matches_hook
import torch
import torch.nn as nn

def make_model():
    return sginet()

class sginet(nn.Module):
    def __init__(self):
        super(sginet, self).__init__()
        self.u_net_light = fr_net(3, 3)
        self.u_net_heavy = u_net_heavy(6, 3)

    def forward(self, x, y, z, coarse):
        if coarse== True:
            coarse_image = self.u_net_light(x)
            return coarse_image
        elif coarse== False:
            refined_image = self.u_net_heavy(x, y, z)
            return refined_image
        
class u_net_light(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(u_net_light, self).__init__()
        
        # initial
        num = 32
        
        # encoder        
        self.encoder_1 = Conv_layer(dim_in,num*1, 7, 1, 3)
        self.encoder_2 = Conv_layer(num*1, num*1, 3, 1, 1)
        self.down_1 = CAS(num*1)
        self.encoder_3 = Conv_layer(num*1, num*2, 3, 1, 1)
        self.encoder_4 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.down_2 = CAS(num*2)
        self.encoder_5 = Conv_layer(num*2, num*4, 3, 1, 1)
        self.encoder_6 = Conv_layer(num*4, num*4, 3, 1, 1)
        self.down_3 = CAS(num*4)
        
        # Middle
        blocks = []
        for _ in range(4):
            block = res_bolock(num*4)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        
        # decoder
        self.up_1 = CAS(num*8, True)
        self.decoder_1 = Conv_layer(num*8, num*4, 3, 1, 1)
        self.decoder_2 = Conv_layer(num*4, num*2, 3, 1, 1)
        self.up_2 = CAS(num*4, True)
        self.decoder_3 = Conv_layer(num*4, num*2, 3, 1, 1)
        self.decoder_4 = Conv_layer(num*2, num*1, 3, 1, 1)
        self.up_3 = CAS(num*2, True)
        self.decoder_5 = Conv_layer(num*2, num*1, 3, 1, 1)
        self.decoder_6 = nn.Conv2d(num, dim_out, 3, 1, 1)

    def encoder(self, x):
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_2 = self.down_1(x_2)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        x_4 = self.down_2(x_4)
        x_5 = self.encoder_5(x_4)
        x_6 = self.encoder_6(x_5)
        x_6 = self.down_3(x_6)
        x_encoder = x_6
        return x_2, x_4, x_6
    
    def decoder(self, x, x_2, x_4, x_6):
        x_7 = self.up_1(torch.cat([x, x_6], 1))
        x_7 = self.decoder_1(x_7)
        x_8 = self.decoder_2(x_7)
        x_9 = self.up_2(torch.cat([x_8, x_4], 1))
        x_9 = self.decoder_3(x_9)
        x_10 = self.decoder_4(x_9)
        x_11 = self.up_3(torch.cat([x_10, x_2], 1))
        x_11 = self.decoder_5(x_11)
        x_12 = self.decoder_6(x_11)
        x_12 = (torch.tanh(x_12) + 1) / 2
        return x_12
        
    def forward(self, x):
        x_2, x_4, x_6 = self.encoder(x)
        x_middle =  self.middle(x_6)
        out = self.decoder(x_middle, x_2, x_4, x_6)
        return out

class u_net_heavy(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(u_net_heavy, self).__init__()
        
        # initial
        num = 32
        
        # encoder        
        self.encoder_1 = Conv_layer(dim_in,num*1, 7, 1, 3)
        self.encoder_2 = Conv_layer(num*1, num*1, 3, 1, 1)
        self.down_1 = CAS(num*1)
        self.encoder_3 = Conv_layer(num*1, num*2, 3, 1, 1)
        self.encoder_4 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.down_2 = CAS(num*2)
        self.encoder_5 = Conv_layer(num*2, num*4, 3, 1, 1)
        self.encoder_6 = Conv_layer(num*4, num*4, 3, 1, 1)
        self.down_3 = CAS(num*4)
        self.encoder_7 = Conv_layer(num*4, num*8, 3, 1, 1)
        self.encoder_8 = Conv_layer(num*8, num*8, 3, 1, 1)
        self.down_4 = CAS(num*8)
        self.encoder_9 = Conv_layer(num*8, num*16, 3, 1, 1)
        self.encoder_10 = Conv_layer(num*16, num*16, 3, 1, 1)
        self.down_5 = CAS(num*16)

        # Middle
        blocks = []
        for _ in range(4):
            block = res_bolock(num*16)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # sementic
        self.sementic_1 = Conv_layer(1+3,num*1, 7, 1, 3)
        self.sementic_2 = Conv_layer(num*1, num*1, 3, 1, 1)
        self.trans_1 = CAS(num*1)
        self.sementic_3 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.sementic_4 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.trans_2 = CAS(num*2)
        self.sementic_5 = Conv_layer(num*4, num*4, 3, 1, 1)
        self.sementic_6 = Conv_layer(num*4, num*4, 3, 1, 1)
        self.trans_3 = CAS(num*4)
        self.sementic_7 = Conv_layer(num*8, num*8, 3, 1, 1)
        self.sementic_8 = Conv_layer(num*8, num*8, 3, 1, 1)
        self.trans_4 = CAS(num*8)
        self.sementic_9 = Conv_layer(num*16, num*16, 3, 1, 1)
        self.sementic_10 = Conv_layer(num*16, num*16, 3, 1, 1)
        self.trans_5 = CAS(num*16)

        # decoder
        self.up_1 = CAS(num*32, True)
        self.decoder_1 = Conv_layer(num*32, num*32, 3, 1, 1)
        self.decoder_2 = Conv_layer(num*32, num*16, 3, 1, 1)
        self.up_2 = CAS(num*24, True)
        self.decoder_3 = Conv_layer(num*24, num*24, 3, 1, 1)
        self.decoder_4 = Conv_layer(num*24, num*8, 3, 1, 1)
        self.up_3 = CAS(num*12, True)
        self.decoder_5 = Conv_layer(num*12, num*12, 3, 1, 1)
        self.decoder_6 = Conv_layer(num*12, num*6, 3, 1, 1)
        self.up_4 = CAS(num*8, True)
        self.decoder_7 = Conv_layer(num*8, num*8, 3, 1, 1)
        self.decoder_8 = Conv_layer(num*8, num*2, 3, 1, 1)
        self.up_5 = CAS(num*3, True)
        self.decoder_9 = Conv_layer(num*3, num*1, 3, 1, 1)
        self.decoder_10 = nn.Conv2d(num, dim_out, 3, 1, 1)

    def encoder(self, x, y):
        x = torch.cat([x, y], 1)
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_2 = self.down_1(x_2)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        x_4 = self.down_2(x_4)
        x_5 = self.encoder_5(x_4)
        x_6 = self.encoder_6(x_5)
        x_6 = self.down_3(x_6)
        x_7 = self.encoder_7(x_6)
        x_8 = self.encoder_8(x_7)
        x_8 = self.down_4(x_8)
        x_9 = self.encoder_9(x_8)
        x_10 = self.encoder_10(x_9)
        x_10 = self.down_5(x_10)
        x_encoder = x_10
        return x_2, x_4, x_6, x_8, x_10

    def sementic(self, coarse, sementic, x_2, x_4, x_6, x_8):
        x1 = torch.cat([coarse, sementic], 1)
        s_1 = self.sementic_1(x1)
        s_2 = self.sementic_2(s_1)
        s_2 = self.trans_1(s_2)
        x2 = torch.cat([s_2, x_2], 1)
        s_3 = self.sementic_3(x2)
        s_4 = self.sementic_4(s_3)
        s_4 = self.trans_2(s_4)
        x3 = torch.cat([s_4, x_4], 1)
        s_5 = self.sementic_5(x3)
        s_6 = self.sementic_6(s_5)
        s_6 = self.trans_3(s_6)
        x4 = torch.cat([s_6, x_6], 1)
        s_7 = self.sementic_7(x4)
        s_8 = self.sementic_8(s_7)
        s_8 = self.trans_4(s_8)
        x5 = torch.cat([s_8, x_8], 1)
        s_9 = self.sementic_9(x5)
        s_10 = self.sementic_10(s_9)
        s_10 = self.trans_5(s_10)
        s_encoder = s_10
        return s_2, s_4, s_6, s_8, s_10

    def decoder(self, x, s_2, s_4, s_6, s_8, s_10):
        x_11 = self.up_1(torch.cat([x, s_10], 1))
        x_12 = self.decoder_1(x_11)
        x_12 = self.decoder_2(x_12)
        x_13 = self.up_2(torch.cat([x_12, s_8], 1))
        x_14 = self.decoder_3(x_13)
        x_14 = self.decoder_4(x_14)
        x_15 = self.up_3(torch.cat([x_14, s_6], 1))
        x_16 = self.decoder_5(x_15)
        x_16 = self.decoder_6(x_16)
        x_17 = self.up_4(torch.cat([x_16, s_4], 1))
        x_18 = self.decoder_7(x_17)
        x_18 = self.decoder_8(x_18)
        x_19 = self.up_5(torch.cat([x_18, s_2], 1))
        x_20 = self.decoder_9(x_19)
        x_20 = self.decoder_10(x_20)
        x_20 = (torch.tanh(x_20) + 1) / 2
        return x_20
        
    def forward(self, rain, coarse, sementic):
        x_2, x_4, x_6, x_8, x_10 = self.encoder(rain, coarse)
        s_2, s_4, s_6, s_8, s_10 = self.sementic(coarse, sementic, x_2, x_4, x_6, x_8)
        x_middle =  self.middle(x_10)
        out = self.decoder(x_middle, s_2, s_4, s_6, s_8, s_10)
        return out

class fr_net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(fr_net, self).__init__()
        
        num = 32

        # encoder        
        self.encoder_1 = Conv_layer(dim_in,num*1, 7, 1, 3)
        self.encoder_2 = Conv_layer(num*1, num*2, 3, 1, 1)
        self.encoder_3 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.encoder_4 = Conv_layer(num*2, num*4, 3, 1, 1)
        
        # Middle
        blocks = []
        for _ in range(4):
            block = Conv_layer(num*4, num*4, 3, 1, 1)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        
        # decoder
        self.decoder_1 = Conv_layer(num*8, num*2, 3, 1, 1)
        self.decoder_2 = Conv_layer(num*2, num*2, 3, 1, 1)
        self.decoder_3 = Conv_layer(num*4, num*1, 3, 1, 1)
        self.decoder_4 = nn.Conv2d(num, dim_out, 3, 1, 1)

    def encoder(self, x):
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        return x_2, x_4
    
    def decoder(self, x, x_2, x_4):
        x_5 = self.decoder_1(torch.cat([x, x_4], 1))
        x_6 = self.decoder_2(x_5)
        x_7 = self.decoder_3(torch.cat([x_6, x_2], 1))
        x_8 = self.decoder_4(x_7)
        x_8 = (torch.tanh(x_8) + 1) / 2
        return x_8

    def forward(self, x):
        x_2, x_4 = self.encoder(x)
        x_middle =  self.middle(x_4)
        out = self.decoder(x_middle, x_2, x_4)
        return out

class Conv_layer(nn.Module):
    
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, up=False):
        super(Conv_layer,self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_dim)
        self.activation = nn.GELU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = up
        
    def forward(self,x):
        if self.up == True:
            x = self.upsample(x)
        x = self.activation(self.norm(self.conv(x)))
        return x

class res_bolock(nn.Module):
    
    def __init__(self, channels):
        super(res_bolock, self).__init__()
        self.layer_1 = Conv_layer(channels, channels, 3, 1, 1)
        self.layer_2 = Conv_layer(channels, channels, 3, 1, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class CAS(nn.Module):
    # context-aware scaling
    def __init__(self, channels, is_up=False):
        super(CAS, self).__init__()
        
        # context extraction
        self.context = CA(channels)
        
        # downsampling
        self.down = nn.Conv2d(channels, channels, 3, 2, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.is_up = is_up
        
    def forward(self, x):
        con_attention = self.context(x)
        if self.is_up == True:
            x = self.up(x)
        else:
            x = self.down(x)
        return x * con_attention.expand_as(x)
    
    
class CA(nn.Module):
    # context attention
    def __init__(self, channels, reduction=16):
        super(CA, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return y