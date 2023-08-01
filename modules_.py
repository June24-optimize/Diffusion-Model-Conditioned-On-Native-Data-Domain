#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:50:45 2023

@author: wb885
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import mriForwardOp, mriAdjointOp

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x): #[12, m, 64, 64]
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) #[48, 1024, m]
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x 
        attention_value = self.ff_self(attention_value) + attention_value #[48, 1024, m]
        
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, device='cuda'):
        super().__init__()
        self.img_size = 320
        self.init_ch = 64
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, self.init_ch ) #64
        self.down1 = Down(self.init_ch , self.init_ch * 2) #64, 128
        self.sa1 = SelfAttention(self.init_ch * 2, self.img_size//2) #128, 64  self.img_size//2 4 #inchannel, size
        self.down2 = Down(self.init_ch * 2, self.init_ch * 4)  #128, 256 self.init_ch * 4
        self.sa2 = SelfAttention(self.init_ch * 4, self.img_size//4) #256, 32  self.img_size//4 16
        self.down3 = Down(self.init_ch * 4, self.init_ch * 4) #256, 256
        self.sa3 = SelfAttention(self.init_ch * 4, self.img_size//8) #256, 16  self.img_size//8 64

        self.bot1 = DoubleConv(self.init_ch * 4, self.init_ch * 8) #256, 512
        self.bot2 = DoubleConv(self.init_ch * 8, self.init_ch * 8) #512, 512
        self.bot3 = DoubleConv(self.init_ch * 8, self.init_ch * 4) #512, 256

        self.up1 = Up(self.init_ch * 8, self.init_ch * 2)  #512, 128
        self.sa4 = SelfAttention(self.init_ch * 2, self.img_size//4) #128, 32  self.img_size//4 16
        self.up2 = Up(self.init_ch * 4, self.init_ch)    #256, 64
        self.sa5 = SelfAttention(self.init_ch, self.img_size//2)  #64, 64  self.img_size//2 4
        self.up3 = Up(self.init_ch * 2, self.init_ch)   #128, 64
        self.sa6 = SelfAttention(self.init_ch, self.img_size) #64, 128  self.img_size
        self.outc = nn.Conv2d(self.init_ch, c_out, kernel_size=1) #64

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)#[1, 256]
        
        x1 = self.inc(x)  #[2, 64, 128, 128]        [2, 64, 320, 320]  320
        x2 = self.down1(x1, t) #[2, 128, 64, 64]    [2, 128, 160, 160] 80 
        x2 = self.sa1(x2) #[2, 128, 64, 64]         [2, 128, 160, 160] 80
        x3 = self.down2(x2, t) #[2, 256, 32, 32]    [2, 256, 80, 80]   20
        x3 = self.sa2(x3) #[2, 256, 32, 32]         [2, 256, 80, 80]   20
        x4 = self.down3(x3, t) #[2, 256, 16, 16]    [2, 256, 40, 40]   5
        x4 = self.sa3(x4) #[2, 256, 16, 16]         [2, 256, 40, 40]   5

        x4 = self.bot1(x4) #[2, 512, 16, 16]        [2, 512, 40, 40]   5
        x4 = self.bot2(x4) #[2, 512, 16, 16]       [2, 512, 40, 40]    5
        x4 = self.bot3(x4) #[2, 256, 16, 16]        [2, 256, 40, 40]   5

        x = self.up1(x4, x3, t) #[2, 128, 32, 32]   [2, 128, 80, 80]  20
        x = self.sa4(x) #[2, 128, 32, 32]           [2, 128, 80, 80]  20
        x = self.up2(x, x2, t) #[2, 64, 64, 64]     [2, 64, 160, 160] 80
        x = self.sa5(x) #[2, 64, 64, 64]            [2, 64, 160, 160] 80
        x = self.up3(x, x1, t) #[2, 64, 128, 128]   [2, 64, 320, 320] 320
        #x = self.sa6(x) #[2, 64, 128, 128]          [2, 64, 320, 320] 320
        output = self.outc(x) #[2, 2, 128, 128]     [2, 2, 320, 320]  320
        return output

  
class ReconNet_BasicBlock_nolearn(nn.Module):
    def __init__(self, chk=32, device="cuda"):
        super(ReconNet_BasicBlock_nolearn, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.001]))
        
    def forward(self, x, patial_kspace, mask):
        #patial_kspace.shape [2, 128, 128]
        ATAx = mriAdjointOp(mriForwardOp(x, mask), mask)# FTPT(PFx)
        ATf  = mriAdjointOp(patial_kspace, mask)#FTPT(f)
        
        x = x - self.lambda_step * ATAx
        x = x + self.lambda_step * ATf #[2, 128, 128]
        update_img = x
        
        return update_img

class ReconNet_nolearn(torch.nn.Module):
    def __init__(self, PhaseNo, device="cuda"):
        super(ReconNet_nolearn, self).__init__()
        
        
        onephase = []
        self.PhaseNo = PhaseNo
        for i in range(PhaseNo):
            onephase.append(ReconNet_BasicBlock_nolearn())
        self.fcs = nn.ModuleList(onephase)
        
    def forward(self, noised_t, partial_kspace, mask):
        
        x = torch.fft.ifft2(noised_t) 
        
        for i in range(self.PhaseNo):            
            x = self.fcs[i](x, partial_kspace, mask)

        x_kspace = torch.fft.fft2(x) #[2, 128, 128]
        return x_kspace
    
    
class ReconNet_BasicBlock(nn.Module):
    def __init__(self, chk=128, device='cuda'):
        super(ReconNet_BasicBlock, self).__init__()
        self.W0 = nn.Conv2d(2, chk, kernel_size=3, padding=1, bias=False)
        self.W1 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W3 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W4 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W5 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        
        self.W6 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W7 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W8 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W9 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W10 = nn.Conv2d(chk, chk, kernel_size=3, padding=1, bias=False)
        self.W11 = nn.Conv2d(chk, 2, kernel_size=3, padding=1, bias=False)
        self.lambda_step = nn.Parameter(torch.Tensor([0.01]))
        
    def forward(self, x, patial_kspace, mask):
        #patial_kspace.shape [2, 1, 128, 128]
        ATAx = mriAdjointOp(mriForwardOp(x, mask), mask)# FTPT(PFx)
        ATf  = mriAdjointOp(patial_kspace, mask)#FTPT(f)
        
        x = x - self.lambda_step * ATAx
        x = x + self.lambda_step * ATf #[2, 1, 128, 128]
        
        img = torch.cat((x.real, x.imag), dim=1)
        
        img0 = F.relu(self.W0(img))
        img1 = F.relu(self.W1(img0))
        img2 = F.relu(self.W2(img1))
        img3 = F.relu(self.W3(img2))
        img4 = F.relu(self.W4(img3))
        img5 = F.relu(self.W5(img4))
        img6 = F.relu(self.W6(img5))
        img7 = F.relu(self.W7(img6))
        img8 = F.relu(self.W8(img7))
        img9 = F.relu(self.W9(img8))
        img10 = F.relu(self.W10(img9))
        img11 = self.W11(img10)
        
        update_img = torch.unsqueeze(torch.complex(img11[:,0,:,:],img11[:,1,:,:]), 1) + x #[2, 1, 128, 128]
        
        return update_img

class ReconNet(torch.nn.Module):
    def __init__(self, PhaseNo, device='cuda'):
        super(ReconNet, self).__init__()
        ch = 32
        
        onephase = []
        self.PhaseNo = PhaseNo
        for i in range(PhaseNo):
            onephase.append(ReconNet_BasicBlock())
        self.fcs = nn.ModuleList(onephase)
        self.w1 = nn.Conv2d(2, ch, kernel_size=3, padding=1, bias=False)
        self.w2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w4 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w5 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w6 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        
        self.w7 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w8 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w9 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w10 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w11 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.w12 = nn.Conv2d(ch, 2, kernel_size=3, padding=1, bias=False)
        
    def forward(self, sampled_kspace, scanned_data, mask):
        
        initial = torch.fft.ifft2(sampled_kspace)
        initial_real, initial_imag = torch.unsqueeze(initial.real, 1), torch.unsqueeze(initial.imag, 1)
        initial = torch.cat((initial_real, initial_imag), dim=1) #[2, 2, 128, 128]
        print(initial.shape)
        initial = F.relu(self.w1(initial))
        initial = F.relu(self.w2(initial))
        initial = F.relu(self.w3(initial))
        initial = F.relu(self.w4(initial))
        initial = F.relu(self.w5(initial))
        initial = F.relu(self.w6(initial))
        
        initial = F.relu(self.w7(initial))
        initial = F.relu(self.w8(initial))
        initial = F.relu(self.w9(initial))
        initial = F.relu(self.w10(initial))
        initial = F.relu(self.w11(initial))
        x = self.w12(initial)
        x = torch.unsqueeze(torch.complex(x[:,0,:,:],x[:,1,:,:]), 1) #[2, 1, 128, 128]
        
        for i in range(self.PhaseNo):            
            x = self.fcs[i](x, scanned_data, mask)

        x_kspace = torch.fft.fft2(x) #[2, 1, 128, 128]
        recon = x
        
        return recon, x_kspace
    
        # self.device = device
        # self.time_dim = time_dim
        # self.inc = DoubleConv(c_in, 16) #64
        # self.down1 = Down(16, 32) #64, 128
        # self.sa1 = SelfAttention(128, 64) #128, 64
        # self.down2 = Down(32, 64)  #128, 256
        # self.sa2 = SelfAttention(256, 32) #256, 32
        # self.down3 = Down(64, 64) #256, 256
        # self.sa3 = SelfAttention(256, 16) #256, 16

        # self.bot1 = DoubleConv(64, 128) #256, 512
        # self.bot2 = DoubleConv(128, 128) #512, 512
        # self.bot3 = DoubleConv(128, 64) #512, 256

        # self.up1 = Up(128, 32)  #512, 128
        # self.sa4 = SelfAttention(128, 32) #128, 32
        # self.up2 = Up(64, 16)    #256, 64
        # self.sa5 = SelfAttention(64, 64)  #64, 64
        # self.up3 = Up(32, 16)   #128, 64
        # self.sa6 = SelfAttention(64, 128) #64, 128
        # self.outc = nn.Conv2d(16, c_out, kernel_size=1) #64
        
# class UNet_conditional(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8)

#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 16)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 32)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

#         if num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_dim)

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t, y):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)

#         if y is not None:
#             t += self.label_emb(y)

#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)

#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)

#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output


if __name__ == '__main__':
    net = UNet(device="cpu")
    #net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
