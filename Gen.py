#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:47:40 2023

@author: Wanyu Bian
"""
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
from ddpm_utils import *
import Utils
from modules import UNet, ReconNet_nolearn
import logging
import cv2
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start= 0.00001, beta_end=0.001, img_size=320, device=device):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_kspace(self, x, scanned_data, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        #print(sqrt_one_minus_alpha_hat)
        E_real, E_imag = torch.randn_like(x[:,0,:,:]), torch.randn_like(x[:,1,:,:])
        E = torch.cat((torch.unsqueeze(E_real, dim=1), torch.unsqueeze(E_imag, dim=1)), dim = 1)
        noised_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E 
        # lambda_ = 1 #torch.exp(torch.tensor(-(i-1) / (self.noise_steps/5)))
        # noised_dc = mask * (lambda_ * scanned_data + (1-lambda_)* noised_x) + (one - mask)*noised_x
        E_dc = (one - mask)*E
        
        return noised_x, E_dc

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, mask, scanned_data, n):
        logging.info(f"Sampling {n} new images....")
        
        model.eval()
        with torch.no_grad():
            x_real = torch.rand(n, 1, self.img_size, self.img_size).to(self.device) 
            x_imag = torch.rand(n, 1, self.img_size, self.img_size).to(self.device) 
            
            x = torch.cat((x_real, x_imag), dim = 1) #[4, 2, 320, 320]
            # lambda_ = 1 #torch.exp(torch.tensor(-(i-1) / (self.noise_steps/5)))
            # x = mask * (lambda_ * scanned_data + (1-lambda_)* x) + (one - mask)*x
            # x_kspace = torch.complex(x[:,0,:,:],x[:,1,:,:])
            
            # kspace_out = np.squeeze(x_kspace[-1].cpu().detach().numpy())
            # imgs = np.abs(np.fft.ifft2(kspace_out))  #[4, 320, 320]
            # v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)
            # img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max)
            # cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/sample_initial.png" , img_enhanced) 
                    
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t) #[4, 2, 320, 320]
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise_real, noise_imag = torch.randn_like(x[:,0,:,:]), torch.randn_like(x[:,1,:,:])
                    noise = torch.cat((torch.unsqueeze(noise_real, dim=1),torch.unsqueeze(noise_imag, dim=1)), dim = 1)
                    
                else:
                    noise = torch.zeros_like(x)
                                      
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise                
    
                #lambda_ = 1-(i-1)/self.noise_steps
                
                lambda_ = 1 #torch.exp(torch.tensor(-(i-1) / (self.noise_steps/10)))
                x = mask * (lambda_ * scanned_data + (1-lambda_)* x) + (one - mask)*x
                kspace = torch.complex(x[:,0,:,:],x[:,1,:,:])
                kspace = torch.unsqueeze(kspace, dim = 1) #[2, 1, 320, 320]
                
                if i % 100 == 0 or i == 1 or i == self.noise_steps-1:    
                    kspace = np.squeeze(kspace.cpu().detach().numpy())
                    imgs = np.abs(np.fft.ifft2(kspace))  #[2, 1, 320, 320]
                    v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)#[-1]
                    img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max) #[-1] 320, 320
                    cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/sample_%d.png" % (i), img_enhanced)    
                    
        model.train()
        return kspace

training_data = sio.loadmat('./mat_data/kspace_corpd/kspace_320.mat')['kspace'].astype(np.complex64) #319 320 320
# training_data1 = sio.loadmat('./mat_data/kspace_corpd/corpd_kspace1.mat')['corpd_kspace'].astype(np.complex64) #319 320 320
# training_data2 = sio.loadmat('./mat_data/kspace_corpd/corpd_kspace2.mat')['corpd_kspace'].astype(np.complex64) #317 320 320
# training_data = np.vstack((training_data1, training_data2)) #636 320 320

mask = sio.loadmat('./mat_data/mask/cartesian_r.mat')['cartesian_r'].astype(np.float32) #320 320
mask = torch.Tensor(mask).float().to(device)

total, m, n = training_data.shape[-3], training_data.shape[-2], training_data.shape[-1]
full_kspace = training_data.reshape(total,-1,m,n)#(34, 1, 320, 320)
one = torch.ones((320, 320)).to(device)

class RandomDataset(Dataset):
    def __init__(self, kspace, length):
        self.k_real = kspace.real
        self.k_imag = kspace.imag
        self.len = length

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        k_real = torch.Tensor(self.k_real[index, :]).float()
        k_imag = torch.Tensor(self.k_imag[index, :]).float()
        
        return k_real, k_imag
    
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = DataLoader(dataset=RandomDataset(full_kspace, 32), batch_size=args.batch_size, num_workers=32, shuffle=False)#get_data(args) #
    model = UNet().to(device)  #len(full_kspace)
    refine_model = ReconNet_nolearn(PhaseNo = 10).to(device) 
    parameters = list(model.parameters()) + list(refine_model.parameters()) #model.parameters() 
    optimizer = optim.AdamW(parameters, lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    result_file_name = "./models/DDPM_Uncondtional/largedata5/train_320/MSE_loss.txt" 
    if args.start_epoch > 0:
        model.load_state_dict(torch.load('./models/DDPM_Uncondtional/largedata5/train_320/model_%d.pt' % (args.start_epoch)))
        refine_model.load_state_dict(torch.load('./models/DDPM_Uncondtional/largedata5/train_320/refine_model_%d.pt' % (args.start_epoch))) #

    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        
        recon_cat =  []
        for i, (k_real, k_imag) in enumerate(pbar):
            
            target_kspace = torch.Tensor(torch.complex(k_real, k_imag).to(device))
            
            k_space = np.squeeze(target_kspace.cpu().detach().numpy())
            imgs = np.abs(np.fft.ifft2(k_space))#(4, 320, 320)
            #imgs_trans = imgs.transpose(1,2,0)
            # im_rec_name = "./data/loadback.mat" 
            # Utils.saveAsMat(imgs, im_rec_name, 'loadback',  mat_dict=None)
            # v_min, v_max = Utils.getContrastStretchingLimits(imgs[-1], saturated_pixel=0.002)
            # img_enhanced = Utils.normalize(imgs[-1], v_min=v_min, v_max=v_max)
            # cv2.imwrite("./data/loadback.png", img_enhanced)
            
            target_kspace_real = torch.Tensor(target_kspace.real).to(device)
            target_kspace_imag = torch.Tensor(target_kspace.imag).to(device) #[2, 1, 320, 320]
            target_kspace_cat = torch.cat((target_kspace_real, target_kspace_imag), dim=1) #[2, 2, 320, 320]
            
            scanned_data = target_kspace_cat * mask  #2, 2, 320, 320
            
            t = diffusion.sample_timesteps(target_kspace_cat.shape[0]).to(device)
            
            kspace_dc, noise = diffusion.noise_kspace(target_kspace_cat, scanned_data, t) #x_t is kspace print(x_t.shape) [2, 2, 320, 320]
               
            kspace = torch.complex(kspace_dc[:,0,:,:],kspace_dc[:,1,:,:])#[4, 320, 320]
            
            # kspace_out = np.squeeze(kspace[-1].cpu().detach().numpy())
            # imgs = np.abs(np.fft.ifft2(kspace_out))  #[4, 320, 320]
            # v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)
            # img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max)
            # cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/data_consist.png" , img_enhanced) 
                    
            #input to gd
            partial_kspace = torch.complex(scanned_data[:,0,:,:],scanned_data[:,1,:,:])#[2, 320, 320]
        
            x_t_gd = refine_model(kspace, partial_kspace, mask)
            x_t =  torch.cat((torch.unsqueeze(x_t_gd.real, dim=1), torch.unsqueeze(x_t_gd.imag, dim=1)), dim = 1) #[2, 2, 320, 320])

            predicted_noise = model(x_t, t) #[1, 2, 320, 320])
            loss =  mse(noise, predicted_noise) #+ mse(target_r, recon_r) + mse(target_i, recon_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            
            if epoch > 1 and epoch % 100 == 0:#
                sampled_kspace = diffusion.sample(model, mask, scanned_data, n=target_kspace.shape[0])
                
                sampled_kspace_r = torch.Tensor(sampled_kspace.real).to(device)
                sampled_kspace_i = torch.Tensor(sampled_kspace.imag).to(device)
                sampled_kspace = torch.complex(sampled_kspace_r, sampled_kspace_i) #[2, 320, 320]
                
                #partial_kspace = torch.unsqueeze( torch.complex(scanned_data[:,0,:,:],scanned_data[:,1,:,:]), dim = 1)
                
                #recon, recon_kspace = refine_model(sampled_kspace, partial_kspace, mask)
                # target_r = torch.squeeze(target_kspace_real)
                # target_i = torch.squeeze(target_kspace_imag)
                # recon_r = torch.squeeze(recon_kspace.real)
                # recon_i = torch.squeeze(recon_kspace.imag)
                
                sampled_img = torch.fft.ifft2(sampled_kspace)
                recon_img = np.abs(np.squeeze(sampled_img.cpu().detach().numpy()))#[-1]
                v_min, v_max = Utils.getContrastStretchingLimits(recon_img, saturated_pixel=0.002)
                img_enhanced = Utils.normalize(recon_img, v_min=v_min, v_max=v_max)
                cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/recon/recon_%d.png" % (epoch), img_enhanced)
                
                recon_cat.append(np.abs(np.squeeze(sampled_img.cpu().detach().numpy())))
            
                #save recon as mat
                recon_output = np.concatenate( (recon_cat),axis=0)
                im_rec_name = "./results/DDPM_Uncondtional/largedata5/train_320/recon/recon_%d.mat" % (epoch)
                Utils.saveAsMat(recon_output, im_rec_name, 'recon_%d' % (epoch),  mat_dict=None) 
                
                x_t_kspace = torch.complex(x_t[-1][0], x_t[-1][1])
                x_t_kspace = np.squeeze(x_t_kspace.cpu().detach().numpy())
                
                imgs = np.abs(np.fft.ifft2(x_t_kspace))             
                v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)
                img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max)
                cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/noise/x_t_%d.png" % (epoch), img_enhanced)
            
                imgs = np.squeeze(noise[-1][0].cpu().detach().numpy())                                              
                v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)
                img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max)
                cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/noise/noise_%d.png" % (epoch), img_enhanced)
                
                imgs = np.squeeze(predicted_noise[-1][0].cpu().detach().numpy())                                              
                v_min, v_max = Utils.getContrastStretchingLimits(imgs, saturated_pixel=0.002)
                img_enhanced = Utils.normalize(imgs, v_min=v_min, v_max=v_max)
                cv2.imwrite("./results/DDPM_Uncondtional/largedata5/train_320/noise/predicted_noise_%d.png" % (epoch), img_enhanced)
    
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"model_%d.pt") % (epoch))
                #torch.save(refine_model.state_dict(), os.path.join("models", args.run_name, f"refine_model_%d.pt") % (epoch))
            
        output = "%.4f \n" % (loss.item())
        output_file = open(result_file_name, 'a')
        output_file.write(output)
        output_file.close()  

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional/largedata5/train_320/"
    args.start_epoch = 0
    args.epochs = 70000
    args.batch_size = 1
    args.image_size = n
    #args.dataset_path = r"/local_mount/space/cookie1/1/users/wb885/diffusion_model/data/"
    args.device = device
    args.lr = 0.0001
    train(args)


if __name__ == '__main__':
    launch()
    
    model = UNet().to(device)
    ckpt = torch.load("./models/DDPM_Uncondtional/largedata5/train_320/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=n, device=device)
    x = diffusion.sample(model, mask, scanned_data, 1)
    
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
