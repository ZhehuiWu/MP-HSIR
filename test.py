import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import torch.nn as nn 
from net.MP_HSIR import MP_HSIR_Net

from utils.dataset_utils import Gaussian_Denoise_Dataset, Gaussian_Denoise_inid_Dataset, Stripe_Denoise_Dataset, Deadline_Denoise_Dataset, Impulse_Denoise_Dataset, Gaussian_Deblur_Dataset, Motion_Deblur_Dataset, Super_Resolution_Dataset, Inpaint_Dataset, Dehaze_Dataset, Bandmis_Dataset, Poisson_Denoise_Dataset, Real_Degrad_Dataset
from utils.val_utils import AverageMeter, compute_psnr_ssim,compute_psnr_ssim2
from utils.image_io import save_image_tensor
from utils.schedulers import LinearWarmupCosineAnnealingLR

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import random
from utils.image_utils import *
import scipy.io as sio
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')  
print(torch.cuda.is_available()) 
print(torch.cuda.device_count())  



class PromptIRModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.rank = self.args.rank
        self.net = MP_HSIR_Net(in_channel=31,out_channel=31,dim=64,task_classes=6)
        # self.net = MP_HSIR_Net(in_channel=100,out_channel=100,dim=96,task_classes=7)        
      
        self.loss_fn = nn.L1Loss()

    def forward(self, x1,x2):
        return self.net(x1,x2)

    
    def training_step(self, batch, batch_idx):

        [clean_name, de_id, sr], degrad_A, clean_A, prompt = batch
        A_restored = torch.clamp(self.net(degrad_A,prompt), min=0, max=1)
        loss = self.loss_fn(A_restored, clean_A)

        self.log("train_loss", loss,on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=1000,
            max_epochs=10000
        )

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss', 
                'interval': 'epoch',
                'frequency': 1
            }
        }



def test_real_denoise(net, dataset, degrad_id = None, select_bands = None, use_conv3d = False, device = None):
    output_path = testopt.output_path + 'urban/ours/' 
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)
     
            restored = net(degrad_patch,prompt)
            clean_patch = torch.clamp(clean_patch, 0, 1)
 
            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'real_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Degrad_Id=%d: psnr: %.2f, ssim: %.4f" % (degrad_id, psnr.avg, ssim.avg))

def test_poisson_denoise(net, dataset, degrad_id = None, select_bands = None, use_conv3d = False, device = None):
    output_path = testopt.output_path + 'poisson/' + str(degrad_id) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)
            restored = net(degrad_patch,prompt)
            clean_patch = torch.clamp(clean_patch, 0, 1)
 
            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)
           
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'poisson_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Degrad_Id=%d: psnr: %.2f, ssim: %.4f" % (degrad_id, psnr.avg, ssim.avg))

def test_gaussian_denoise(net, dataset, rank =20, sigma=30, select_bands = None, use_conv3d = False, device = None):
    output_path = testopt.output_path + 'gaussian_denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)

            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'gaussian_noisy_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))

def test_gaussian_inid_denoise(net, dataset, rank =20, sigma=30, select_bands = None, use_conv3d = False, device = None):
    output_path = testopt.output_path + 'gaussian_inid_denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)

            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)
 
            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'gaussian_noisy_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Denoise sigma=%s: psnr: %.2f, ssim: %.4f" % (str(sigma), psnr.avg, ssim.avg))

def test_stripe_denoise(net, dataset, rank =20, stripe_ratio=0.1, select_bands = None, use_conv3d = False):
    output_path = testopt.output_path + 'destripe/' + str(stripe_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'striped_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Destripe stripe ratio=%s: psnr: %.2f, ssim: %.4f" % (str(stripe_ratio), psnr.avg, ssim.avg))

def test_deadline_denoise(net, dataset, rank=20, deadline_ratio=0.1, select_bands = None, use_conv3d =False):
    output_path = testopt.output_path + 'deadline_denoise/' + str(deadline_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'deadlined_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Deadline denoise deadline ratio=%s: psnr: %.2f, ssim: %.4f" % (str(deadline_ratio), psnr.avg, ssim.avg))

def test_impulse_denoise(net, dataset, rank=20, impulse_ratio=0.1, select_bands = None, use_conv3d=False):
    output_path = testopt.output_path + 'impulse_denoise/' + str(impulse_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([1]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'impulse_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_instructir' + clean_name[0] + '.png')

        print("Impulse denoise impulse ratio=%s: psnr: %.2f, ssim: %.4f" % (str(impulse_ratio), psnr.avg, ssim.avg))

def test_gaussian_deblur(net, dataset, rank=20, sigma=1.6, select_bands = None, use_conv3d=False):
    output_path = testopt.output_path + 'gaussian_deblur/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([2]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'gaussian_blurred_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_instructir' + clean_name[0] + '.png')

        print("Gaussian deblur sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_motion_deblur(net, dataset, rank=20, motion_radius=5, select_bands = None, use_conv3d=False):
    output_path = testopt.output_path + 'motion_deblur/' + str(motion_radius) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([0]).to(device)

            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)


            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)

            if use_conv3d:
                restored = restored.squeeze(1)
                degrad_patch = degrad_patch.squeeze(1)
                clean_patch = clean_patch.squeeze(1)


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'motion_blurred_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')

        print("Motion deblur motion radius=%s: psnr: %.2f, ssim: %.4f" % (motion_radius, psnr.avg, ssim.avg))


def test_super_resolution(net, dataset, rank=20, downsample_factor=4, select_bands = None,use_conv3d=False):
    output_path = testopt.output_path + 'super_resolution/' + str(downsample_factor) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([3]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'downsampled_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')
        print("Super resolution downsample factor=%d: psnr: %.2f, ssim: %.4f" % (downsample_factor, psnr.avg, ssim.avg))


def test_inpaint(net, dataset, rank=20, mask_ratio=0.1, select_bands = None,use_conv3d=False):
    output_path = testopt.output_path + 'inpaint/' + str(mask_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch, mask) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]

            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([4]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            
            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'masked_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')
        print("Inpaint mask ratio=%f: psnr: %.2f, ssim: %.4f" % (mask_ratio, psnr.avg, ssim.avg))           

def test_dehaze(net, dataset, rank=80, haze_omega=0.5, select_bands = None,use_conv3d=False):
    output_path = testopt.output_path + 'dehaze/real' + str(haze_omega) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([5]).to(device)
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)
            
            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'hazed_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_hair' + clean_name[0] + '.png')
        print("Dehaze haze omega=%d: psnr: %.2f, ssim: %.4f" % (haze_omega, psnr.avg, ssim.avg))

def test_bandmis(net, dataset, bandmis_ratio=0.3, select_bands = None,use_conv3d = False):
    output_path = testopt.output_path + 'bandmis/' + str(bandmis_ratio) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            C, H, W = clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1]
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            prompt = torch.tensor([5]).to(device)#natural_scene
            # prompt = torch.tensor([6]).to(device)#remote_sensing
            
            if use_conv3d:
                degrad_patch = degrad_patch.unsqueeze(1)
                clean_patch = clean_patch.unsqueeze(1)

            restored = net(degrad_patch,prompt)

            clean_patch = torch.clamp(clean_patch, 0, 1)
            temp_psnr, temp_ssim, N = compute_psnr_ssim2(restored, clean_patch, degrad_patch)#Only record the PSNR and SSIM of the complementary band
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(clean_patch[:,select_bands,:,:], output_path + 'origin_' + clean_name[0] + '.png')
            save_image_tensor(degrad_patch[:,select_bands,:,:], output_path + 'bandmis_' + clean_name[0] + '.png')
            save_image_tensor(restored[:,select_bands,:,:], output_path + 'restored_' + clean_name[0] + '.png')
        print("Bandmiss ratio=%f: psnr: %.2f, ssim: %.4f" % (bandmis_ratio, psnr.avg, ssim.avg))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mode', type=int, default=0,
                        help='Used to select degradation mode.')

    parser.add_argument('--test_dir', type=str, default='/home/wuzhehui/Hyper_Restoration/PromptIR-main/data/Test/ARAD',
                        help='where clean HSIs of test saves.')   
    parser.add_argument('--test_degrad_dir', type=str, default='',
                        help='where real degraded HSIs of test saves.') 
    parser.add_argument('--degrad_id', type=int, default=1, help='')
    parser.add_argument('--degrad_range', type=list, default=[(15,),(30, 30),(0.05, 0.15)], help='')
    parser.add_argument('--gaussian_noise_sigma', type=int, default=70, help='Gaussian Noise intensity')
    parser.add_argument('--gaussian_noise_sigmas', type=list, default=[10,30,50,70], help='Gaussian Noise inid intensity')
    parser.add_argument('--stripe_nosie_ratio', type=str, default=[0.05,0.15], help='Stripe ratio')
    parser.add_argument('--deadline_nosie_ratio', type=str, default=[0.05,0.15], help='Deadline ratio')
    parser.add_argument('--impulse_nosie_ratio', type=str, default=[0.1, 0.3, 0.5, 0.7], help='Impulse ratio')
    parser.add_argument('--gaussian_blur_radius', type=str, default=15, help='Gaussian Blur')
    parser.add_argument('--motion_blur_radius', type=tuple, default=(15,45), help='Motion Blur')
    parser.add_argument('--downsample_factor', type=int, default=8, help='factor')
    parser.add_argument('--mask_ratio', type=str, default=0.9, help='Inpaint Mask Ratio')
    parser.add_argument('--haze_omega', type=str, default=1, help='haze')
    parser.add_argument('--bandmis_ratio', type=str, default=0.3, help='Bandmis Ratio')
    parser.add_argument('--select_bands', type=list, default=[27,15,9], help='The bands used to compose the pseudo-color image')
    parser.add_argument('--output_path', type=str, default="/home/wuzhehui/Hyper_Restoration/PromptIR-main/output_supple/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="/home/wuzhehui/Hyper_Restoration/MP-HSIR-main/ckpt/Natural_scene.ckpt", help='checkpoint save path')
    parser.add_argument('--rank', type=int, default=31, help='When the number of bands is less than 100, it is 20; when it is greater than or equal to 100, it is 80.')
    testopt = parser.parse_args()

    set_seed(testopt.seed)
    torch.cuda.set_device(testopt.cuda)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    ckpt_path = testopt.ckpt_path
    net  = PromptIRModel(testopt).load_from_checkpoint(ckpt_path, args=testopt, map_location=device, strict=False).to(device)
    net.eval()
   
    print("CKPT name : {}".format(ckpt_path))

    if testopt.mode == 0:
        gaussian_denoise_testset = Gaussian_Denoise_Dataset(testopt) 
        print('Start gaussian denoise testing sigma={}'.format(testopt.gaussian_noise_sigma))
        test_gaussian_denoise(net, gaussian_denoise_testset, rank=testopt.rank, sigma=testopt.gaussian_noise_sigma, select_bands = testopt.select_bands, device = device)
    
    if testopt.mode == 1:
        gaussian_denoise_testset = Gaussian_Denoise_inid_Dataset(testopt) 
        print('Start inid gaussian denoise testing sigma={}'.format(testopt.gaussian_noise_sigmas))
        test_gaussian_inid_denoise(net, gaussian_denoise_testset, rank=testopt.rank, sigma=testopt.gaussian_noise_sigmas, select_bands = testopt.select_bands, device = device)

    elif testopt.mode == 2:
        print('Start destripe testing stripe ratio={}'.format(testopt.stripe_nosie_ratio))
        destripe_denoise_testset = Stripe_Denoise_Dataset(testopt) 
        test_stripe_denoise(net, destripe_denoise_testset, rank=testopt.rank, stripe_ratio=testopt.stripe_nosie_ratio, select_bands = testopt.select_bands)    

    elif testopt.mode == 3:
        print('Start deadline denoise testing deadline ratio={}'.format(testopt.deadline_nosie_ratio))
        deadline_denoise_testset = Deadline_Denoise_Dataset(testopt) 
        test_deadline_denoise(net, deadline_denoise_testset, rank=testopt.rank, deadline_ratio=testopt.deadline_nosie_ratio, select_bands = testopt.select_bands)    

    elif testopt.mode == 4:
        print('Start impulse denoise testing impulse ratio={}'.format(testopt.impulse_nosie_ratio))
        impulse_denoise_testset = Impulse_Denoise_Dataset(testopt) 
        test_impulse_denoise(net, impulse_denoise_testset, rank=testopt.rank, impulse_ratio=testopt.impulse_nosie_ratio, select_bands = testopt.select_bands)    
    
    elif testopt.mode == 5:
        print('Start gaussian deblur testing sigma={}'.format(testopt.gaussian_blur_radius))
        gaussian_deblur_testset = Gaussian_Deblur_Dataset(testopt) 
        test_gaussian_deblur(net, gaussian_deblur_testset, rank=testopt.rank, sigma=testopt.gaussian_blur_radius, select_bands = testopt.select_bands)    

    ##Needs fine-tuning
    elif testopt.mode == 6:
        print('Start Motion deblur testing motion radius={}'.format(testopt.motion_blur_radius))
        motion_deblur_testset = Motion_Deblur_Dataset(testopt) 
        test_motion_deblur(net, motion_deblur_testset, rank=testopt.rank, motion_radius=testopt.motion_blur_radius, select_bands = testopt.select_bands)    
    
    elif testopt.mode == 7:
        print('Start super-resolution testing downsampling factor={}'.format(testopt.downsample_factor))
        super_resolution_testset = Super_Resolution_Dataset(testopt) 
        test_super_resolution(net, super_resolution_testset, rank=testopt.rank, downsample_factor=testopt.downsample_factor, select_bands = testopt.select_bands)

    elif testopt.mode == 8:
        print('Start inpaint testing mask ratio ={}'.format(testopt.mask_ratio))
        inpaint_testset = Inpaint_Dataset(testopt) 
        test_inpaint(net, inpaint_testset, rank=testopt.rank, mask_ratio=testopt.mask_ratio, select_bands = testopt.select_bands)

    elif testopt.mode == 9:
        print('Start dehaze testing haze omega ={}'.format(testopt.haze_omega))
        dehaze_testset = Dehaze_Dataset(testopt) 
        test_dehaze(net, dehaze_testset, rank=testopt.rank, haze_omega=testopt.haze_omega, select_bands = testopt.select_bands)

    elif testopt.mode == 10:
        print('Start bandmis ratio ={}'.format(testopt.bandmis_ratio))
        bandmis_testset = Bandmis_Dataset(testopt) 
        test_bandmis(net, bandmis_testset, bandmis_ratio=testopt.bandmis_ratio, select_bands = testopt.select_bands)
    
    ##Zero-shot
    elif testopt.mode == 11:
        print('Start poisson degradation testing {}'.format(testopt.degrad_id))
        multi_degrad_testset = Poisson_Denoise_Dataset(testopt) 
        test_poisson_denoise(net, multi_degrad_testset, degrad_id=testopt.degrad_id, select_bands = testopt.select_bands, device = device)  

    elif testopt.mode == 12:
        print('Start real noise degradation testing {}'.format(testopt.degrad_id))
        multi_degrad_testset = Real_Degrad_Dataset(testopt) 
        test_real_denoise(net, multi_degrad_testset, degrad_id=testopt.degrad_id, select_bands = testopt.select_bands, device = device)  


