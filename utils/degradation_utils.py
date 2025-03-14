import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
from utils.image_utils import crop_img
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import convolve
from skimage.draw import disk
import os
import scipy.io as sio


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.downsample_factor = None
        self.intensity = None

    def _add_gaussian_noise(self, clean_patch, sigmas):
        min_sigma, max_sigma = sigmas
        sigma = np.random.uniform(min_sigma, max_sigma) / 255
        noise = np.random.randn(*clean_patch.shape) * sigma
        noisy_patch = clean_patch + noise

        return noisy_patch.astype(np.float32)

    def _add_gaussian_noise_non_iid(self, clean_patch, sigmas):
        sigmas = np.array(sigmas) / 255.
        bwsigmas = np.reshape(sigmas[np.random.randint(0, len(sigmas), clean_patch.shape[0])], (-1,1,1))
        noise = np.random.randn(*clean_patch.shape) * bwsigmas
        noisy_patch = clean_patch + noise

        return noisy_patch.astype(np.float32)
    
    def _add_stripe_noise(self, clean_patch, min_amount, max_amount):
        num_bands_fraction = 1/3
        B, H, W = clean_patch.shape
        stripe_patch = clean_patch.copy()
        all_bands = np.random.permutation(range(B))
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = all_bands[0:num_bands]
        num_stripe = np.random.randint(np.floor(min_amount*W), np.floor(max_amount*W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*0.5-0.25           
            stripe_patch[i, :, loc] -= np.reshape(stripe, (-1, 1))

        return stripe_patch.astype(np.float32)
    
    def _add_deadline_noise(self, clean_patch, min_amount=0.05, max_amount=0.15):
        B, H, W = clean_patch.shape
        num_bands_fraction=1/3
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = np.random.permutation(B)[:num_bands]
        num_deadline = np.random.randint(np.ceil(min_amount*W), np.ceil(max_amount*W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            clean_patch[i, :, loc] = 0

        return clean_patch.astype(np.float32)

    def _add_impulse_noise(self, clean_patch, amount, salt_vs_pepper=0.5):
        B, H, W = clean_patch.shape
        num_bands_fraction = 1/3
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = np.random.permutation(B)[:num_bands]
        for b in bands:
            p = amount  
            q = salt_vs_pepper  
            flipped = np.random.choice([True, False], size=(H, W), p=[p, 1 - p])
            salted = np.random.choice([True, False], size=(H, W), p=[q, 1 - q])
            peppered = ~salted
            clean_patch[b, flipped & salted] = 1
            clean_patch[b, flipped & peppered] = 0
        
        return clean_patch.astype(np.float32)   
    
    def _apply_poisson(self, clean_patch, scale=10.0):
        clean_patch = np.clip(clean_patch, 0, None) 
        noisy_patch = np.random.poisson(clean_patch * scale) / scale
        return noisy_patch.astype(np.float32)
    
    def _apply_gaussian_blur(self, clean_patch, kernel_size):
        B, H, W = clean_patch.shape
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        x = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1) / 2
        kernel_1d = torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()  
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0)  

        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  
        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=kernel_size // 2, stride=1, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy() 

        return output_np.astype(np.float32)

    def _apply_circle_blur(self, clean_patch, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        radius = kernel_size // 2
        center = kernel_size // 2
        for y in range(kernel_size):
            for x in range(kernel_size):
                distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if distance <= radius:
                    kernel[y, x] = np.exp(-(distance ** 2) / (2 * (radius ** 2)))
        kernel /= kernel.sum()  
        kernel_2d = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  
        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  
        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=kernel_size // 2, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy() 
        
        return output_np.astype(np.float32)
    
    def _apply_motion_blur(self, clean_patch, kernel_size, angle):
        motion_kernel = np.zeros((kernel_size, kernel_size))
        motion_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        motion_kernel /= kernel_size

        rotation_matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(motion_kernel, rotation_matrix, (kernel_size, kernel_size))

        kernel_2d = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()  

        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  

        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=kernel_size // 2, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy()
        return output_np.astype(np.float32)

    
    def _apply_square_blur(self, clean_patch, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= kernel.size  
        kernel_2d = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)

        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  

        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=kernel_size // 2, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy()

        return output_np.astype(np.float32)

    def _bicubic_downsample(self, clean_patch, downsample_factor):
        H, W = clean_patch.shape[1], clean_patch.shape[2]
        new_h = H // downsample_factor
        new_w = W // downsample_factor
        clean_patch = torch.from_numpy(clean_patch).float()
        clean_patch = clean_patch.unsqueeze(0)  
        ms = torch.nn.functional.interpolate(clean_patch, size=(new_h, new_w), mode='bicubic', align_corners=True)
        # lms = torch.nn.functional.interpolate(ms, size=(H, W), mode='bicubic', align_corners=True)
        # lms = lms.squeeze(0).detach().numpy()
        ms = ms.squeeze(0).detach().numpy()

        return ms.astype(np.float32)
    
    def _upsample(self, clean_patch, upsample_factor):
        H, W = clean_patch.shape[1], clean_patch.shape[2]
        new_h = H * upsample_factor
        new_w = W * upsample_factor
        clean_patch = torch.from_numpy(clean_patch).float()
        clean_patch = clean_patch.unsqueeze(0)  
        lms = torch.nn.functional.interpolate(clean_patch, size=(new_h, new_w), mode='bicubic', align_corners=True)
        lms = lms.squeeze(0).detach().numpy()

        return lms.astype(np.float32)
    
    def _resize(self, clean_patch, resize_factor):
        C, H, W = clean_patch.shape
        new_h = H * resize_factor
        new_w = W * resize_factor
        clean_patch = torch.from_numpy(clean_patch).float()
        clean_patch = clean_patch.unsqueeze(0)  
        clean_patch = clean_patch.unsqueeze(3).unsqueeze(5)
        expanded_patch = clean_patch.repeat(1, 1, 1, resize_factor, 1, resize_factor)
        expanded_patch = expanded_patch.view(1,C,new_h,new_w)
        expanded_patch = expanded_patch.squeeze(0).detach().numpy()

        return expanded_patch.astype(np.float32)
    
    def _sd_cassi(self, clean_patch):
        C, H, W = clean_patch.shape
        step = 2
        folder_path = ''
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        selected_file = random.choice(mat_files)
        file_path = os.path.join(folder_path, selected_file)
        mat_data = sio.loadmat(file_path)
        mask = mat_data['mask']
        start_h = random.randint(0, mask.shape[0] - H)
        start_w = random.randint(0, mask.shape[1] - W)
        mask = mask[start_h:start_h + H, start_w:start_w + W]
        mask3d = np.tile(mask[np.newaxis, :, :], (C, 1, 1))
        modulated_data = clean_patch * mask3d
        temp = np.zeros((C, H, W + (C - 1) * step), dtype=clean_patch.dtype)
        for i in range(C):
            temp[i, :, step * i:step * i + W] = modulated_data[i, :, :]
        measurement = np.sum(temp, axis=0)
        H, W = measurement.shape
        output = np.zeros((C, H, W - (C - 1) * step), dtype=clean_patch.dtype)
        for i in range(C):
            output[i, :, :] = measurement[:, step * i:step * i + W - (C - 1) * step]
        output = (output-output.min())/(output.max()-output.min())
        return output.astype(np.float32)

    def _apply_random_mask(self, hsi, mask_ratio):
        C, H, W = hsi.shape
        masked_img = np.zeros_like(hsi)
        mask = np.random.rand(C, H, W) > mask_ratio
        masked_img = hsi * mask

        return masked_img.astype(np.float32)

    def _simulate_haze(self, hsi, omega=0.2, gamma=1.0, name=None, wavelength=None, top_percent=0.01):

        cirrus_band_folder = '/home/wuzhehui/Hyper_Restoration/PromptIR-main/data/Train/Haze/512'

        mat_files = [f for f in os.listdir(cirrus_band_folder) if f.endswith('.mat')]
        if not mat_files:
            raise ValueError(f"No .mat files found in the folder: {cirrus_band_folder}")
        chosen_file = random.choice(mat_files)
        cirrus_band_path = os.path.join(cirrus_band_folder, chosen_file)
        mat_data = sio.loadmat(cirrus_band_path)

        if 'haze' not in mat_data:
            raise KeyError(f"'haze' key not found in the .mat file: {cirrus_band_path}")
        cirrus_band = mat_data['haze']

        C, H, W = hsi.shape
        cirrus_band = cv2.resize(cirrus_band, (W,H), interpolation=cv2.INTER_LINEAR)

        wavelength = np.linspace(400, 1000, 100)

        C, H, W = hsi.shape
        num_pixels = H * W
        top_k = max(int(num_pixels * top_percent / 100), 1)  
        atmospheric_light = np.zeros(C)
        for i in range(C):
            band = hsi[i, :, :].flatten()
            top_pixels = np.partition(band, -top_k)[-top_k:]  
            atmospheric_light[i] = np.mean(top_pixels)  

        t1 = 1 - omega * cirrus_band
        t1 = np.where(t1 <= 0, 1e-10, t1)

        hazy_hsi = np.zeros_like(hsi)

        for band in range(C):
            lambda_ratio = wavelength[0] / wavelength[band]          
            transmission = np.exp((lambda_ratio ** gamma) * np.log(t1))
            hazy_hsi[band] = hsi[band] * transmission + atmospheric_light[band] * (1 - transmission)

        return hazy_hsi.astype(np.float32)
    
    def _simulate_band_loss(self, hsi, loss_percentage=0.1):
        B, H, W = hsi.shape
        num_bands_to_loss = int(loss_percentage * B)
        
        lost_bands_indices = np.random.choice(B, num_bands_to_loss, replace=False)
        
        simulated_hypercube = hsi.copy()
        simulated_hypercube[lost_bands_indices] = np.zeros((H,W))
        return simulated_hypercube.astype(np.float32)
    

    def _degrade_by_type(self, clean_patch, degrade_type, degrade_range, name = None):

        if degrade_type == 'gaussianN':
            # Gaussin Denoise 
      
            sigmas = degrade_range
            degraded_patch = self._add_gaussian_noise(clean_patch, sigmas)
            

        elif degrade_type == 'complexN':
            Gaussin_sigmas = degrade_range[0]
            deadline_min_amount = degrade_range[1][0]
            deadline_max_amount = degrade_range[1][1]
            impulse_amounts = random.choice(degrade_range[2])
            stripe_min_amount = degrade_range[3][0]
            stripe_max_amount = degrade_range[3][1]
            type_idx = random.randint(0, 2)

            if type_idx == 0:
                degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, Gaussin_sigmas)
                degraded_patch = self._add_deadline_noise(degraded_patch, deadline_min_amount, deadline_max_amount)
            elif type_idx == 1:
                degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, Gaussin_sigmas)
                degraded_patch = self._add_impulse_noise(degraded_patch, impulse_amounts)
            elif type_idx == 2:
                degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, Gaussin_sigmas)
                degraded_patch = self._add_stripe_noise(degraded_patch, stripe_min_amount, stripe_max_amount)
            else:
                raise ValueError('Invalid degrade type')
            self.intensity = type_idx


        # elif degrade_type == 'gaussianN_noiid':
        #     # GaussianN_noiid Denoise 
        #     Gaussin_sigmas = [10,30,50,70]
        #     min_amount = degrade_range[0]
        #     max_amount = degrade_range[1]
        #     degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, Gaussin_sigmas)

        elif degrade_type == 'stripe':
            # Stripe Denoise 
            Gaussin_sigmas = [10,30,50,70]
            min_amount = degrade_range[0]
            max_amount = degrade_range[1]
            degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, Gaussin_sigmas)
            degraded_patch = self._add_stripe_noise(clean_patch, min_amount, max_amount)

        elif degrade_type == 'deadline':
            # Deadline Denoise 
            sigmas = [10,30,50,70]
            min_amount = degrade_range[0]
            max_amount = degrade_range[1]
            degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, sigmas)
            degraded_patch = self._add_deadline_noise(clean_patch, min_amount, max_amount)

        elif degrade_type == 'impulse':
            # Impulse Denoise 
            sigmas = [10,30,50,70]
            amounts = random.choice(degrade_range)
            degraded_patch = self._add_gaussian_noise_non_iid(clean_patch, sigmas)
            degraded_patch = self._add_impulse_noise(clean_patch, amounts)

        elif degrade_type == 'poissonN':
            # Poisson Denoise 
  
            scale = random.choice(degrade_range)
            degraded_patch = self._apply_poisson(clean_patch, scale)

        elif degrade_type == 'blur':
            # Gaussian Deblur

            blur_sigma = random.choice(degrade_range)
            degraded_patch = self._apply_gaussian_blur(clean_patch, blur_sigma)


        elif degrade_type == 'motion_blur':
            # Motion Deblur

            kernel_sizes_angles = random.choice(degrade_range)
            kernel_size, angle = kernel_sizes_angles

            degraded_patch = self._apply_motion_blur(clean_patch, kernel_size, angle)

            
        elif degrade_type == 'sr':
            # Super Resolution

            self.intensity = random.randint(0, 2)
            self.downsample_factor =  degrade_range[self.intensity]
            degraded_patch = self._bicubic_downsample(clean_patch, self.downsample_factor)

        elif degrade_type == 'inpaint':
            # Inpaint

            self.intensity = 0
            mask_ratio =  random.choice(degrade_range)
            degraded_patch= self._apply_random_mask(clean_patch, mask_ratio)

        
        elif degrade_type == 'haze':
            # Dehaze
            omega = random.choice(degrade_range)  
            degraded_patch = self._simulate_haze(hsi = clean_patch, omega = omega)  


        elif degrade_type == 'bandmiss':
            # bandmissing
            self.intensity = 0
            loss_percentage = random.choice(degrade_range)
            degraded_patch = self._simulate_band_loss(hsi = clean_patch, loss_percentage = loss_percentage)  

        elif degrade_type == 'upsample':
            # random
            degraded_patch = self._upsample(clean_patch, self.downsample_factor)
        elif degrade_type == 'resize':
            # random
            degraded_patch = self._resize(clean_patch, self.downsample_factor)

        else:
            raise ValueError('Invalid degradation type')

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None, name=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 11)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch, degrade_type = None, degrade_range = None, name = None):


        if degrade_type == 'complexN':
            degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type, degrade_range, name)
        else:
            degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type, degrade_range[0], name)
        if degrade_type == 'sr':
            degrad_patch_1, _ = self._degrade_by_type(degrad_patch_1, 'resize', degrade_range[0], name)

        return degrad_patch_1, self.intensity
    



