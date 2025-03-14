import os
import random
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

from utils.image_utils import *
import cv2
import scipy.io as sio
import lmdb
import torch.utils.data as data




class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])

        return img


class LMDBDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        if self.args.classifier == True:
            self.db_path = self.args.classifier_path
        else:
            self.db_path = self.args.db_path
        self.env = lmdb.open(self.db_path, max_readers=64, readonly=True, lock=False,
                             readahead=True, meminit=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.length = int(self.length)
         
        self.meta_info = self._read_meta_info()

        self.valid_idxs = []
        self.valid_meta_info = {}
        self.dataset_names = ['BerlinUrGrad','Chikusei','Eagle','Xiongan','Houston','PaviaC','PaviaU','WDC']  #'ARAD_1k','ICVL' ##'BerlinUrGrad','Chikusei','Eagle','Xiongan','Houston','PaviaC','PaviaU','WDC'
        if self.dataset_names:
            for idx, info in self.meta_info.items():
                if any(info['source_file'].startswith(name) for name in self.dataset_names):
                    self.valid_idxs.append(idx)
                    self.valid_meta_info[f'{idx}'] = info

            self.length = len(self.valid_idxs)  


    def _read_meta_info(self):
        meta_info = {}
        with open(os.path.join(self.db_path, 'meta_info.txt')) as fin:
            lines = fin.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                idx = parts[0]
                dimensions = tuple(map(int, parts[1].strip('()').split(',')))
                source_file = parts[2].strip('source_file=')

                meta_info[idx] = {
                    'dimensions': dimensions,
                    'source_file': source_file
                }
        return meta_info
      
    def __getitem__(self, index):
        index = index % (self.length)
        env = self.env
        index_str = self.valid_idxs[index]

        with env.begin(write=False) as txn:
            data = txn.get(index_str.encode('ascii'))

        meta = self.valid_meta_info[f'{index_str}']
        dimensions = meta['dimensions']
        C, H, W = (dimensions[2], dimensions[0], dimensions[1])
        source_file = meta['source_file']
        X = np.frombuffer(data, dtype=np.float32)
        X = X.reshape(C, H, W)    

        return X,source_file

    def __len__(self):
        return self.length 

class ImageTransformDataset(Dataset):
    def __init__(self, dataset, args):
        super(ImageTransformDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.D = Degradation(args)
        self.to_tensor = HSI2Tensor(use_2dconv=True)
        self.data_type = self.args.data_type

        if self.data_type == 'natural_scene':
            self.de_dict = {'gaussianN': [(30, 70)], 'complexN':[(10,30,50,70),(0.05, 0.15),(0.1, 0.3, 0.5, 0.7),(0.05, 0.15)], 'blur':[(9, 15, 21)], 'sr': [(2, 4, 8)], 'inpaint':[(0.7, 0.8, 0.9)], 'bandmiss':[(0.1,0.2,0.3)], 'cassi':[(0,)], 'motion_blur':[((15,45),)]}#[(9, 15, 21)]

            self.de_type = self.args.natural_scene_single_de_type

        elif self.data_type == 'remote_sensing':
            self.de_dict = {'gaussianN': [(30, 70)], 'complexN':[(10,30,50,70),(0.05, 0.15),(0.1, 0.3, 0.5, 0.7),(0.05, 0.15)], 'blur':[(7, 11, 15)], 'sr': [(2, 4, 8)], 'inpaint':[(0.7, 0.8, 0.9)], 'haze':[(0.5, 0.75, 1)], 'bandmiss':[(0.1,0.2,0.3)], 'circle_blur':[(9,)], 'poissonN':[(10,)]}
            
            self.de_type = self.args.remote_sensing_single_de_type
        else:
            raise ValueError('data type must be natural_scene or remote_sensing')
        
        self.length = len(self.dataset)

    def __len__(self):
        return self.length * self.args.repeat

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        img, name = self.dataset[idx]

        if self.data_type == 'natural_scene' and img.shape[0] != 31:
            img, _ = interpolate_bands(img, 31)

        de_id = random.randint(0, len(self.de_type)-1)
        de_range = self.de_dict[self.de_type[de_id]]
        clean_patch = img.copy()
        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), self.de_type[de_id], de_range, name)

        prompt = torch.tensor([de_id])

        degrad_patch, clean_patch = random_augmentation(*(degrad_patch, clean_patch))

        clean_patch = self.to_tensor(clean_patch)
        degrad_patch = self.to_tensor(degrad_patch)
        return [name, de_id], degrad_patch, clean_patch, prompt


class Classifier_Dataset(Dataset):
    def __init__(self, dataset, args):
        super(Classifier_Dataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.D = Degradation(args)
        self.to_tensor = HSI2Tensor(use_2dconv=True)
        self.data_type = self.args.data_type

        if self.data_type == 'natural_scene':

            self.de_dict = {'gaussianN': [(30, 70)], 'deadline':[(0.05, 0.15)], 'stripe':[(0.05, 0.15)], 'impulse':[(0.1, 0.3, 0.5, 0.7)], 'blur':[(9, 15, 21)], 'sr': [(2, 4, 8)], 'inpaint':[(0.7, 0.8, 0.9)], 'bandmiss':[(0.1,0.2,0.3)]}           
            self.de_type = ['gaussianN', 'deadline', 'impulse', 'stripe', 'blur', 'sr', 'inpaint']

            
        elif self.data_type == 'remote_sensing':

            self.de_dict = {'gaussianN': [(30, 70)],  'deadline':[(0.05, 0.15)], 'stripe':[(0.05, 0.15)], 'impulse':[(0.1, 0.3, 0.5, 0.7)], 'blur':[(9, 15, 21)], 'sr': [(2, 4, 8)], 'inpaint':[(0.7, 0.8, 0.9)], 'haze':[(0.5, 0.75, 1)], 'bandmiss':[(0.1,0.2,0.3)]}           
            self.de_type = ['gaussianN', 'deadline', 'impulse', 'stripe', 'blur', 'sr', 'inpaint', 'haze']
        else:
            raise ValueError('data type must be natural_scene or remote_sensing')
        
        self.length = len(self.dataset)

    def get_degradation_label(self, degradation_combination):

        label = np.zeros(5)
        #label = np.zeros(6)
        for i, unit in enumerate(self.de_type):
            if unit in degradation_combination:
                if i in {1, 2, 3}:  
                    label[1] = 1
                elif i == 0:
                    label[i] = 1
                elif i > 3:
                    label[i - 2] = 1  
        return label


    def __len__(self):
        return self.length * self.args.repeat

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        img, name = self.dataset[idx]

        if self.data_type == 'natural_scene' and img.shape[0] != 31:
            img, _ = interpolate_bands(img, 31)


        de_id = random.randint(0, len(self.de_type)-1)#0-2
        de_range = self.de_dict[self.de_type[de_id]]#10,70
        clean_patch = img.copy()
        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), self.de_type[de_id], de_range, name)
        label = self.get_degradation_label(self.de_type[de_id])

        degrad_patch = random_augmentation(degrad_patch)[0]

        degrad_patch = self.to_tensor(degrad_patch)

        return [name, self.de_type[de_id]], degrad_patch, label
    

class Real_Degrad_Dataset(Dataset):
    def __init__(self, args):
        super(Real_Degrad_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.noisy_ids = []
        self._init_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_ids(self):
        name_list = os.listdir(self.args.test_dir)
        degrad_name_list  = os.listdir(self.args.test_degrad_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.noisy_ids += [self.args.test_degrad_dir + '/' + id_ for id_ in degrad_name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def __getitem__(self, idx):
        clean_img = crop_img(np.array(sio.loadmat(self.clean_ids[idx])['data']), base=64)
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = crop_img(np.array(sio.loadmat(self.noisy_ids[idx])['data']), base=64)

        clean_img, noisy_img = self.to_tensor(clean_img), self.to_tensor(noisy_img)
        
        return [clean_name], noisy_img.float(), clean_img.float()
        
    def __len__(self):
        return self.num_clean



class Poisson_Denoise_Dataset(Dataset):
    def __init__(self, args):
        super(Poisson_Denoise_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = self.args.gaussian_noise_sigma
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def apply_poisson(self, clean_patch, scale=10.0):
        clean_patch = np.clip(clean_patch, 0, None)  
        noisy_patch = np.random.poisson(clean_patch * scale) / scale
        return noisy_patch

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.apply_poisson(clean_img_)

        clean_img, noisy_img = self.to_tensor(clean_img), self.to_tensor(noisy_img)
        
        return [clean_name], noisy_img.float(), clean_img.float()
        
    def __len__(self):
        return self.num_clean

class Gaussian_Denoise_Dataset(Dataset):
    def __init__(self, args):
        super(Gaussian_Denoise_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = self.args.gaussian_noise_sigma
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)


    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def add_gaussian_noise(self, clean_patch):
        sigma = self.sigma / 255
        noise = np.random.randn(*clean_patch.shape) * sigma
        noisy_patch = clean_patch + noise

        return noisy_patch

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.add_gaussian_noise(clean_img_)

        clean_img, noisy_img = self.to_tensor(clean_img), self.to_tensor(noisy_img)
        
        return [clean_name], noisy_img.float(), clean_img.float()
        
    def __len__(self):
        return self.num_clean


class Gaussian_Denoise_inid_Dataset(Dataset):
    def __init__(self, args):
        super(Gaussian_Denoise_inid_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigmas = self.args.gaussian_noise_sigmas
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def add_gaussian_noise_inid(self, clean_patch):
        sigmas = np.array(self.sigmas) / 255.
        bwsigmas = np.reshape(sigmas[np.random.randint(0, len(sigmas), clean_patch.shape[0])], (-1,1,1))
        noise = np.random.randn(*clean_patch.shape) * bwsigmas
        noisy_patch = clean_patch + noise
        return noisy_patch
    
    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.add_gaussian_noise_inid(clean_img_)
        clean_img, noisy_img = self.to_tensor(clean_img), self.to_tensor(noisy_img)

        return [clean_name], noisy_img.float(), clean_img.float()
        
    def __len__(self):
        return self.num_clean


class Stripe_Denoise_Dataset(Dataset):
    def __init__(self, args):
        super(Stripe_Denoise_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.stripe_ratio = self.args.stripe_nosie_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))
    
    def add_stripe(self, clean_patch):
        num_bands = 1 / 3
        B, H, W = clean_patch.shape
        stripe_patch = clean_patch.copy()
        all_bands = np.random.permutation(range(B))
        num_band = int(np.floor(num_bands * B))
        bands = all_bands[0:num_band]
        
        min_ratio, max_ratio = self.stripe_ratio
        num_stripe = [np.random.randint(int(min_ratio * W), int(max_ratio * W)) for _ in range(len(bands))]
        
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            stripe_patch[i, :, loc] -= np.reshape(stripe, (-1, 1))
        
        return stripe_patch
    
    def add_gaussian_noise_iid(self, clean_patch):
        sigmas = [10,30,50,70]
        sigmas = np.array(sigmas) / 255.
        bwsigmas = np.reshape(sigmas[np.random.randint(0, len(sigmas), clean_patch.shape[0])], (-1,1,1))
        noise = np.random.randn(*clean_patch.shape) * bwsigmas
        noisy_patch = clean_patch + noise

        return noisy_patch    

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.add_gaussian_noise_iid(clean_img_)
        stripe_img = self.add_stripe(noisy_img)
        clean_img, stripe_img = self.to_tensor(clean_img), self.to_tensor(stripe_img)

        return [clean_name], stripe_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean

class Deadline_Denoise_Dataset(Dataset):
    def __init__(self, args):
        super(Deadline_Denoise_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.deadline_ratio = self.args.deadline_nosie_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))
    
    def add_deadline_noise(self, clean_patch):
        B, H, W = clean_patch.shape
        num_bands_fraction = 1 / 3
        num_bands = int(np.floor(num_bands_fraction * B))  
        bands = np.random.permutation(B)[:num_bands]  
        min_amount, max_amount = self.deadline_ratio

        num_deadline = np.random.randint(
            low=int(np.ceil(min_amount * W)),
            high=int(np.ceil(max_amount * W)),
            size=num_bands
        )  # (num_bands,)


        for i in range(num_bands):
            band_idx = bands[i]  
            n_stripes = num_deadline[i]  
            loc = np.random.permutation(W)[:n_stripes]  # (n_stripes,)
            clean_patch[band_idx, :, loc] = 0  

        return clean_patch

    def add_gaussian_noise_iid(self, clean_patch):
        sigmas = [10,30,50,70]
        sigmas = np.array(sigmas) / 255.
        bwsigmas = np.reshape(sigmas[np.random.randint(0, len(sigmas), clean_patch.shape[0])], (-1,1,1))
        noise = np.random.randn(*clean_patch.shape) * bwsigmas
        noisy_patch = clean_patch + noise

        return noisy_patch 

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.add_gaussian_noise_iid(clean_img_)
        deadline_img = self.add_deadline_noise(noisy_img)
        clean_img, deadline_img = self.to_tensor(clean_img), self.to_tensor(deadline_img)

        return [clean_name], deadline_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean
    
class Impulse_Denoise_Dataset(Dataset):
    def __init__(self, args):
        super(Impulse_Denoise_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.impulse_ratio = self.args.impulse_nosie_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))
    
    def _add_impulse_noise(self, clean_patch, salt_vs_pepper=0.5):
        B, H, W = clean_patch.shape
        num_bands_fraction = 1/3
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = np.random.permutation(B)[:num_bands]
        p = random.choice(self.impulse_ratio)  
        q = salt_vs_pepper  
        for b in bands:

            flipped = np.random.choice([True, False], size=(H, W), p=[p, 1 - p])
            salted = np.random.choice([True, False], size=(H, W), p=[q, 1 - q])
            peppered = ~salted

            clean_patch[b, flipped & salted] = 1
            clean_patch[b, flipped & peppered] = 0
        
        return clean_patch

    def add_gaussian_noise_iid(self, clean_patch):
        sigmas = [10,30,50,70]
        sigmas = np.array(sigmas) / 255.
        bwsigmas = np.reshape(sigmas[np.random.randint(0, len(sigmas), clean_patch.shape[0])], (-1,1,1))
        noise = np.random.randn(*clean_patch.shape) * bwsigmas
        noisy_patch = clean_patch + noise

        return noisy_patch 

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        noisy_img = self.add_gaussian_noise_iid(clean_img_)
        impulse_img = self._add_impulse_noise(noisy_img)
        clean_img, impulse_img = self.to_tensor(clean_img), self.to_tensor(impulse_img)

        return [clean_name], impulse_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean    

class Impulse_Denoise_inid_Dataset(Dataset):
    def __init__(self, args):
        super(Impulse_Denoise_inid_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.impulse_ratio = self.args.impulse_nosie_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))
    
    def _add_impulse_noise(self, clean_patch, salt_vs_pepper=0.5):
        B, H, W = clean_patch.shape
        num_bands_fraction = 1/3
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = np.random.permutation(B)[:num_bands]
        for b in bands:
         
            p = random.choice([0.1,0.3,0.5,0.7])  
            q = salt_vs_pepper  

            flipped = np.random.choice([True, False], size=(H, W), p=[p, 1 - p])
            salted = np.random.choice([True, False], size=(H, W), p=[q, 1 - q])
            peppered = ~salted

            clean_patch[b, flipped & salted] = 1
            clean_patch[b, flipped & peppered] = 0
        
        return clean_patch

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        impulse_img = self._add_impulse_noise(clean_img_)
        clean_img, impulse_img = self.to_tensor(clean_img), self.to_tensor(impulse_img)

        return [clean_name], impulse_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean  

class Gaussian_Deblur_Dataset(Dataset):
    def __init__(self, args):
        super(Gaussian_Deblur_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.radius= self.args.gaussian_blur_radius
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def apply_gaussian_blur(self, clean_patch):
        B, H, W = clean_patch.shape
        sigma = 0.3 * ((self.radius - 1) * 0.5 - 1) + 0.8
        x = torch.arange(self.radius, dtype=torch.float32)
        mean = (self.radius - 1) / 2
        kernel_1d = torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()  
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0)  
        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  
        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=self.radius // 2, stride=1, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy() 

        return output_np

    def add_gaussian_noise(self, clean_patch):
        sigma = 30
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = clean_patch + noise * sigma / 255
        return noisy_patch

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        blurred_img = self.apply_gaussian_blur(clean_img_)

        clean_img, blurred_img = self.to_tensor(clean_img), self.to_tensor(blurred_img)
        
        return [clean_name], blurred_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean  

class Motion_Deblur_Dataset(Dataset):
    def __init__(self, args):
        super(Motion_Deblur_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.kernel_size, self.angle= self.args.motion_blur_radius
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def _apply_motion_blur(self, clean_patch):
        B, H, W = clean_patch.shape
        motion_kernel = np.zeros((self.kernel_size, self.kernel_size))
        motion_kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        motion_kernel /= self.kernel_size

        rotation_matrix = cv2.getRotationMatrix2D((self.kernel_size / 2, self.kernel_size / 2), self.angle, 1)
        kernel = cv2.warpAffine(motion_kernel, rotation_matrix, (self.kernel_size, self.kernel_size))

        kernel_2d = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, kernel_size, kernel_size)

        input_tensor = torch.from_numpy(clean_patch).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  
            kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)  

        blurred_image = F.conv2d(input_tensor, kernel_2d, padding=self.kernel_size // 2, groups=input_tensor.shape[1])
        output_np = blurred_image.squeeze(0).detach().numpy()
        
        return output_np

    def add_gaussian_noise(self, clean_patch):
        sigma = 30
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = clean_patch + noise * sigma / 255
        return noisy_patch

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        motion_blurred_img = self._apply_motion_blur(clean_img_)

        clean_img, motion_blurred_img = self.to_tensor(clean_img), self.to_tensor(motion_blurred_img)

        return [clean_name], motion_blurred_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean 
    

class Super_Resolution_Dataset(Dataset):
    def __init__(self, args):
        super(Super_Resolution_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.downsample_factor = self.args.downsample_factor
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def bicubic_downsample(self, clean_patch):
        C, H, W = clean_patch.shape[0],clean_patch.shape[1],clean_patch.shape[2]
        new_h = H // self.downsample_factor
        new_w = W // self.downsample_factor

        clean_patch = torch.from_numpy(clean_patch).float()
        clean_patch = clean_patch.unsqueeze(0)  

        ms = torch.nn.functional.interpolate(clean_patch, size=(new_h, new_w), mode='bicubic', align_corners=True)

        clean_patch = ms.unsqueeze(3).unsqueeze(5)
        expanded_patch = clean_patch.repeat(1, 1, 1, self.downsample_factor, 1, self.downsample_factor)
        expanded_patch = expanded_patch.view(1,C,H,W)
        lms = expanded_patch.squeeze(0).detach().numpy()

        return lms

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        lms = self.bicubic_downsample(clean_img_)

        clean_img, lms = self.to_tensor(clean_img), self.to_tensor(lms)
        
        return [clean_name], lms.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean


class Inpaint_Dataset(Dataset):
    def __init__(self, args):
        super(Inpaint_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.mask_ratio = self.args.mask_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def _apply_mask(self, hsi):
        C, H, W = hsi.shape
        masked_img = np.zeros_like(hsi)
        mask = np.random.rand(C, H, W) > self.mask_ratio
        masked_img = hsi * mask

        return masked_img, mask

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)

        clean_img_tensor = torch.tensor(clean_img, dtype=torch.float32)

        clean_img = clean_img_tensor.squeeze(0).numpy()

        
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        masked_img, mask = self._apply_mask(clean_img_)

        clean_img, masked_img, mask = self.to_tensor(clean_img), self.to_tensor(masked_img), self.to_tensor(mask)

        return [clean_name], masked_img.float(), clean_img.float(), mask.float()
    
    def __len__(self):
        return self.num_clean

class Dehaze_Dataset(Dataset):
    def __init__(self, args):
        super(Dehaze_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.haze_omega = self.args.haze_omega
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def _simulate_haze(self, hsi, omega=0.5, gamma=1.0, name=None, wavelength=None, top_percent=0.01):

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

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        hazed_img = self._simulate_haze(hsi = clean_img_,omega = self.haze_omega,name = clean_name)
        clean_img, hazed_img = self.to_tensor(clean_img), self.to_tensor(hazed_img)

        return [clean_name], hazed_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean

class Bandmis_Dataset(Dataset):
    def __init__(self, args):
        super(Bandmis_Dataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.bandmis_ratio = self.args.bandmis_ratio
        self._init_clean_ids()
        self.to_tensor = HSI2Tensor(use_2dconv = True)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.test_dir)
        self.clean_ids += [self.args.test_dir + '/' + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)
        print("Total Test HSIs Ids : {}".format(self.num_clean))

    def _simulate_band_loss(self, hsi, loss_percentage=0.1):
        loss_percentage = self.bandmis_ratio
        B, H, W = hsi.shape
        num_bands_to_loss = int(loss_percentage * B)
        
        lost_bands_indices = np.random.choice(B, num_bands_to_loss, replace=False)
        
        simulated_hypercube = hsi.copy()
        simulated_hypercube[lost_bands_indices] = np.zeros((H,W))
        return simulated_hypercube.astype(np.float32)

    def __getitem__(self, idx):
        clean_id = self.clean_ids[idx]
        clean_img = crop_img(np.array(sio.loadmat(clean_id)['data']), base=64)
        clean_img_ = clean_img.copy()
        clean_name = self.clean_ids[idx].split("/")[-1].split('.')[0]
        hazed_img = self._simulate_band_loss(hsi = clean_img_)
        clean_img, hazed_img = self.to_tensor(clean_img), self.to_tensor(hazed_img)

        return [clean_name], hazed_img.float(), clean_img.float()
    
    def __len__(self):
        return self.num_clean


