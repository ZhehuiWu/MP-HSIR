import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from image_utils import crop_center, minmax_normalize, rand_crop, BandMinMaxQuantileStateful
from PIL import Image
from skimage import io
import torch
import rasterio
import scipy.io as sio
import glob
from pathlib import Path
from tqdm import tqdm
from spectral import *

def create_WDC_dataset():
    imgpath = data_path
    img = io.imread(imgpath)

    img = torch.tensor(img, dtype=torch.float)
    test = img[:, 510:766, 25:281].clone()
    train_0 = img[:, :510, :].clone()
    train_1 = img[:, 766:, :].clone()

    train_0 = train_0.cpu().numpy().transpose(1,2,0)
    train_1 = train_1.cpu().numpy().transpose(1,2,0)
    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)
    savemat("/data/WDC/minmax/train/1.mat", {'data': train_0})
    savemat("/data/WDC/minmax/train/2.mat", {'data': train_1})
    savemat("/data/WDC/minmax/test/wdc_test.mat", {'data': test})


def create_Chikusei_dataset():
    imgpath = data_path
    img = h5py.File(imgpath)['chikusei']
    img = np.array(img).transpose(2,1,0).astype(np.float32)

    img = img[106:2410, 143:2191, :]  # Adjusted index for 0-based in Python
    H, W, C = img.shape
    test_img_size = 512
    test_pic_num = W // test_img_size
    for i in range(test_pic_num):
        left = i * test_img_size
        right = left + test_img_size
        test = img[:test_img_size, left:right, :]
        test = torch.tensor(test, dtype=torch.float).permute(2, 0, 1)#C，H, W
        test = (test - test.min()) / (test.max() - test.min())
        test = test.cpu().numpy().transpose(1,2,0)
        savemat(f'/data/Chikusei/minmax/test/chikusei_test_{i+1}.mat', {'data': test})

    img = img[test_img_size:, :, :]
    img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)

    img = img.cpu().numpy().transpose(1,2,0)
    savemat('/data/Chikusei/minmax/train/1.mat', {'data': img})

def create_Xiongan_dataset():
    imgpath = data_path
    with rasterio.open(imgpath) as src:
        img = src.read()
    img = img.transpose(1,2,0).astype(np.float32)

    H, W, C = img.shape
    test_img_size = 512
    test_pic_num = H // test_img_size
    for i in range(test_pic_num):
        left = i * test_img_size
        right = left + test_img_size
        test = img[left:right, :test_img_size, :]
        test = (test - test.min()) / (test.max() - test.min())

        savemat(f'/data/Xiongan/minmax/test/xiongan_test_{i+1}.mat', {'data': test})

    train_0 = img[:, test_img_size:2100, :]
    savemat('/data/Xiongan/minmax/train/1.mat', {'data': train_0})

    train_1 = img[:, 2100:, :]
    savemat('/data/Xiongan/minmax/train/2.mat', {'data': train_1})

def create_PaviaC_dataset():#
    imgpath = data_path
    img = sio.loadmat(imgpath)['pavia']
    img = img.astype(np.float32) 

    img = torch.tensor(img, dtype=torch.float)
    test = img[420:676,256:512, :].clone().permute(2, 0, 1)
    train_0 = img[:, :230, :].clone().permute(2, 0, 1)
    train_1 = img[:420, 230:, :].clone().permute(2, 0, 1)
    train_2 = img[420:676:, 512:, :].clone().permute(2, 0, 1)
    train_3 = img[676:, 230:, :].clone().permute(2, 0, 1)

    train_0 = train_0.cpu().numpy().transpose(1,2,0)
    train_1 = train_1.cpu().numpy().transpose(1,2,0)
    train_2 = train_2.cpu().numpy().transpose(1,2,0)
    train_3 = train_3.cpu().numpy().transpose(1,2,0)

    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)

    savemat("/data/PaviaC/minmax/train/1.mat", {'data': train_0})
    savemat("/data/PaviaC/minmax/train/2.mat", {'data': train_1})
    savemat("/data/PaviaC/minmax/train/3.mat", {'data': train_2})
    savemat("/data/PaviaC/minmax/train/4.mat", {'data': train_3})
    savemat("/data/PaviaC/minmax/test/paviac_test.mat", {'data': test})

def create_PaviaU_dataset():#
    imgpath = data_path
    img = sio.loadmat(imgpath)['paviaU']
    img = img.astype(np.float32) 

    img = torch.tensor(img, dtype=torch.float)
    test = img[200:400,:, :].clone().permute(2, 0, 1)
    train_0 = img[:200, :, :].clone().permute(2, 0, 1)
    train_1 = img[400:, :, :].clone().permute(2, 0, 1)

    train_0 = train_0.cpu().numpy().transpose(1,2,0)
    train_1 = train_1.cpu().numpy().transpose(1,2,0)

    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)

    savemat("/data/PaviaU/minmax/train/1.mat", {'data': train_0})
    savemat("/data/PaviaU/minmax/train/2.mat", {'data': train_1})
    savemat("/data/PaviaU/minmax/test/paviau_test.mat", {'data': test})

def create_Houston_dataset():#
    imgpath = data_path
    with rasterio.open(imgpath) as src:
        img = src.read().astype(np.float32).transpose(1, 2, 0)

    img = torch.tensor(img, dtype=torch.float)
    test = img[:, 1024:1280, :].clone().permute(2, 0, 1)
    train_0 = img[:, :1024, :].clone().permute(2, 0, 1)
    train_1 = img[:, 1280:, :].clone().permute(2, 0, 1)

    train_0 = train_0.cpu().numpy().transpose(1,2,0)
    train_1 = train_1.cpu().numpy().transpose(1,2,0)

    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)

    savemat("/data/Houston/minmax/train/1.mat", {'data': train_0})
    savemat("/data/Houston/minmax/train/2.mat", {'data': train_1})
    savemat("/data/Houston/minmax/test/houston_test.mat", {'data': test})

def create_Eagle_dataset():#
    imgpath = data_path
    with rasterio.open(imgpath) as src:
        img = src.read()
    mask = np.all(img == 0, axis=0)#CHW
        
    img = img.astype(np.float32).transpose(1, 2, 0)#HWC
    img = torch.tensor(img, dtype=torch.float)
    test = img[1024:1280, 1024:1280, :248].clone().permute(2, 0, 1)
    test_mask = mask[1024:1280, 1024:1280]
    train_0 = img[:,:,:248].clone().permute(2, 0, 1)
    train_mask = mask
    train_mask[1024:1280, 1024:1280] = True

    train_0 = train_0.cpu().numpy().transpose(1,2,0)

    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)

    savemat("/data/Eagle/minmax/train/1.mat", {'data': train_0, 'mask': train_mask})
    savemat("/data/Eagle/minmax/test/eagle_test.mat", {'data': test, 'mask': test_mask})

def create_Berlin_dataset():#
    imgpath = data_path
    with rasterio.open(imgpath) as src:
        img = src.read()
    mask = np.all(img == 0, axis=0)#CHW

    img = img.astype(np.float32).transpose(1, 2, 0)#HWC
    img = torch.tensor(img, dtype=torch.float)
    test = img[3000:3512, 600:1112, :].clone().permute(2, 0, 1)
    test_mask = mask[3000:3512, 600:1112]
    mask[3000:3512, 600:1112] = True
    train_0 = img[:,:600,:].clone().permute(2, 0, 1)
    train_1 = img[:, 1112:, :].clone().permute(2, 0, 1)
    train_2 = img[:3000:, :, :].clone().permute(2, 0, 1)
    train_3 = img[3512:, :, :].clone().permute(2, 0, 1)
    train_mask_0 = mask[:,:600]
    train_mask_1 = mask[:, 1112:]
    train_mask_2 = mask[:3000, :]
    train_mask_3 = mask[3512:, :]

    train_0 = train_0.cpu().numpy().transpose(1,2,0)
    train_1 = train_1.cpu().numpy().transpose(1,2,0)
    train_2 = train_2.cpu().numpy().transpose(1,2,0)
    train_3 = train_3.cpu().numpy().transpose(1,2,0)

    test = (test - test.min()) / (test.max() - test.min())
    test = test.cpu().numpy().transpose(1,2,0)

    savemat("/data/BerlinUrGrad/minmax/train/1.mat", {'data': train_0, 'mask': train_mask_0},)
    savemat("/data/BerlinUrGrad/minmax/train/2.mat", {'data': train_1, 'mask': train_mask_1},)
    savemat("/data/BerlinUrGrad/minmax/train/3.mat", {'data': train_2, 'mask': train_mask_2},)
    savemat("/data/BerlinUrGrad/minmax/train/4.mat", {'data': train_3, 'mask': train_mask_3},)
    savemat("/data/BerlinUrGrad/minmax/test/berlin_test.mat", {'data': test, 'mask': test_mask})

def create_Apex_dataset():
    img = open_image(data_path)
    img = img.load()
    img = img.transpose((2,0,1))
    data = img[:210]
    total_num = 20

    save_dir = '/data/APEX/minmax/Train/'
    for i in range(total_num):
        data = rand_crop(data, 512, 512)
        data = minmax_normalize(data)
        savemat(save_dir+str(i)+'.mat',{'data': data})
        print(i)

def create_Urban_dataset():
    img = loadmat(data_path)
    imgg  = img['Y'].reshape((210,307,307))
    imggt = imgg.astype(np.float32)
    norm_gt = imggt.transpose((1,2,0))
    norm_gt = norm_gt[:304,:304,:]
    norm_gt = minmax_normalize(norm_gt)

    savemat("/data/Urban/minmax/Urban_F210.mat", {'data': norm_gt})

def create_EO1_Hyperion_dataset():#
    imgpath = data_path
    file_paths = sorted(glob.glob(os.path.join(imgpath, '*.TIF')), 
                        key=lambda x: int(x.split('_')[-2][1:]))

    remove_indices = list(range(1, 8)) + list(range(58, 77)) + \
                    list(range(121, 127)) + list(range(167, 181)) + \
                    list(range(222, 243))

    remove_indices = [i - 1 for i in remove_indices]

    remaining_file_paths = [file for idx, file in enumerate(file_paths) if idx not in remove_indices]

    image_stack = []

    for file_path in remaining_file_paths:
        with rasterio.open(file_path) as src:
            image = src.read()  
            image_stack.append(image[0,:,:].astype('float32')) 
    image_stack = np.stack(image_stack, axis=0) 

    mask = np.all(image_stack == 0, axis=0)#CHW

    image_stack = torch.tensor(image_stack, dtype=torch.float)

    image_stack = image_stack.cpu().numpy().transpose(1,2,0)
    image_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min())

    savemat("/data/EO1_Hyperion/EO1/1.mat", {'data': image_stack, 'mask': mask})

def create_ICVL_train():
    save_path = Path('/data/ICVL/train')
    src_img_root_dir_path = Path("/data/ICVL_raw/train")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['rad']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        img = data.astype(np.float32).transpose(1,2,0)#
        savemat(save_path / f'{idx}.mat', {'data':img})
        idx+=1  

def create_ICVL_test():
    save_path = Path('/data/ICVL/test')
    src_img_root_dir_path = Path("/data/ICVL_raw/test")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))
    crop_sizes=(512, 512)

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['rad']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        data = crop_center(data, crop_sizes[0], crop_sizes[1])
        img = data.astype(np.float32).transpose(1,2,0)#
        savemat(save_path / f'ICVL_test_{idx}.mat', {'data':img})
        idx+=1  

def create_ARAD_train():
    save_path = Path('/data/ARAD_1k/train')
    src_img_root_dir_path = Path("/data/ARAD_1k_raw/train")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['cube']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=1, axes=(2,1)) # ARAD
        img = data.astype(np.float32).transpose(1,2,0)#
        savemat(save_path / f'{idx}.mat', {'data':img})
        idx+=1  

def create_ARAD_test():
    save_path = Path('/data/ARAD_1k/test')
    src_img_root_dir_path = Path("/data/ARAD_1k_raw/test")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['cube']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=1, axes=(2,1)) # ARAD
        img = data.astype(np.float32).transpose(1,2,0)#
        savemat(save_path / f'ARAD_test_{idx}.mat', {'data':img})
        idx+=1


if __name__ == '__main__':
    data_path = '/data/EO1_Hyperion/EO1_test'
    #create_WDC_dataset()
    #create_Chikusei_dataset()
    #create_Xiongan_dataset()
    #create_PaviaC_dataset()
    #create_PaviaU_dataset()
    #create_Houston_dataset()
    #create_Eagle_dataset()
    #create_Berlin_dataset()
    create_EO1_Hyperion_dataset()
    #create_ICVL_train()
    #create_ICVL_test()
    #create_ARAD_train()
    #create_ARAD_test()
