"""Create lmdb dataset"""
from image_utils import *
import lmdb
import scipy.io as scio
import h5py
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pickle
from scipy import interpolate
torch.set_num_threads(1)


def process_files(source_folder, destination_folder):

    folder_name = os.path.basename(os.path.dirname(os.path.dirname(source_folder))) #remote_sensing
    # folder_name = os.path.basename(os.path.dirname(source_folder)) #natural scene

    file_names = [f for f in os.listdir(source_folder) if f.endswith(".mat")]

    for file_name in tqdm(file_names, desc=f"Processing {folder_name}", unit="file"):
        if file_name.endswith(".mat"):
            src_file_path = os.path.join(source_folder, file_name)
            
            mat_file = sio.loadmat(src_file_path)
            data = mat_file['data']

            new_data = {"data": data}

            if 'mask' in mat_file:
                new_data['mask'] = mat_file['mask']

            dest_file_name = f"{folder_name}_{file_name}"
            dest_file_path = os.path.join(destination_folder, dest_file_name)
            sio.savemat(dest_file_path, new_data)


def create_lmdb(
    datadir, fns, name, 
    scales, ksizes, strides,
    load=sio.loadmat, augment=True, seed=2024):
    def preprocess(data, mask, ksizes, strides):
        new_data = [] 

        height, width = data.shape[-2], data.shape[-1]
        new_height = (height // 256) * 256
        new_width = (width // 256) * 256
        data = data[..., :new_height, :new_width]

        if mask is not None:
            mask = mask[:new_height, :new_width]
        else:
            mask = np.zeros((new_height, new_width), dtype=bool)

        for i in range(len(scales)):
            if scales[i] != 1:
                temp_data = zoom(data, zoom=(1, scales[i], scales[i]))
                temp_mask = zoom(mask, zoom=(scales[i], scales[i]), order=0)  
            else:
                temp_data = data
                temp_mask = mask
            
            # Convert to patches while ensuring all patches are within valid regions

            valid_patches = Data2Volume(temp_data, temp_mask, ksizes=ksizes, strides=list(strides[i]))
            new_data.extend(valid_patches)
        
        new_data = np.stack(new_data, axis=0) if new_data else np.zeros((0,) + tuple(ksizes))
                
        return new_data.astype(np.float32)
    
    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    
    # Initialize LMDB environment
    if datadir == "/data/Train/Natural_scene":
        map_size = 80 * 1024 ** 3  # 150GB in byte
    else:
        map_size = 80 * 1024 ** 3  # 200GB in byte
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    txt_file = open(os.path.join(name+'.db', 'meta_info.txt'), 'w')

    with env.begin(write=True) as txn:
        k = 0
        for i, fn in tqdm(enumerate(fns), desc=f"Processing {name}", unit="file"):
            try:
                # Load .mat file
                mat_data = load(os.path.join(datadir, fn))
                data = mat_data['data'].transpose(2,0,1)#CHW
                mask = mat_data.get('mask', None)  
                
                C, H, W = data.shape[-3], data.shape[-2], data.shape[-1]

                adjusted_ksizes = (C, ksizes[1], ksizes[2])
                adjusted_strides = [(C, s[1], s[2]) for s in strides]

                X = preprocess(data, mask, adjusted_ksizes, adjusted_strides)


            except Exception as e:
                print(f"Error processing file {fn}: {e}")
                continue
            
            # Save patches to LMDB
            for j in range(X.shape[0]):
                c, h, w = X.shape[1:]
                X_byte = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txt_file.write(f'{str_id} ({h},{w},{c}) source_file={fn}\n')
                txn.put(str_id.encode('ascii'), X_byte)  # Add data with key
            
            print(f'Processed file ({i+1}/{len(fns)}): {fn}')
    
    print('Database creation completed.')

def create_lmdb_remote_sensing(
    datadir, fns, name, 
    scales, ksizes, strides,
    load=sio.loadmat, augment=True, seed=2024):
    def preprocess(data, mask, ksizes, strides):
        new_data = [] 

        height, width = data.shape[-2], data.shape[-1]
        new_height = (height // 128) * 128
        new_width = (width // 128) * 128
        data = data[..., :new_height, :new_width]

        if mask is not None:
            mask = mask[:new_height, :new_width]
        else:
            mask = np.zeros((new_height, new_width), dtype=bool)

        for i in range(len(scales)):
            if scales[i] != 1:
                temp_data = zoom(data, zoom=(1, scales[i], scales[i]))
                temp_mask = zoom(mask, zoom=(scales[i], scales[i]), order=0)  
            else:
                temp_data = data
                temp_mask = mask
            
            # Convert to patches while ensuring all patches are within valid regions

            valid_patches = Data2Volume(temp_data, temp_mask, ksizes=ksizes, strides=list(strides[i]))
            new_data.extend(valid_patches)
        
        new_data = np.stack(new_data, axis=0) if new_data else np.zeros((0,) + tuple(ksizes))
                
        return new_data.astype(np.float32)
    
    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)

    datasets = {
        'Xiongan': {'range': (400, 1000), 'bands': 256},
        'WDC': {'range': (400, 2400), 'bands': 191,},
        'PaviaC': {'range': (430, 860), 'bands': 102},
        'PaviaU': {'range': (430, 860), 'bands': 103},
        'Houston': {'range': (364, 1046), 'bands': 144},
        'Chikusei': {'range': (343, 1018), 'bands': 128},
        'Eagle': {'range': (401, 999), 'bands': 248},
        'BerlinUrGrad': {'range': (455, 2447), 'bands': 111}
    }
    
    target_wavelength = np.linspace(400, 1000, 100)
    # Initialize LMDB environment
    if datadir == "/data/Train/Natural_scene":
        map_size = 80 * 1024 ** 3  # 150GB in byte
    else:
        map_size = 80 * 1024 ** 3  # 200GB in byte
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    txt_file = open(os.path.join(name+'.db', 'meta_info.txt'), 'w')

    with env.begin(write=True) as txn:
        k = 0
        for i, fn in tqdm(enumerate(fns), desc=f"Processing {name}", unit="file"):
            try:
                # Load .mat file
                mat_data = load(os.path.join(datadir, fn))
                data = mat_data['data'].transpose(2,0,1)#CHW
                mask = mat_data.get('mask', None)  
                
                dataset_name = fn.split('_')[0]  
                dataset_info = datasets.get(dataset_name)

                if dataset_info is None:
                    raise ValueError(f"Dataset {dataset_name} not found in datasets dictionary.")

                original_wavelength_min, original_wavelength_max = dataset_info['range']
                original_bands = dataset_info['bands']

                original_wavelength = np.linspace(original_wavelength_min, original_wavelength_max, original_bands)


                interp_func = interpolate.interp1d(original_wavelength, data, axis=0, kind='linear', fill_value="extrapolate")
                data = interp_func(target_wavelength)
                print(data.shape)


                C, H, W = data.shape[-3], data.shape[-2], data.shape[-1]
                adjusted_ksizes = (C, ksizes[1], ksizes[2])
                adjusted_strides = [(C, s[1], s[2]) for s in strides]
                X = preprocess(data, mask, adjusted_ksizes, adjusted_strides)

            except Exception as e:
                print(f"Error processing file {fn}: {e}")
                continue   
            # Save patches to LMDB
            for j in range(X.shape[0]):
                c, h, w = X.shape[1:]
                X_byte = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txt_file.write(f'{str_id} ({h},{w},{c}) source_file={fn}\n')
                txn.put(str_id.encode('ascii'), X_byte)  # Add data with key
            
            print(f'Processed file ({i+1}/{len(fns)}): {fn}')
    
    print('Database creation completed.')

def main():
    # Natural_scene 
    for folder in tqdm(source_folders["Natural_scene"], desc="Processing Natural_scene HSIs", unit="folder"):
        #folder_name = os.path.basename(os.path.dirname(folder))
        folder_name = os.path.basename(os.path.dirname(folder))
        process_files(folder, destination_folders["Natural_scene"])
    
    # Remote_sensing 
    for folder in tqdm(source_folders["Remote_sensing"], desc="Processing Remote_sensing HSIs", unit="folder"):
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(folder)))
        print(folder_name)
        process_files(folder, destination_folders["Remote_sensing"])


    print('Create Natural_scene Dataset')
    fns = os.listdir(destination_folders["Natural_scene"]) # your own data address
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    create_lmdb(
        destination_folders["Natural_scene"], fns, destination_folders_patch["Natural_scene_64"],   # your own dataset address
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],       
        load=sio.loadmat, augment=True,
    )

    print('Create Remote_sensing Dataset')
    fns = os.listdir(destination_folders["Remote_sensing"]) # your own data address
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    create_lmdb_remote_sensing(
        destination_folders["Remote_sensing"], fns, destination_folders_patch["Remote_sensing_64"],   # your own dataset address     
        scales=(1, 0.5, 0.25),        
        ksizes=(100, 64, 64),
        strides=[(100, 64, 64), (100, 32, 32), (100, 32, 32)],      
        load=sio.loadmat, augment=True,
    )



if __name__ == "__main__":
    source_folders = {
        "Natural_scene": [
            "/data/ARAD_1k/train",
            "/data/ICVL/train",
        ],
        "Remote_sensing": [
            "/data/BerlinUrGrad/minmax/train",
            "/data/Chikusei/minmax/train",
            "/data/Eagle/minmax/train",
            "/data/Houston/minmax/train",
            "/data/PaviaC/minmax/train",
            "/data/PaviaU/minmax/train",
            "/data/WDC/minmax/train",
            "/data/Xiongan/minmax/train",
        ]
    }

    destination_folders = {
        "Natural_scene": "/data/Train/Natural_scene_minmax",
        "Remote_sensing": "/data/Train/Remote_sensing_minmax"
    }

    destination_folders_patch = {
        "Natural_scene_64": "/data/Train/Natural_scene_minmax_patch_64",
        "Remote_sensing_64": "/data/Train/Remote_sensing_minmax_patch_64"
    }

    main()
