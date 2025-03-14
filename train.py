import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from utils.dataset_utils import *
from net.MP_HSIR import MP_HSIR_Net
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
import os
import time
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler
import math
import open_clip

torch.set_float32_matmul_precision('medium')  
print(torch.cuda.is_available())  
print(torch.cuda.device_count()) 

torch.set_num_threads(1)


class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.loss_fn = nn.L1Loss()

        # self.net = MP_HSIR_Net(in_channel=31,out_channel=31,dim=64,task_classes=6)
        self.net = MP_HSIR_Net(in_channel=100,out_channel=100,dim=96,task_classes=7)

    def forward(self, x1,x2):
        return self.net(x1,x2)

    def training_step(self, batch, batch_idx, use_conv3d=False):

        [clean_name, de_type], degrad_patch, clean_patch, prompt = batch

        if use_conv3d:
            degrad_patch = degrad_patch.unsqueeze(1)
            clean_patch = clean_patch.unsqueeze(1)

        restored = self.net(degrad_patch,prompt)
        restored = torch.clamp(restored, 0, 1)  

        loss1 = self.loss_fn(restored, clean_patch)
    
        loss = loss1

        self.log("train_loss", loss,on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=int(0.1 * self.args.epochs),  
            max_epochs=self.args.epochs,  
            eta_min=1e-6  
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    print("Options")
    print(opt)
    set_seed(opt.seed)
    logger = TensorBoardLogger(save_dir = "/home/wuzhehui/Hyper_Restoration/PromptIR-main/logs/",name = "Remote_sensing_text")##Your own file path
    
    train_data = LMDBDataset(opt) 
    trainset = ImageTransformDataset(train_data, opt)

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, every_n_epochs = 50, save_top_k=-1)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, pin_memory=True,  num_workers=opt.num_workers, persistent_workers=True)

    model = PromptIRModel(opt)
    if opt.ckpt_path is not None:
        checkpoint = torch.load(opt.ckpt_path,map_location=torch.device(f"cuda:{opt.num_gpus[0]}"))
        state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(filtered_dict)
        model.load_state_dict(model_state_dict, strict=False)

    trainer = pl.Trainer(precision="16-mixed", max_epochs=opt.epochs, devices=opt.num_gpus, accelerator="gpu", strategy="auto", logger=logger,callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=trainloader) #ddp_find_unused_parameters_true



if __name__ == '__main__':
    main()



