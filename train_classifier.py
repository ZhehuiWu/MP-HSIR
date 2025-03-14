import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.dataset_utils import *
from net.classifier import BackboneClassifier,FFCResNet
import numpy as np
from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision('medium')  
print(torch.cuda.is_available())  
print(torch.cuda.device_count())  

torch.set_num_threads(1)

class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        num_classes = 5
        pos_weight = torch.ones(num_classes)  
        pos_weight[1] = 3.0  
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

        self.net = FFCResNet([2, 2, 2, 2],in_channel=31, inplanes=64, size = (256,256),num_classes=5)
        # self.net = FFCResNet([2, 2, 2, 2],in_channel=100, inplanes=128, size = (256,256),num_classes=6)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx, use_conv3d=False):

        [clean_name, de_id], degrad_patch, label = batch
        
        if use_conv3d:
            degrad_patch = degrad_patch.unsqueeze(1)

        logits = self.net(degrad_patch)

        loss1 = self.loss_fn(logits, label)
        loss = loss1 

        self.log("train_loss", loss,on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    def configure_optimizers(self):
        # 配置优化器
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000*400, eta_min= 1e-6)

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
    logger = TensorBoardLogger(save_dir = "/logs/",name = "Remote_sensing_text")
    
    train_data = LMDBDataset(opt) 
    trainset = Classifier_Dataset(train_data, opt)
    
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, every_n_epochs = 1, monitor="train_loss", mode="min", save_top_k=5)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, drop_last=True, num_workers=opt.num_workers, persistent_workers=True)
    model = PromptIRModel(opt)
    trainer = pl.Trainer(precision=16, max_epochs=opt.epochs, devices=opt.num_gpus, accelerator="gpu", strategy="auto", logger=logger,callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=trainloader)#,ckpt_path = opt.ckpt_path) #ddp_find_unused_parameters_true


if __name__ == '__main__':
    main()



