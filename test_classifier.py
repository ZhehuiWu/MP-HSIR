import subprocess
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from utils.dataset_utils import *
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.classifier import BackboneClassifier,FFCResNet
import numpy as np
from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import accuracy_score


torch.set_float32_matmul_precision('medium')  
print(torch.cuda.is_available())  
print(torch.cuda.device_count())  

torch.set_num_threads(1)



class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.net = FFCResNet([2, 2, 2, 2],in_channel=31,size = (256,256),num_classes=8)
        self.net = FFCResNet([2, 2, 2, 2],in_channel=31, inplanes=64, size = (256,256),num_classes=5)

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

        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,  
            gamma=1      
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

def Precision(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    num_classes = labels.shape[1]

    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    
    for i in range(labels.shape[0]):
        for j in range(num_classes):
           
            if labels[i, j] == 1 and preds[i, j] == 1:
                true_positives[j] += 1

            elif labels[i, j] == 0 and preds[i, j] == 1:
                false_positives[j] += 1

    precision_per_class = true_positives / (true_positives + false_positives)

    for i, precision in enumerate(precision_per_class):
        print(f"Class {i+1} Precision: {precision:.2f}")

def Accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    num_classes = labels.shape[1]
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    for i in range(labels.shape[0]):
        for j in range(num_classes):

            if labels[i, j] == preds[i, j]:
                correct_per_class[j] += 1

            total_per_class[j] += 1 

    auc_per_class = correct_per_class / total_per_class

    for i, auc in enumerate(auc_per_class):
        print(f"Class {i+1} Auc: {auc:.2f}")

def test_classifier(net, dataset, device = None):

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    all_preds = []
    all_lables = []
    with torch.no_grad():
        for ([clean_name,_], degrad_patch, label) in tqdm(testloader):

            degrad_patch = degrad_patch.to(device)

            logits = net(degrad_patch)
            preds = (logits.sigmoid() > 0.5).long()
            preds_np = preds.cpu().numpy().astype(int)  
            label_np = label.cpu().numpy().astype(int)  
            all_preds.append(preds_np[0])
            all_lables.append(label_np[0])

        batch_accuracy = accuracy_score(all_preds, all_lables)
        print(f'Batch Accuracy: {batch_accuracy}')
        Accuracy(all_preds,all_lables)
        Precision(all_preds,all_lables)



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

    parser.add_argument('--test_dir', type=str, default='/home/wuzhehui/Hyper_Restoration/PromptIR-main/data/Test/ICVL',
                        help='where clean HSIs of test saves.') 
    parser.add_argument('--degrad_id', type=int, default=4, help='')
    parser.add_argument('--degrad_range', type=list, default=[(4, ),(30, 30)], help='')
    parser.add_argument('--repeat', type=int, default=10, help='')
    parser.add_argument('--select_bands', type=list, default=[28,15,9], help='The bands used to compose the pseudo-color image')
    parser.add_argument('--output_path', type=str, default="/home/wuzhehui/Hyper_Restoration/PromptIR-main/output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="/home/wuzhehui/Hyper_Restoration/PromptIR-main/checkpoint/1076/epoch=962-step=257121.ckpt", help='checkpoint save path')
    parser.add_argument('--rank', type=int, default=31, help='When the number of bands is less than 100, it is 20; when it is greater than or equal to 100, it is 80.')
    testopt = parser.parse_args()

    test_data = LMDBDataset(opt) 
    print(len(test_data))
    testset = Classifier_Dataset(test_data, opt)
    
    set_seed(testopt.seed)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    ckpt_path = testopt.ckpt_path
    net  = PromptIRModel(testopt).load_from_checkpoint(ckpt_path, args=testopt, map_location=device).to(device)
    net.eval()
  
    print("CKPT name : {}".format(ckpt_path))
    test_classifier(net, testset,  device = device)



