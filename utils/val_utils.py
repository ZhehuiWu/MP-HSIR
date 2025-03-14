import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe
import torch

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    C = clean.shape[-3]
    psnr = 0
    ssim = 0
    for i in range(recoverd.shape[0]):
        psnr_ch = []
        ssim_ch = []
        for ch in range(C):
            x = recoverd[i,ch,:,:]
            y = clean[i,ch,:,:]
            psnr_temp = peak_signal_noise_ratio(x, y, data_range=1)
            ssim_temp = structural_similarity(x, y, data_range=1)
            psnr_ch.append(psnr_temp)
            ssim_ch.append(ssim_temp)
        psnr += np.mean(psnr_ch)
        ssim += np.mean(ssim_ch)    
    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def compute_psnr_ssim2(recoverd, clean, degrad_patch=None):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    
    if degrad_patch is not None:
        degrad_patch = degrad_patch.detach().cpu().numpy()
    
    C = clean.shape[-3]
    psnr = 0
    ssim = 0
    count = 0
    
    for i in range(recoverd.shape[0]):
        psnr_ch = []
        ssim_ch = []
        for ch in range(C):
            if degrad_patch is not None and not np.all(degrad_patch[i, ch, :, :] == 0):
                continue  
            
            x = recoverd[i, ch, :, :]
            y = clean[i, ch, :, :]
            psnr_temp = peak_signal_noise_ratio(x, y, data_range=1)
            ssim_temp = structural_similarity(x, y, data_range=1)
            psnr_ch.append(psnr_temp)
            ssim_ch.append(ssim_temp)
        
        if psnr_ch:  
            psnr += np.mean(psnr_ch)
            ssim += np.mean(ssim_ch)
            count += 1
    
    return (psnr / count if count > 0 else 0, 
            ssim / count if count > 0 else 0, 
            count)

def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0