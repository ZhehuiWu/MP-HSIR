import os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import init
from PIL import Image
from scipy.ndimage import zoom
from itertools import product
from scipy.linalg import qr, solve_triangular
from scipy.linalg import inv, norm

class EdgeComputation(nn.Module):
    def __init__(self, test=False):
        super(EdgeComputation, self).__init__()
        self.test = test
    def forward(self, x):
        if self.test:
            x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

            # y = torch.Tensor(x.size()).cuda()
            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, :, 1:] += x_diffx
            y[:, :, :, :-1] += x_diffx
            y[:, :, 1:, :] += x_diffy
            y[:, :, :-1, :] += x_diffy
            y = torch.sum(y, 1, keepdim=True) / 3
            y /= 4
            return y
        else:
            x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])

            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, 1:] += x_diffx
            y[:, :, :-1] += x_diffx
            y[:, 1:, :] += x_diffy
            y[:, :-1, :] += x_diffy
            y = torch.sum(y, 0) / 3
            y /= 4
            return y.unsqueeze(0)


# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64):
    if image.ndim == 2:
        h = image.shape[0]
        w = image.shape[1]
        crop_h = h % base
        crop_w = w % base
        img = image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2]
    elif image.ndim == 3:
        h = image.shape[1]
        w = image.shape[2]
        crop_h = h % base
        crop_w = w % base
        img = image[:, crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2]
    else:
        raise ValueError("image dimension should be 2 or 3")
    return img

# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
    assert len(image_size) > 1
    assert patches.shape[-3] == patches.shape[-2]
    H = image_size[0]
    W = image_size[1]
    patch_size = patches.shape[-2] - overlap
    image = np.zeros(image_size)
    idx = 0
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            image[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = patches[idx,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 :]
            idx += 1
    return image


# def data_augmentation(image, mode):
#     if mode == 0:
#         # original
#         out = image.numpy()
#     elif mode == 1:
#         # flip up and down
#         out = np.flipud(image)
#     elif mode == 2:
#         # rotate counterwise 90 degree
#         out = np.rot90(image, axes=(1, 2))
#     elif mode == 3:
#         # rotate 90 degree and flip up and down
#         out = np.rot90(image, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree
#         out = np.rot90(image, k=2, axes=(1, 2))
#     elif mode == 5:
#         # rotate 180 degree and flip
#         out = np.rot90(image, k=2, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree
#         out = np.rot90(image, k=3, axes=(1, 2))
#     elif mode == 7:
#         # rotate 270 degree and flip
#         out = np.rot90(image, k=3, axes=(1, 2))
#         out = np.flipud(out)
#     else:
#         raise Exception('Invalid choice of image transformation')
#     return out

def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :] 
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D. 
    # disable it when training 2D CNN, e.g. MemNet
    # if random.random() < 0.5:
    #     image = image[::-1, :, :] 
    
    return np.ascontiguousarray(image)


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


class BaseNormalizer:
    def __init__(self):
        assert hasattr(self, "STATEFUL"), "Missing STATEFUL class attribute"

    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_id(self):
        attributes = [self.__class__.__name__]
        attributes += [
            k[:3] + str(v)
            for k, v in self.__dict__.items()
            if not isinstance(v, torch.Tensor)
        ]
        return "_".join(attributes).replace(".", "")

    def __repr__(self):
        return self.get_id()

    def filename(self):
        return f"{self.get_id()}.pth"

    def save(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        torch.save(self.__dict__, filename)

    def load(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        state = torch.load(filename)
        for k, v in state.items():
            setattr(self, k, v)



class BandMinMaxQuantileStateful(BaseNormalizer):
    STATEFUL = True

    def __init__(self, low=0.02, up=0.98, epsilon=0.001):
        super().__init__()
        self.low = low
        self.up = up
        self.epsilon = epsilon

    def fit(self, imgs, masks=None):
        x_train = []
        for i, img in enumerate(imgs):
            if masks is not None and masks[i] is not None:

                mask = masks[i]
                valid_pixels = img[:, ~mask]  
            else:

                valid_pixels = img

            if valid_pixels.numel() > 0: 
                x_train.append(valid_pixels.flatten(start_dim=1))
        
        if len(x_train) > 0:
   
            x_train = torch.cat(x_train, dim=1)
            bands = x_train.shape[0]
            q_global = np.zeros((bands, 2))
            for b in range(bands):
                q_global[b] = np.percentile(
                    x_train[b].cpu().numpy(), q=100 * np.array([self.low, self.up])
                )

     
            self.q = torch.tensor(q_global, dtype=torch.float32).T[..., None, None]

    def transform(self, x):
   
        x = torch.minimum(x, self.q[1])
        x = torch.maximum(x, self.q[0])
        return (x - self.q[0]) / (self.epsilon + (self.q[1] - self.q[0]))

def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def rand_crop(img, cropx, cropy):
    _,y,x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1:y1+cropy, x1:x1+cropx]


def Data2Volume(data, mask, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data, only keep patches with valid pixels.
    """
    dshape = data.shape

    valid_patches = []

    args = [range(0, dshape[i] - ksizes[i] + 1, strides[i]) for i in range(len(ksizes))]
    for s in product(*args):

        slices_data = tuple(slice(s[i], s[i] + ksizes[i]) for i in range(len(ksizes)))

        slices_mask = (slice(s[1], s[1] + ksizes[1]), slice(s[2], s[2] + ksizes[2]))

        patch_data = data[slices_data]
        patch_mask = mask[slices_mask] if mask is not None else np.zeros(patch_data.shape[1:], dtype=bool)
                

        if not np.any(patch_mask):  
 
            p_min = np.min(patch_data)
            p_max = np.max(patch_data)
            patch_data = (patch_data - p_min) / (p_max - p_min)
            
            valid_patches.append(patch_data)

    if len(valid_patches) > 0:
        V = np.stack(valid_patches)
    else:
        V = np.zeros((0,) + tuple(ksizes))  
    
    return V

def givens(x, y):
    if y == 0:
        c = 1
        s = 0
    else:
        if abs(y) >= abs(x):
            cotangent = x / y
            s = 1 / np.sqrt(1 + cotangent ** 2)
            c = s * cotangent
        else:
            tangent = y / x
            c = 1 / np.sqrt(1 + tangent ** 2)
            s = c * tangent

    return np.array([[c, s], [-s, c]])


def QR_rank(A, f, k):
    if f < 1:
        print('Parameter f given is less than 1. Automatically set f = 2')
        f = 2

    m, n = A.shape
    k = min(k, m, n)
    Q, R, p = qr(A, mode='economic', pivoting=True)

    if k == n:
        return Q, R, p

    ss = np.sign(np.diag(R)) if R.shape[0] > 1 and R.shape[1] > 1 else np.sign(R[0, 0])
    R = R * ss[:, np.newaxis]
    Q = Q * ss[np.newaxis, :]
    AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)
    gamma = np.sqrt(np.sum(R[k:, k:] ** 2, axis=0)) if k < R.shape[0] else np.zeros(n - k)
    tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = np.sqrt(np.sum(tmp ** 2, axis=1)) ** -1

    Rm = R.shape[0]
    while True:
        tmp = (1. / omega[:, np.newaxis] * gamma[np.newaxis, :]) ** 2 + AB ** 2
        # i, j = np.unravel_index(np.argmax(tmp.T > f * f), tmp.shape)
        indices = np.where(tmp.T > f * f)
        if len(indices[0]) == 0:  # 或者你可以用 if indices[0].size == 0
            break

        j, i = indices[0][0], indices[1][0]
        if j > 0:
            AB[:, [0, j]] = AB[:, [j, 0]]
            gamma[[0, j]] = gamma[[j, 0]]
            R[:, [k, k + j]] = R[:, [k + j, k]]
            p[[k, k + j]] = p[[k + j, k]]

        if i < k:
            p[i:k] = np.roll(p[i:k], shift=-1)
            R[:, i:k] = np.roll(R[:, i:k], shift=-1, axis=1)
            omega[i:k] = np.roll(omega[i:k], shift=-1)
            AB[i:k, :] = np.roll(AB[i:k, :], shift=-1, axis=0)
            for ii in range(i, k - 1):
                G = givens(R[ii, ii], R[ii + 1, ii])
                if np.dot(G[0, :], [R[ii, ii], R[ii + 1, ii]]) < 0:
                    G = -G
                R[ii:ii + 2, :] = np.dot(G, R[ii:ii + 2, :])
                Q[:, ii:ii + 2] = np.dot(Q[:, ii:ii + 2], G.T)
            if k - 1 < R.shape[0] and R[k - 1, k - 1] < 0:
                R[k - 1, :] = -R[k - 1, :]
                Q[:, k - 1] = -Q[:, k - 1]


        if k < Rm:
            for ii in range(k + 1, Rm):
                G = givens(R[k, k], R[ii, k])
                if np.dot(G[0, :], [R[k, k], R[ii, k]]) < 0:
                    G = -G
                R[[k, ii], :] = np.dot(G, R[[k, ii], :])
                Q[:, [k, ii]] = np.dot(Q[:, [k, ii]], G.T)

        p[[k - 1, k]] = p[[k, k - 1]]
        ga = R[k - 1, k - 1].copy()
        mu = R[k - 1, k] / ga
        nu = R[k, k] / ga if k < Rm else 0
        rho = np.sqrt(mu ** 2 + nu ** 2)
        ga_bar = ga * rho
        b1 = R[:k - 1, k - 1].copy()
        b2 = R[:k - 1, k].copy()
        c1T = R[k - 1, k + 1:].copy()
        c2T = R[k, k + 1:] if k < Rm else np.zeros(len(c1T))
        c1T_bar = (mu * c1T + nu * c2T) / rho
        c2T_bar = (nu * c1T - mu * c2T) / rho
        
        R[:k - 1, k - 1] = b2
        R[:k - 1, k] = b1
        R[k - 1, k - 1] = ga_bar
        R[k - 1, k] = ga * mu / rho

        if k >= R.shape[0]:
    
            extra_rows = k + 1 - R.shape[0]
            R = np.vstack([R, np.zeros((extra_rows, R.shape[1]))])
        R[k, k] = ga * nu / rho
        R[k - 1, k + 1:] = c1T_bar
        R[k, k + 1:] = c2T_bar

        u = solve_triangular(R[:k - 1, :k - 1], b1, lower=False, trans=False)
        u1 = AB[:k - 1, 0].copy()
        
        AB[:k - 1, 0] = (nu ** 2 * u - mu * u1) / rho ** 2
        AB[k - 1, 0] = mu / rho ** 2
        AB[k - 1, 1:] = c1T_bar / ga_bar

        AB[:k - 1, 1:] = AB[:k - 1, 1:] + (nu * u[:, np.newaxis] * c2T_bar[np.newaxis, :] - u1[:, np.newaxis] * c1T_bar[np.newaxis, :]) / ga_bar

        gamma[0] = ga * nu / rho
        gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2 - c2T ** 2))

        u_bar = u1 + mu * u

        omega[k - 1] = ga_bar
        omega[:k - 1] = 1 / np.sqrt(1 / omega[:k - 1] ** 2 + u_bar ** 2 / ga_bar ** 2 - u ** 2 / ga ** 2)

        Gk = np.array([[mu / rho, nu / rho], [nu / rho, -mu / rho]])
        if k < Rm:
            Q[:, [k - 1, k]] = np.dot(Q[:, [k - 1, k]], Gk.T)

    return Q[:, :k], R[:k, :], p

def LS_rank(data, rank):
    C, H, W = data.shape[-3], data.shape[-2], data.shape[-1]
    band_indices = np.linspace(0, C - 1, rank, dtype=int)
    rank_img = np.take(data, band_indices, axis=0).reshape(rank, H*W)
    t1 = np.matmul(rank_img, rank_img.transpose(1, 0))  
    t2 = np.matmul(data.reshape(C, H*W), rank_img.transpose(1, 0)) 
    E = np.matmul(t2, np.linalg.inv(t1))#C*K
    E = E.reshape(C, rank)
    A = rank_img.reshape(rank, H, W)
    
    return A, E

def svd_rank(data, rank):
    C, H, W = data.shape[-3], data.shape[-2], data.shape[-1]
    data = data.reshape(C, H * W)
    U,sigma,Vt = np.linalg.svd(data,full_matrices=False)
    E = U[:, 0:rank]                        
    A = np.matmul(E.T,data)               
    A = A.reshape(rank,H,W)#CHW

    return A, E

def interpolate_bands(original_data, target_bands):
    original_data = original_data.transpose(1,2,0)
    original_bands = original_data.shape[2]

    original_indices = np.round(np.linspace(0, target_bands - 1, original_bands)).astype(int)

    interpolated_data = np.zeros((original_data.shape[0], original_data.shape[1], target_bands))
    interpolated_data[..., original_indices] = original_data  

    for i in range(len(original_indices) - 1):
        left_band = original_data[..., i]
        right_band = original_data[..., i + 1]

        start, end = original_indices[i], original_indices[i + 1]
        interp_positions = np.linspace(0, 1, end - start + 1)[1:-1]
        
        for j, pos in enumerate(interp_positions, start=start + 1):

            interpolated_data[..., j] = left_band * (1 - pos) + right_band * pos

    interpolated_data = interpolated_data.transpose(2, 0, 1).astype(np.float32)

    return interpolated_data, original_indices