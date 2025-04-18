import scipy.io
import numpy as np
import math
import h5py
import pickle
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse
from wave_trans import tensor_to_coef, coef_to_tensor
from ddpm_burgers.wavelet_utils import upsample_coef
from ddpm_burgers.model_utils import cycle
from typing import Tuple
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from IPython import embed

def get_wavelet_super_preprocess(
    rescaler=70, 
    is_super_model=False,
    N_downsample=0,
    mode='zero',
    wave_type='bior2.4',
    is_condition_u0=True,
    is_condition_uT=True,
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')

    def preprocess(db):
        data = db['coef']
        nonlocal N_downsample
        N_downsample = 0 if not is_super_model else N_downsample
        w_u = data[N_downsample][:, 0][:40000]
        w_f = data[N_downsample][:, 1][:40000]
        if is_super_model:
            w_u_sub = data[N_downsample+1][:, 0][:40000]
            w_f_sub = data[N_downsample+1][:, 1][:40000]
        ori_shape = list(db['ori_shape'])
        ori_shape[0] = math.ceil(ori_shape[0]/2**N_downsample)
        ori_shape[1] = math.ceil(ori_shape[1]/2**N_downsample)
            
        N = w_u.size(0)
        nt = w_f.size(-2)
        nx = w_f.size(-1)
        shape = w_f.shape[2:]

        # pad f for stack 
        pad_t, pad_x = int(64 / 2**N_downsample), int(64 / 2**N_downsample)
        
        w_uf = torch.cat((w_u, w_f), dim=1) # assuming dim 0 is N_samples e.g. [40000, 8, 41, 60]
        w_uf = nn.functional.pad(w_uf, (0, pad_x - nx, 0, pad_t - nt), 'constant', 0)
        data = w_uf # [40000, 8, 64, 64]

        if is_super_model: # low-resolution
            w_uf_sub = torch.cat((upsample_coef(w_u_sub, shape), upsample_coef(w_f_sub, shape)), dim=1) # e.g. [40000, 8, 21*2, 60]
            w_uf_sub = nn.functional.pad(w_uf_sub, (0, pad_x - w_uf_sub.shape[-1], 0, pad_t - w_uf_sub.shape[-2]), 'constant', 0)
            data = w_uf
            data[:, :, nt, :] = data[:, :, nt-1, :] # repeat the last timestep due to the odd number of timesteps
            # data = w_uf - w_uf_sub
            data = torch.cat((data, w_uf_sub), dim=1)
        
        if is_condition_u0 or is_condition_uT: # 1d wavelet transformation pf u0, uT
            ifm = DWTInverse(mode=mode, wave=wave_type).to(data.device)
            Yl, Yh = tensor_to_coef(w_uf[:, :8], shape)
            u_f = ifm((Yl, Yh))[:, :, :ori_shape[-2], :ori_shape[-1]]
            u, f = u_f[:, 0], u_f[:, 1, :ori_shape[-2]-1]

            # concatenate W_u0, W_uT
            xfm1d = DWT1DForward(J=1, mode=mode, wave=wave_type).to(data.device)
            Yl0, Yh0 = xfm1d(u[:, [0, -1], :ori_shape[-1]])
            W_condition = torch.zeros_like(data[:, [0]]) # (N, 1, (W_u0)+(W_uT), pad_x)
            n_repeat = int(pad_t / 4)
            if is_condition_u0:
                W_condition[:, :, :n_repeat, :nx] = Yl0[:, [0]].unsqueeze(1).expand(N, 1, n_repeat, nx)
                W_condition[:, :, n_repeat:n_repeat*2, :nx] = Yh0[0][:, [0]].unsqueeze(1).expand(N, 1, n_repeat, nx)
            if is_condition_uT:
                W_condition[:, :, n_repeat*2:n_repeat*3, :nx] = Yl0[:, [1]].unsqueeze(1).expand(N, 1, n_repeat, nx)
                W_condition[:, :, n_repeat*3:n_repeat*4, :nx] = Yh0[0][:, [1]].unsqueeze(1).expand(N, 1, n_repeat, nx)
            data = torch.cat((data, W_condition), dim=1) 

        data = data / rescaler
        return data, list(shape), list(ori_shape)

    return preprocess


def get_wavelet_preprocess(
    rescaler=70, 
    mode='zero',
    wave_type='bior2.4',
    is_condition_u0=True,
    is_condition_uT=True
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')
    
    def preprocess(db):
        '''
        '''
        
        shape = db['shape']
        ori_shape = db['ori_shape']
        data = db['coef']
        w_u = data[:, 0][:40000]
        w_f = data[:, 1][:40000]

        N = w_u.size(0)
        nt = w_f.size(-2)
        nx = w_f.size(-1)

        w_u = w_u.reshape(N, -1, nt, nx)
        w_f = w_f.reshape(N, -1, nt, nx)

        # pad f for stack 
        w_f = nn.functional.pad(w_f, (0, 64 - nx, 0, 64 - nt), 'constant', 0)
        w_u = nn.functional.pad(w_u, (0, 64 - nx, 0, 64 - nt), 'constant', 0)
        
        data = torch.cat((w_u, w_f), dim=1) # assuming dim 0 is N_samples
        if is_condition_u0 or is_condition_uT:
            ifm = DWTInverse(mode=mode, wave=wave_type).to(data.device)
            J = len(shape)

            Yl, Yh = tensor_to_coef(data, shape)
            u_f = ifm((Yl, Yh))[:, :, :ori_shape[-2], :ori_shape[-1]]
            u, f = u_f[:, 0], u_f[:, 1, :ori_shape[-2]-1]
            u = nn.functional.pad(u, (0, 128 - ori_shape[-1]), 'constant', 0)

            # concatenate W_u0, W_uT
            xfm1d = DWT1DForward(J=len(shape), mode=mode, wave=wave_type).to(data.device)
            ifm1d = DWT1DInverse(mode=mode, wave=wave_type).to(data.device)
            Yl0, Yh0 = xfm1d(u[:, [0, -1], :ori_shape[-1]])
            W_condition = torch.zeros_like(data[:, [0]]) # (N, 1, 32(W_u0)+32(W_uT), 64)
            assert 32 % (J+1) == 0
            n_repeat = int(32 / (J+1))
            if is_condition_u0:
                W_condition[:, :, :n_repeat, :nx] = Yl0[:, [0]].unsqueeze(-1).expand(N, n_repeat, shape[-1][-1], 2**(J-1)).reshape(N, 1, n_repeat, nx)
                for j in range(J):
                    W_condition[:, :, n_repeat*(j+1):n_repeat*(j+2), :nx] = Yh0[j][:, [0]].unsqueeze(-1)\
                                .expand(N, n_repeat, shape[j][-1], 2**j).reshape(N, 1, n_repeat, nx)
            if is_condition_uT:
                W_condition[:, :, 32:32+n_repeat, :nx] = Yl0[:, [1]].unsqueeze(-1).expand(N, n_repeat, shape[-1][-1], 2**(J-1)).reshape(N, 1, n_repeat, nx)
                for j in range(J):
                    W_condition[:, :, 32+n_repeat*(j+1):32+n_repeat*(j+2), :nx] = Yh0[j][:, [1]].unsqueeze(-1)\
                                .expand(N, n_repeat, shape[j][-1], 2**j).reshape(N, 1, n_repeat, nx)
            data = torch.cat((data, W_condition), dim=1) 

            # concatenate u0, uT
            if is_condition_u0:
                u_condition = u[:, 0].reshape(-1, 1, 2, 1, 64).expand(N, 1, 2, 16, 64).reshape(-1, 1, 32, 64)
            else:
                u_condition = 0 * u[:, 0].reshape(-1, 1, 2, 1, 64).expand(N, 1, 2, 16, 64).reshape(-1, 1, 32, 64)
            if is_condition_uT:
                uT = u[:, ori_shape[-2]-1].reshape(-1, 1, 2, 1, 64).expand(N, 1, 2, 16, 64).reshape(-1, 1, 32, 64)
            else:
                uT = 0 * u[:, ori_shape[-2]-1].reshape(-1, 1, 2, 1, 64).expand(N, 1, 2, 16, 64).reshape(-1, 1, 32, 64)
            u_condition = torch.cat((u_condition, uT), dim=2) # (N, 1, 16(u0[:64])+16(u0[64:])+16(uT[:64])+16(uT[64:]), 64)
            
            data = torch.cat((data, u_condition), dim=1)


        data = data / rescaler
        return data, [list(shape[i]) for i in range(len(shape))], list(ori_shape)

    return preprocess

def get_burgers_preprocess(
    rescaler=10, 
    is_super_model_train = False,
    N_downsample = 0,
    is_super_model_test = False,
    upsample_t = 0,
    upsample_x = 0,
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')
    
    def preprocess(db):
        '''We are only returning f and u for now, in the shape of 
        (u0, u1, ..., f0, f1, ...)
        '''
        
        # db should be a tensor: u: (N_samples, 81, 120); f: (N_samples, 80, 120)
        # if super resolution data: u: (N_samples, 161, 240); f: (N_samples, 161, 240)
        if is_super_model_test:
            super_nt, super_nx = db['f'].shape[-2], db['u'].shape[-1]
            assert super_nt/80/2**upsample_t > 0
            u = db['u'][:, ::int(super_nt/80/2**upsample_t), ::int(super_nx/120/2**upsample_x)]
            f = db['f'][:, ::int(super_nt/80/2**upsample_t), ::int(super_nx/120/2**upsample_x)]
        else:
            u = db['u'][:40000]
            f = db['f'][:40000]
        
        # pad f for stack
        nt = f.size(-2) # 80
        nx = f.size(-1) # 120
        shape = u[..., ::2**N_downsample, ::2**N_downsample].shape[-2:] # [81, 120]
        f = nn.functional.pad(f, (0, 128*2**upsample_x - nx, 0, 128*2**upsample_t - nt), 'constant', 0)
        u = nn.functional.pad(u, (0, 128*2**upsample_x - nx, 0, 128*2**upsample_t - 1 - nt), 'constant', 0)
        
        data = torch.stack((u, f), dim=1) # assuming dim 0 is N_samples
        if is_super_model_train:
            uf = data[:, :, ::2**N_downsample, ::2**N_downsample]
            uf_sub = upsample_coef(data[:, :, ::2**(N_downsample+1), ::2**(N_downsample+1)], shape)
            nt_sub = int(nt / 2**N_downsample)
            uf[:, :, nt_sub+1, :] = uf[:, :, nt_sub, :] # repeat the last timestep due to the odd number of timesteps
            data = torch.cat((uf, uf_sub), dim=1)
    
        data = data / rescaler
        return data, list(shape), list(shape)

    return preprocess


class DiffusionDataset(Dataset):
    def __init__(
        self, 
        fname, 
        preprocess=get_wavelet_preprocess(),  
    ):
        '''
        Arguments:

        '''
        self.db = torch.load(fname)
        self.x, self.shape, self.ori_shape = preprocess(self.db)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]

    def get(self, idx):
        return self.__getitem__(idx)
    
    def len(self):
        return self.__len__()


class SuperDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1):
        self.dataset = dataset # a list of datasets
        self.dl = [cycle(DataLoader(dataset[i], batch_size = batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)) for i in range(len(dataset))]
        self.num_batches = len(dataset) * ((len(dataset[0]) + batch_size - 1) // batch_size)

    def __iter__(self):
        group_id = random.randint(0, len(self.dl)-1)
        yield next(self.dl[group_id])

    def __len__(self):
        return self.num_batches