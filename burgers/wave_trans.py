import pywt
from pywt import wavedec, waverec
from pytorch_wavelets import DWT1DForward, DWT1DInverse 
from pytorch_wavelets import DWTForward, DWTInverse 

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import numpy as np  
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from ddpm_burgers.wavelet_utils import upsample_coef
import matplotlib.pyplot as plt


def tensor_to_coef_super(coef_tensor, shape):
    '''
    input: [N, more than 2*(1+3), padded, padded]
    '''
    u_Yl = coef_tensor[:, None, 0, :shape[-2], :shape[-1]]
    u_Yh = coef_tensor[:, None, 1:4, :shape[-2], :shape[-1]]
    f_Yl = coef_tensor[:, None, 4, :shape[-2], :shape[-1]]
    f_Yh = coef_tensor[:, None, 5:8, :shape[-2], :shape[-1]]
    Yl = torch.cat((u_Yl, f_Yl), dim=1)
    Yh = [torch.cat((u_Yh, f_Yh), dim=1)]
    return Yl, Yh

def tensor_to_coef(coef_tensor, shape):
    '''
    input: [N, more than 2*(1+3), padded, padded]
    '''
    u_Yl = coef_tensor[:, None, 0, :shape[-2], :shape[-1]]
    u_Yh = coef_tensor[:, None, 1:4, :shape[-2], :shape[-1]]
    f_Yl = coef_tensor[:, None, 4, :shape[-2], :shape[-1]]
    f_Yh = coef_tensor[:, None, 5:8, :shape[-2], :shape[-1]]
    Yl = torch.cat((u_Yl, f_Yl), dim=1)
    Yh = [torch.cat((u_Yh, f_Yh), dim=1)]
    return Yl, Yh

    
def coef_to_tensor(Yl, Yh, pad = False):
    '''
    return: repeat Yh[i] 2**i times 
    if pad: [Yl.shape[0], Yl.shape[1], 1+3*J, 64, 64]
    else: [Yl.shape[0], Yl.shape[1], 1+3*J, Yh[0].shape[-2]+2**(J-1)-1 (because Yh[0].shape[-2]%2=1), Yh[0].shape[-1] (because Yh[0].shape[-1]%2=0)]
    '''
    J = len(Yh)
    coef_tensor = torch.zeros(Yl.shape[0], Yl.shape[1], 1+3*J, Yh[0].shape[-2]+2**(J-1)-1, Yh[0].shape[-1], device=Yl.device)
    Yl_repeat = Yl.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, 1, 2**(J-1), 1, 2**(J-1))\
                .reshape(Yl.shape[0], Yl.shape[1], 2**(J-1)*Yl.shape[2], 2**(J-1)*Yl.shape[3]).clone()
    coef_tensor[:, :, 0] = Yl_repeat
    for i in range(J):
        Yh_repeat = Yh[i].unsqueeze(-2).unsqueeze(-1).repeat(1, 1, 1, 1, 2**i, 1, 2**i)\
            .reshape(Yh[i].shape[0], Yh[i].shape[1], Yh[i].shape[2], 2**i*Yh[i].shape[3], 2**i*Yh[i].shape[4]).clone()
        coef_tensor[:, :, 1+3*i:1+3*(i+1)] = torch.cat((Yh_repeat, Yh_repeat[:, :, :, [-1]].repeat(1, 1, 1, 2**(J-1)-2**i, 1)), dim=3)
    if pad:
        upsample_t = int(coef_tensor.shape[-2] / 40)
        upsample_x = int(coef_tensor.shape[-1] / 60)
        coef_tensor = nn.functional.pad(coef_tensor, (0, 64*upsample_x - coef_tensor.shape[-1], 0, 64*upsample_t - coef_tensor.shape[-2]), 'constant', 0)
    return coef_tensor



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = '1d'
    all_data = torch.load(f'data/{dataset}/train')
    u_data, f_data = all_data['u'], all_data['f']
    Ndata = u_data.shape[0]
    num_t = 80
    num_x = 120
    f_data = torch.cat((f_data, torch.zeros(Ndata, 1, f_data.shape[-1])), dim=1)

    data = torch.cat((u_data.unsqueeze(1), f_data.unsqueeze(1)), dim=1) # [N, 2, nt, nx]
    batch_size = 20000
    N_downsample = 4 # less than 5: 120%(2**4) != 0

    dataloader = DataLoader(TensorDataset(data), batch_size = batch_size, shuffle = False)

    for wave_type in ['bior2.4']:
        wavelet = pywt.Wavelet(wave_type)
        mode = 'periodization' 

        max_J = pywt.dwt_max_level(num_t, wave_type)
        coef_list = []
        if max_J>=0:
            print('Max layer: ', max_J)

            # 2D
            start = datetime.now()
            xfm = DWTForward(J=max_J, mode=mode, wave=wave_type).to(device)  # Accepts all wave types available to PyWavelets
            ifm = DWTInverse(mode=mode, wave=wave_type).to(device)
            xfms = {}
            for i in range(1, max_J):
                xfms[i] = DWTForward(J=i, mode=mode, wave=wave_type).to(device)  # Accepts all wave types available to PyWavelets
            coef_downsample = {}
            for i in range(N_downsample):
                coef_downsample[i] = []
            for n, (X,) in enumerate(dataloader):
                X = X.to(device)
                # downsample
                for i in range(N_downsample):
                    X_sub = X[:, :, ::2**i, ::2**i]
                    Yl_, Yh_ = xfms[1](X_sub)
                    coef_downsample[i].append(coef_to_tensor(Yl_, Yh_).cpu())
                    Y_ = ifm((Yl_, Yh_))
                    Y_[:,1,-1] = 0
                    rec_error_Y_ = torch.norm(X_sub - Y_[:,:,:X_sub.shape[-2],:X_sub.shape[-1]])/torch.norm(X_sub)
                    # print(Yl_.shape, rec_error_Y_)

            print('coef_downsample shape: ', [torch.cat(coef_downsample[i]).shape for i in range(N_downsample)])
            print('coef_downsample max:')
            for k in range(N_downsample):
                print(f'N_downsample: {k}')
                print([torch.cat(coef_downsample[k])[:,0,i].abs().max() for i in range(4)])
                print([torch.cat(coef_downsample[k])[:,1,i].abs().max() for i in range(4)])
            torch.save({'coef': [torch.cat(coef_downsample[i]) for i in range(N_downsample)],
                        'shape': [coef_downsample[i][0].shape[2:] for i in range(N_downsample)],
                        'ori_shape': X.shape[2:]}, 
                        'data/{}/coef_{}_{}_super'.format(dataset, wave_type, mode))
            print('Save data/{}/coef_{}_{}_super'.format(dataset, wave_type, mode))
        
        end = datetime.now()
        print('Time: ', end - start)


