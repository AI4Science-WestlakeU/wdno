from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse
import pywt
import ptwt  

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
from scipy.sparse import load_npz
import os
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

    
def tensor_to_coef(coef_tensor, shape, upsample_type = None):
    '''
    input: [N, more than 5*8, padded, padded, padded]
    upsample_type: None (base) or space (super) or time (super)
    output: list [coarse, detail]
    '''
    if upsample_type == None:
        time_indices1, time_indices2 = 0, shape[-3]
        space_indices1, space_indices2 = 0, shape[-2]
    elif upsample_type == 'time':
        time_indices1, time_indices2 = 1, shape[-3]+1
        space_indices1, space_indices2 = 0, shape[-2]
    elif upsample_type == 'space':
        time_indices1, time_indices2 = 0, shape[-3]
        space_indices1, space_indices2 = 1, shape[-2]+1
    u_Yl = coef_tensor[:, None, 0, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    u_Yh = coef_tensor[:, None, 1:8, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    v1_Yl = coef_tensor[:, None, 8, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    v1_Yh = coef_tensor[:, None, 9:16, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    v2_Yl = coef_tensor[:, None, 16, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    v2_Yh = coef_tensor[:, None, 17:24, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    c1_Yl = coef_tensor[:, None, 24, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    c1_Yh = coef_tensor[:, None, 25:32, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    c2_Yl = coef_tensor[:, None, 32, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    c2_Yh = coef_tensor[:, None, 33:40, time_indices1:time_indices2, space_indices1:space_indices2, space_indices1:space_indices2]
    Yl = torch.cat((u_Yl, v1_Yl, v2_Yl, c1_Yl, c2_Yl), dim=1).reshape(-1, shape[-3], shape[-2], shape[-1])
    Yh_ = torch.cat((u_Yh, v1_Yh, v2_Yh, c1_Yh, c2_Yh), dim=1).reshape(-1, 7, shape[-3], shape[-2], shape[-1])
    Yh = {}
    Yh["aad"] = Yh_[:, 0]
    Yh["ada"] = Yh_[:, 1]
    Yh["add"] = Yh_[:, 2]
    Yh["daa"] = Yh_[:, 3]
    Yh["dad"] = Yh_[:, 4]
    Yh["dda"] = Yh_[:, 5]
    Yh["ddd"] = Yh_[:, 6]
    return Yl, Yh


def coef_to_tensor(coef, pad = False):
    Yl, Yh = coef[0], coef[1]
    Yh = torch.stack(list(Yh.values()), dim=1)
    return torch.cat((Yl[:, None], Yh), dim=1)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    root = './data/2d/'
    dirname = 'train/'
    n_sim = 20000 
    num_t = 32
    step_t = int(256/num_t)
    num_x = 64
    step_x = int(128/num_x)
    N_downsample = 3 # less than 3
    sim_range = range(n_sim)
    start = datetime.now()
    for wave_type in ['bior1.3']:
        print('Wavelet type: ', wave_type)
        wavelet = pywt.Wavelet(wave_type)
        mode = 'zero' 
        xfm2d = DWTForward(J=1, mode=mode, wave=wave_type).to(device)
        xfm1d = DWT1DForward(J=1, mode=mode, wave=wave_type).to(device)
        wave_dir = os.path.join(root, dirname, "{}_{}/".format(wave_type, mode))
        if not os.path.exists(wave_dir):
            print('Directory not existed. Create directory: ', wave_dir)
            os.mkdir(wave_dir)
        if not os.path.exists(os.path.join(wave_dir, 'space_downsample')):
            print('Directory not existed. Create directory: ', os.path.join(wave_dir, 'space_downsample'))
            os.mkdir(os.path.join(wave_dir, 'space_downsample'))
        if not os.path.exists(os.path.join(wave_dir, 'time_downsample')):
            print('Directory not existed. Create directory: ', os.path.join(wave_dir, 'time_downsample'))
            os.mkdir(os.path.join(wave_dir, 'time_downsample'))

        max_J = pywt.dwt_max_level(num_t, wave_type)
        print('Max layer: ', max_J)
        max_coef = {}
        for i in range(5*8+2):
            max_coef[i] = 0
        for sim_id in tqdm(sim_range):
            d = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             device=device, dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             device=device, dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                            device=device, dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                            device=device, dtype=torch.float)
            s = (s[:, 1]/s.sum(-1))[:32]
            X = torch.cat((d, v, c), dim=0)[:, :32] #5, 32, 64, 64

            coef_downsample_time = {}
            init_coef_downsample_time = {}
            coef_smokeout_downsample_time = {}
            for i in range(N_downsample):
                coef_downsample_time[i] = []
                init_coef_downsample_time[i] = []
                coef_smokeout_downsample_time[i] = []

            coef_downsample_space = {}
            init_coef_downsample_space = {}
            coef_smokeout_downsample_space = {}
            for i in range(N_downsample):
                coef_downsample_space[i] = []
                init_coef_downsample_space[i] = []
                coef_smokeout_downsample_space[i] = []
            
            # downsample
            for i in range(N_downsample):
                # time downsampling
                X_sub = X[:, ::2**i]
                coef = ptwt.wavedec3(X_sub, wavelet, mode=mode, level=1)
                coef_downsample_time[i].append(coef_to_tensor(coef).cpu())

                Y_ = ptwt.waverec3(coef, wavelet)
                rec_error_Y_ = torch.norm(X_sub - Y_[:,:X_sub.shape[-3],:X_sub.shape[-2],:X_sub.shape[-1]])/torch.norm(X_sub)

                # W_u0, W_uT
                Yl0, Yh0 = xfm2d(X_sub[:, [0]])
                init_coef_downsample_time[i].append(torch.cat((Yl0, Yh0[0][:,0]), dim=1).cpu())

                #W_smokeout
                s_sub = s.reshape(1, 1, -1)[:, :, ::2**i]
                Yl_s, Yh_s = xfm1d(s_sub)
                coef_smokeout_downsample_time[i].append(torch.cat((Yl_s, Yh_s[0]), dim=1).cpu()[0])

                # space downsampling
                X_sub = X[:, :, ::2**i, ::2**i]
                coef = ptwt.wavedec3(X_sub, wavelet, mode=mode, level=1)
                coef_downsample_space[i].append(coef_to_tensor(coef).cpu())

                Y_ = ptwt.waverec3(coef, wavelet)
                rec_error_Y_ = torch.norm(X_sub - Y_[:,:X_sub.shape[-3],:X_sub.shape[-2],:X_sub.shape[-1]])/torch.norm(X_sub)

                # W_u0, W_uT
                Yl0, Yh0 = xfm2d(X_sub[:, [0]])
                init_coef_downsample_space[i].append(torch.cat((Yl0, Yh0[0][:,0]), dim=1).cpu())

                #W_smokeout
                s_sub = s.reshape(1, 1, -1)
                Yl_s, Yh_s = xfm1d(s_sub)
                coef_smokeout_downsample_space[i].append(torch.cat((Yl_s, Yh_s[0]), dim=1).cpu()[0])

            
            for j in range(5): # d, v1, v2, c1, c2
                for i in range(8): # layer
                    for k in range(N_downsample):
                        max_coef[8*j + i] = max(int(max_coef[8*j + i]), int(torch.cat(coef_downsample_time[k])[j, i].abs().max())+1)
                        max_coef[8*j + i] = max(int(max_coef[8*j + i]), int(torch.cat(coef_downsample_space[k])[j, i].abs().max())+1)
            for k in range(N_downsample):
                max_coef[40] = max(int(max_coef[40]), int(torch.cat(init_coef_downsample_time[k])[0].abs().max())+1) # only density
                max_coef[40] = max(int(max_coef[40]), int(torch.cat(init_coef_downsample_space[k])[0].abs().max())+1) # only density
                max_coef[41] = max(int(max_coef[41]), int(torch.cat(coef_smokeout_downsample_time[k]).abs().max())) # smokeout, only time downsampling

            torch.save({'coef': [torch.cat(coef_downsample_time[i]) for i in range(N_downsample)], # [5, 8, 18, 34, 34]
                    'init_coef': [torch.cat(init_coef_downsample_time[i]) for i in range(N_downsample)],
                    'smokeout': [torch.cat(coef_smokeout_downsample_time[i]) for i in range(N_downsample)],
                    'shape': [coef_downsample_time[i][0].shape[-3:] for i in range(N_downsample)],
                    # [[18, 34, 34], [10, 34, 34], [6, 34, 34]]
                    'ori_shape': X.shape[1:]}, # [32, 64, 64]
                    os.path.join(wave_dir, 'time_downsample/', "{:06d}".format(sim_id)))
            torch.save({'coef': [torch.cat(coef_downsample_space[i]) for i in range(N_downsample)], # [5, 8, 18, 34, 34]
                    'init_coef': [torch.cat(init_coef_downsample_space[i]) for i in range(N_downsample)],
                    'smokeout': [torch.cat(coef_smokeout_downsample_space[i]) for i in range(N_downsample)],
                    'shape': [coef_downsample_space[i][0].shape[-3:] for i in range(N_downsample)],
                    # [[18, 34, 34], [18, 18, 18], [18, 10, 10]]
                    'ori_shape': X.shape[1:]}, # [32, 64, 64]
                    os.path.join(wave_dir, 'space_downsample/', "{:06d}".format(sim_id)))

        end = datetime.now()
        print('Time: ', end - start)
        print('Max', list(max_coef.values()))
