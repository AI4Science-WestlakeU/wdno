import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.sparse import load_npz
import pdb
import sys, os
import math
import random
from ddpm.wave_utils import upsample_coef
from ddpm.utils import cycle

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


class Smoke(Dataset):
    def __init__(
        self,
        # dataset,
        dataset_path,
        time_steps=256,
        steps=32,
        all_size=128,
        size=64,
        is_train=True,
        test_mode='control', # control / simulation
        upsample=False,
    ):
        super().__init__()
        self.root = dataset_path
        self.steps = steps
        self.time_steps = time_steps
        self.time_interval = int(time_steps/steps)
        self.all_size = all_size
        self.size = size
        self.space_interval = int(all_size/size)
        self.is_train = is_train
        self.test_mode = test_mode
        self.dirname = "train" if self.is_train else "test"
        self.sub_dirname = "control" if test_mode == 'control' else "simulation" 
        if self.is_train:
            self.n_simu = 20000
        elif test_mode == 'control':
            self.n_simu = 50 
        elif test_mode == 'simulation' and upsample:
            self.n_simu = 100 
        elif test_mode == 'simulation' and not upsample:
            self.n_simu = 2000
        else: 
            raise 
        self.RESCALER = torch.tensor([3, 20, 20, 17, 19, 1]).reshape(1, 6, 1, 1) 

    def __len__(self):
        return self.n_simu

    def __getitem__(self, sim_id):
        if self.is_train:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float) # 33, 8
            s = s[:, 1]/s.sum(-1) # 33
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) # 33, 64, 64
            state = torch.cat((d, v, c, s), dim=0)[:, :32] # 6, 32, 64, 64
        
            data = (
                state.permute(1, 0, 2, 3) / self.RESCALER, # 32, 6, 64, 64
                list(state.shape[-3:]),
                list(state.shape[-3:]),
                sim_id,
            )
        elif self.test_mode == 'control':
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) 
            state = torch.cat((d, v, c, s), dim=0)[:, :256] # 6, 256, 64, 64
        
            data = (
                state.permute(1, 0, 2, 3), # 256, 6, 64, 64, not rescaled
                list(state.shape[-3:]),
                list(state.shape[-3:]),
                sim_id,
            )
        elif self.test_mode == 'simulation':
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], 128, 128)
            state = torch.cat((d, v, c, s), dim=0)[:, :32] # 6, 32, 128, 128
        
            data = (
                state.permute(1, 0, 2, 3), # 32, 6, 128, 128, not rescaled
                list(state.shape[-3:]),
                list(state.shape[-3:]),
                sim_id,
            )
        return data


class Smoke_wave(Dataset):
    def __init__(
        self,
        # dataset,
        dataset_path,
        wave_type, # bior1.3 / bior2.2
        pad_mode, 
        is_train=True,
        is_super_model=False,
        downsample_type = 'time', 
        N_downsample=0, # no more than 1
    ):
        super().__init__()
        self.root = dataset_path
        self.wave_type = wave_type
        self.pad_mode = pad_mode
        assert is_train == True
        self.is_train = is_train
        self.is_super_model = is_super_model
        self.dirname = "train"
        self.downsample_type = downsample_type
        self.N_downsample = 0 if not is_super_model else N_downsample
        self.n_simu = 20000 
        if self.wave_type == 'bior2.2':
            self.RESCALER = torch.tensor([4, 2, 2, 1, 2, 2, 1, 1, 42, 10, 21, 8, 15, 3, 5, 2, 51, 18, 8, 5, 16, 6, 4, 2, \
                                        42, 8, 17, 6, 15, 3, 5, 2, 51, 18, 9, 5, 13, 5, 3, 2, 3, 2]).reshape(1, 42, 1, 1)
        elif self.wave_type == 'bior1.3':
            self.RESCALER = torch.tensor([4, 2, 2, 2, 2, 2, 1, 1, 37, 12, 15, 11, 19, 6, 11, 5, 44, 24, 9, 10, 16, 9, 6, 6, \
                                        37, 10, 15, 8, 19, 5, 11, 5, 43, 24, 9, 10, 16, 9, 5, 5, 3, 2]).reshape(1, 42, 1, 1)
        else:
            raise
        if self.is_super_model:
            self.RESCALER = torch.cat((self.RESCALER[:, :40].repeat(1, 2, 1, 1), self.RESCALER[:, -2:]), dim=1)

    def __len__(self):
        return self.n_simu

    def __getitem__(self, sim_id):
        db = torch.load(os.path.join(self.root, self.dirname, "{}_{}/{}_downsample"\
            .format(self.wave_type, self.pad_mode, self.downsample_type), '{:06d}'.format(sim_id)))
        data = db['coef']
        w_d = data[self.N_downsample][0]
        w_v1 = data[self.N_downsample][1]
        w_v2 = data[self.N_downsample][2]
        w_c1 = data[self.N_downsample][3]
        w_c2 = data[self.N_downsample][4]
        if self.is_super_model:
            w_d_sub = data[self.N_downsample+1][0]
            w_v1_sub = data[self.N_downsample+1][1]
            w_v2_sub = data[self.N_downsample+1][2]
            w_c1_sub = data[self.N_downsample+1][3]
            w_c2_sub = data[self.N_downsample+1][4]
        ori_shape = list(db['ori_shape'])
        if self.downsample_type == 'time':
            ori_shape[0] = math.ceil(ori_shape[0]/2**self.N_downsample)
        else:
            ori_shape[1] = math.ceil(ori_shape[1]/2**self.N_downsample)
            ori_shape[2] = math.ceil(ori_shape[2]/2**self.N_downsample)
        shape = w_d.shape[1:]
        
        # pad
        if self.downsample_type == 'time':
            pad_t, pad_x = int(24 / 2**self.N_downsample), 40 # divisible by 8
        else:
            pad_t, pad_x = 24, int(40 / 2**self.N_downsample)
        nt, nx = w_d.shape[-3], w_d.shape[-1]
        w = torch.cat((w_d, w_v1, w_v2, w_c1, w_c2), dim=0)
        data = F.pad(w, (0, pad_x - nx, 0, pad_x - nx, 0, pad_t - nt), 'constant', 0)
        if self.is_super_model:
            w_sub = torch.cat((w_d_sub, w_v1_sub, w_v2_sub, w_c1_sub, w_c2_sub), dim=0)
            # upsample coefficient, repeat boundary
            if self.downsample_type == 'space':
                w_sub = upsample_coef(w_sub.permute(1, 0, 2, 3).unsqueeze(0), shape, 'space')[0].permute(1, 0, 2, 3)
                data = F.pad(w, (1, 1, 1, 1), mode='replicate')
            else:
                w_sub = upsample_coef(w_sub.permute(1, 0, 2, 3).unsqueeze(0), shape, 'time')[0].permute(1, 0, 2, 3)
                data = torch.cat((w[:,:1], w[:,:shape[0]], w[:,[shape[0]-1]]), dim=1)
            assert data.shape == w_sub.shape
            data = torch.cat((data, w_sub), dim=0)
            data = F.pad(data, (0, pad_x - data.shape[-1], 0, pad_x - data.shape[-2], 0, pad_t - data.shape[-3]), 'constant', 0)

        # initial condition
        data0 = db['init_coef']
        w_d0 = data0[self.N_downsample][0] # only condition on initial density
        W_condition = w_d0.unsqueeze(1).expand(w_d0.shape[0], int(pad_t/4), nx, nx).reshape(-1, nx, nx) # (repeated W_d0, nx, nx)
        W_condition = F.pad(W_condition, (0, pad_x - nx, 0, pad_x - nx), 'constant', 0)
        state = torch.cat((data, W_condition.unsqueeze(0)), dim=0)

        # smoke out
        assert pad_x % 2 == 0
        data_s = db['smokeout']
        w_s = data_s[self.N_downsample].permute(1, 0) # 2, nt
        w_s = w_s.reshape(1, w_s.shape[0], 2, 1, 1).expand(1, w_s.shape[0], 2, int(pad_x/2), pad_x).reshape(1, w_s.shape[0], pad_x, pad_x)
        w_s = F.pad(w_s, (0, 0, 0, 0, 0, pad_t - nt), 'constant', 0)
        state = torch.cat((state, w_s), dim=0)

        data = (
            state.permute(1, 0, 2, 3) / self.RESCALER,
            list(shape),
            ori_shape,
            sim_id,
        )
        return data


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


if __name__ == "__main__":
    dataset = Smoke(
        dataset_path="/data/2d/",
        is_train=True,
    )
    print(len(dataset))
    data = dataset[4]
    print(data[0].shape, data[1], data[2], data[3])

    dataset = Smoke_wave(
        dataset_path="/data/2d/",
        wave_type='bior1.3',
        pad_mode='zero',
        is_super_model=False,
        N_downsample=0,
    )
    data = dataset[4]
    print(data[0].shape, data[1], data[2], data[3])

    dataset = Smoke_wave(
        dataset_path="/data/2d/",
        wave_type='bior1.3',
        pad_mode='zero',
        is_super_model=True,
        N_downsample=1,
    )
    data = dataset[2]
    print(data[0].shape, data[1], data[2], data[3])
