import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from ddpm_burgers.data_burgers_1d import DiffusionDataset, get_burgers_preprocess, get_wavelet_preprocess, get_wavelet_super_preprocess
from ddpm_burgers.diffusion_1d import GaussianDiffusion1D, GaussianDiffusion
from ddpm_burgers.unet import Unet2D
from ddpm_burgers.train_diffusion import Trainer

import matplotlib.pyplot as plt
from ddpm_burgers.result_io import merge_save_dict
from datetime import datetime
import yaml


parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--exp_id', default=datetime.today().strftime("%Y-%m-%d-%H:%M:%S"), type=str,
                    help='experiment folder id')
parser.add_argument('--results_folder', default='./results/', type=str,
                    help='results folder')
parser.add_argument('--dataset', default='1d', type=str,
                    help='dataset name')
parser.add_argument('--train_num_steps', default=100000, type=int,
                    help='train_num_steps')
parser.add_argument('--test_interval', default=10000, type=int,
                    help='test every test_interval steps')
parser.add_argument('--checkpoint_interval', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')

# wavelet
parser.add_argument('--is_wavelet', default=True, type=eval,
                    help='If learning wavelet coefficients')
parser.add_argument('--is_super_model', default=False, type=eval,
                    help='If training the super resolution model')
parser.add_argument('--wave_type', default='bior2.4', type=str,
                    help='type of wavelet: bior2.4, bior4.4, db4, db5, sym4, ...')
parser.add_argument('--pad_mode', default='periodization', type=str,
                    help='padding mode for wavelet transform: zero')
parser.add_argument('--N_downsample', default=3, type=int,
                    help='number of times of subsampling for training of super resolution model,\
                         no more than 3')

# condition of diffusion
parser.add_argument('--is_condition_pad', default=True, type=eval,
                    help='If learning p(u,f | (u,f)_padded)')
parser.add_argument('--is_condition_u0', default=False, type=eval,
                    help='If learning p(u_[1, T],f | u0)')
parser.add_argument('--is_condition_uT', default=False, type=eval,
                    help='If learning p(u_[0, T-1],f | uT)')
parser.add_argument('--is_condition_f', default=False, type=eval,
                    help='If learning p(u_[0, T] | f)')

# unet hyperparam
parser.add_argument('--dim', default=128, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups', default=1, type=int,
                    help='group num in GroupNorm')
parser.add_argument('--dim_muls', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')

# sampling setting: does not affect
parser.add_argument('--using_ddim', default=False, type=eval,
                    help='If using DDIM')
parser.add_argument('--ddim_eta', default=0., type=float, help='eta in DDIM')
parser.add_argument('--ddim_sampling_steps', default=1000, type=int, help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')

parser.add_argument('--timesteps', default=1000, type=int, help='timesteps of diffusion model')
parser.add_argument('--beta_schedule', default='cosine', type=str, 
                    help='beta schedule: cosine | linear')


def get_dataset(RESCALER, is_wavelet, wave_type, pad_mode, dataset='1d'):
    if not args.is_super_model:
        if is_wavelet:
            return DiffusionDataset(
                f'data/{dataset}/coef_{wave_type}_{pad_mode}_super',
                preprocess=get_wavelet_super_preprocess( 
                    rescaler=RESCALER, 
                    is_super_model = args.is_super_model,
                    mode = args.pad_mode,
                    wave_type = args.wave_type,
                    is_condition_u0 = args.is_condition_u0,
                    is_condition_uT = args.is_condition_uT,
                )
            )
        else:
            return DiffusionDataset(
                f'data/{dataset}/train', # dataset of f varying in both space and time
                preprocess=get_burgers_preprocess( 
                    rescaler=RESCALER, 
                )
            )
    else:
        # assert is_wavelet
        if is_wavelet:
            dataset_list = []
            for i in range(args.N_downsample):
                dataset_i = DiffusionDataset(
                    f'data/{dataset}/coef_{wave_type}_{pad_mode}_super',
                    preprocess=get_wavelet_super_preprocess( 
                        rescaler=RESCALER, 
                        is_super_model = args.is_super_model,
                        N_downsample = i,
                        mode = args.pad_mode,
                        wave_type = args.wave_type,
                        is_condition_u0 = args.is_condition_u0,
                        is_condition_uT = args.is_condition_uT,
                    )
                )
                dataset_list.append(dataset_i)
        else:
            dataset_list = []
            for i in range(args.N_downsample):
                dataset_i = DiffusionDataset(
                    f'data/{dataset}/train', # dataset of f varying in both space and time
                    preprocess=get_burgers_preprocess( 
                        rescaler=RESCALER, 
                        is_super_model_train = args.is_super_model,
                        N_downsample = i,
                    )
                )
                dataset_list.append(dataset_i)
        return dataset_list

def get_2d_ddpm(shape, ori_shape, args, RESCALER, is_super_model, upsample_t=0, upsample_x=0):
    # max size
    if args.is_wavelet:
        sim_time_stamps, sim_space_grids = 64*2**upsample_t, 64*2**upsample_x 
    else:
        sim_time_stamps, sim_space_grids = 128*2**upsample_t, 128*2**upsample_x  # sample * 2 * timegrid (padded to 128) * spacegrid (padded to 128)

    # decide channel number.
    # 1 ddpm: u and f in Burger's equation. 
    if args.is_wavelet:
        channels = 8
        if is_super_model:
            channels += 8
        if args.is_condition_u0 or args.is_condition_uT:
            channels += 1
    else:
        channels = 2
        if is_super_model:
            channels = channels * 2
    
    # make model
    u_net = Unet2D(
        dim = args.dim, 
        init_dim = None,
        out_dim = channels,
        dim_mults = args.dim_muls,
        channels = channels, 
        resnet_block_groups = args.resnet_block_groups,
    )
    ddpm = GaussianDiffusion(
        u_net, 
        seq_length=(sim_time_stamps, sim_space_grids), 
        # wavelet
        is_wavelet = args.is_wavelet,
        padded_shape=shape,
        ori_shape = ori_shape,
        pad_mode = args.pad_mode,
        wave_type = args.wave_type,
        # super model
        is_super_model = is_super_model,
        upsample_t=upsample_t,
        upsample_x=upsample_x,
        # diffusion
        timesteps = args.timesteps,
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, 
        ddim_sampling_eta=args.ddim_eta,
        beta_schedule=args.beta_schedule,
        loss_layer_weight=RESCALER,
        # condition of diffusion
        is_condition_pad=args.is_condition_pad,
        is_condition_u0=args.is_condition_u0, 
        is_condition_uT=args.is_condition_uT, 
        is_condition_f=args.is_condition_f, 
    )
    return ddpm


def run_2d_Unet(dataset, shape, ori_shape, args, RESCALER):
    ddpm = get_2d_ddpm(shape, ori_shape, args, RESCALER, args.is_super_model)
    trainer = Trainer(
        ddpm, 
        dataset, 
        is_super_model=args.is_super_model,
        wave_type=args.wave_type,
        pad_mode=args.pad_mode,
        rescaler=RESCALER, 
        exp_name=args.exp_id,
        results_folder=os.path.join(args.results_folder, args.exp_id), 
        train_num_steps=args.train_num_steps, 
        test_every = args.test_interval,
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.train()


def log_exp(args, file='log.yaml'):
    res_dir = os.path.join(args.results_folder, args.exp_id)
    
    res_path = os.path.join(res_dir, file)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(res_path, 'a+') as f:
        s = yaml.safe_load(f)
    # handle the case where s is empty
    if s is None:
        s = {'nothing': None}
    
    if args.exp_id in s:
        raise ValueError('exp_id already exists. specify another one.')
    merge_save_dict(res_path, {args.exp_id: vars(args)})
    

if __name__ == "__main__":    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()
    print(args)

    # rescale data to [-1, 1]
    if not args.is_wavelet:
        RESCALER = torch.tensor([10])
    else:
        if args.pad_mode == 'periodization':
            if args.wave_type == 'bior2.4':
                RESCALER = torch.tensor([10, 3, 3, 1, 21, 5, 5, 1]).view(1, 8, 1, 1)
            elif args.wave_type == 'bior1.3':
                RESCALER = torch.tensor([8, 5, 4, 2, 21, 4, 3, 1]).view(1, 8, 1, 1)
            elif args.wave_type == 'db4':
                RESCALER = torch.tensor([8, 4, 3, 2, 21, 3, 3, 1]).view(1, 8, 1, 1)
            elif args.wave_type == 'sym4':
                RESCALER = torch.tensor([8, 5, 4, 2, 21, 6, 6, 2]).view(1, 8, 1, 1)
            else:
                raise ValueError('Input RESCALER.') 
        else:
            raise ValueError('Input RESCALER.') 

        if args.is_super_model:
            RESCALER = RESCALER.repeat(1, 2, 1, 1)
        if args.is_condition_u0 or args.is_condition_uT:
            RESCALER = torch.cat((RESCALER, 10*torch.ones_like(RESCALER[:,[0]])), dim=1)
    
    # get dataset
    dataset = get_dataset(RESCALER, args.is_wavelet, args.wave_type, args.pad_mode, args.dataset)
    if not args.is_super_model:
        shape = dataset.shape
        ori_shape = dataset.ori_shape
    else:
        shape = [dataset[i].shape for i in range(len(dataset))]
        ori_shape = [dataset[i].ori_shape for i in range(len(dataset))]
    print(f'data shape: {shape}')

    if args.is_wavelet:
        print(f'Rescaling data by dividing {RESCALER.squeeze()}')
    else:
        print(f'Rescaling data by dividing {RESCALER}')

    log_exp(args)
    run_2d_Unet(dataset, shape, ori_shape, args, RESCALER)
