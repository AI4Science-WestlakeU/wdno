import copy
import sys, os

sys.path.append('./ddpm_burgers/')

import torch
import torch.nn as nn
import pywt
from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward
from wave_trans import coef_to_tensor
from ddpm_burgers.data_burgers_1d import DiffusionDataset, get_burgers_preprocess, get_wavelet_super_preprocess
from ddpm_burgers.diffusion_1d import GaussianDiffusion, GaussianDiffusion1D
from ddpm_burgers.train_diffusion import Trainer
from ddpm_burgers.model_utils import get_nablaJ
from ddpm_burgers.generate_burgers import burgers_numeric_solve_free
from ddpm_burgers.wavelet_utils import upsample_coef
from train_ddpm_burgers import get_2d_ddpm
from result_io import save_acc


SOLVER_N_UPSAMPLE = 4 # presicion of solver: 2**4=16

def mse_deviation(u1, u2, report_all=False):
    u1, u2 = u1.clone(), u2.clone()
    if report_all:
        mse = (u1 - u2).square().mean((-1, -2))
        mae = (u1 - u2).abs().mean((-1, -2))
        ep = 1e-5
        return mse, mae, mse / (u2 + ep).square().mean(), mae / (u2 + ep).abs().mean()
    return (u1 - u2).square().mean((-1, -2))


def metric(
    u_target: torch.Tensor, 
    f: torch.Tensor, 
    total_upsample_t: torch.Tensor, 
    wf=0,
    upsample_t=0,
    f_upsampled=None,
    report_all=False, 
    u=None, 
    evaluate=False, 
    ):
    '''
    Evaluates the control based on the state deviation and the control cost.
    Note that f and u should NOT be rescaled. (Should be directly input to the solver)

    Arguments:
        u_target:
            Ground truth states
            size: (batch_size, Nt, Nx) (currently Nt = 81, Nx = 120)
        f: 
            Generated control force
            size: (batch_size, Nt - 1, Nx) (currently Nt = 81, Nx = 120)
        eval: whether to calculate loss given u. If true, evaluate how good the
             u and f are.
    Returns:
        J_actual:
            Deviation of controlled u from target u for each sample, in MSE.
            When target is 'final_u', evaluate only at the final time stamp
            size: (batch_size)
        
        control_energy:
            Cost of the control force for each sample, in sum of square.
            size: (bacth_size)
    '''
    u_target, f = u_target.clone(), f.clone() # avoid side-effect # super resolution u

    assert len(u_target.size()) == len(f.size()) == 3

    if evaluate: # super resolution u
        u_controlled = u.clone()
    else:
        if f_upsampled == None:
            u_controlled = burgers_numeric_solve_free(u_target[:, 0], f, visc=0.01, T=8.0, num_t=80*2**upsample_t)
        else:
            u_controlled = burgers_numeric_solve_free(u_target[:, 0], f_upsampled[..., ::2], visc=0.01, T=8.0, num_t=160)[:, ::2]

    # eval J_actual
    sub_N = int(u_controlled.shape[-1]/f.shape[-1])
    mse = (u_controlled[:, -1, ::sub_N] - u_target[:, -1, ::sub_N]).square().mean(-1)
    
    mse_median, _ = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().median(-1)
    mae = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().mean(-1) # MAE
    mae_median, _ = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().median(-1)
    ep = 1e-5
    nmse = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().mean(-1).sqrt() / (u_target[:, -1, :].square().mean().sqrt() + ep) # normalized MSE
    nmae = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().mean(-1).sqrt() / (u_target[:, -1, :].abs().mean().sqrt() + ep) # normalized MAE

    if not report_all:
        J_actual = mse
    else:
        J_actual = (mse, mse_median, mae, mae_median, nmse, nmae)
    
    control_energy = f.square().sum((-1, -2)) / (2**upsample_t)**2
    total_J = mse + wf * control_energy

    return J_actual, control_energy, total_J

def ddpm_guidance_loss(
        u_target, u, f, 
        wu=0, wf=0,
        condition_f=False,
):
    '''
    Arguments:
        u_target: (batch_size, Nt, Nx)
        u: (batch_size, Nt, Nx)
        f: (batch_size, Nt - 1, Nx)
        
    '''

    u0_gt = u_target[:, 0, :]
    uf_gt = u_target[:, -1, :]
    u0 = u[:, 0, :]
    uf = u[:, -1, :]

    if condition_f:
        loss_u = (u0 - u0_gt).square()
    else:
        loss_u = (u0 - u0_gt).square() + (uf - uf_gt).square()
    loss_u = loss_u.mean(-1).sum() # sum the batch

    loss_f = f.square().sum() # sum the batch

    return (loss_u + loss_f * wf) * wu


# Loading dataset and model

def load_burgers_dataset(args, RESCALER):
    # NOTE: this dataset is only used for instantiating the Trainer class
    tmp_dataset = DiffusionDataset(
        f'data/{args.dataset}/train', # dataset of f varying in both space and time
        preprocess=get_burgers_preprocess(
            rescaler=RESCALER, 
        )
    )
    return tmp_dataset

def load_burgers_dataset_wavelet(args, RESCALER):
    # NOTE: this dataset is only used for instantiating the Trainer class
    tmp_dataset = DiffusionDataset(
        f'data/{args.dataset}/coef_{args.wave_type}_{args.pad_mode}_super', # dataset of f varying in both space and time
        preprocess=get_wavelet_super_preprocess(
            rescaler=1., 
            is_super_model=False,
            mode = args.pad_mode,
            wave_type = args.wave_type,
            is_condition_u0 = args.is_condition_u0,
            is_condition_uT = args.is_condition_uT,
        )
    )
    return tmp_dataset

def get_target(args, is_wave, target_i, f=False, N_upsample=0, device=0, dataset='free_u_f_1e5', **dataset_kwargs):
    '''
    is_wavelet==True & f==True: return [N*(1+3*3), 45(not padded!!), 64(padded!!)]
    is_wavelet==False & f==True: return:[N, 81(not padded!!), 128(padded!!)]
    return: not rescaled
    '''
    test_dataset = DiffusionDataset(
        f'data/{args.dataset}_super/test', # dataset of super-resolution
        preprocess=get_burgers_preprocess(
            rescaler=1., 
            is_super_model_test=True,
            upsample_t=N_upsample,
            upsample_x=N_upsample,
            **dataset_kwargs, 
        )
    )

    ori_shape = test_dataset.ori_shape
    if not f: # return only u
        ret = test_dataset.get(target_i).cuda(device)[:, 0, :ori_shape[-2]]
    else: # return only f
        ret = test_dataset.get(target_i).cuda(device)[:, 1, :ori_shape[-2]]

    if len(ret.size()) == 2: # if target_i is int
        ret = ret.unsqueeze(0)
    
    if is_wave:
        if not f:
            # W_u0, W_uT
            pad_t, pad_x = 64 * 2**N_upsample, 64 * 2**N_upsample
            xfm1d = DWT1DForward(J=1, mode=args.pad_mode, wave=args.wave_type).to(ret.device)
            Yl0, Yh0 = xfm1d(ret[:, [0, -1], :ori_shape[-1]])
            nx = Yl0.shape[-1]
            W_condition = torch.zeros((ret.shape[0], pad_t, pad_x), device = ret.device) # (N, (W_u0)+(W_uT), pad_x)
            n_repeat = int(pad_t / 4)
            if args.is_condition_u0:
                W_condition[:, :n_repeat, :nx] = Yl0[:, [0]].expand(ret.shape[0], n_repeat, nx)
                W_condition[:, n_repeat:n_repeat*2, :nx] = Yh0[0][:, [0]].expand(ret.shape[0], n_repeat, nx)
            if args.is_condition_uT:
                W_condition[:, n_repeat*2:n_repeat*3, :nx] = Yl0[:, [1]].expand(ret.shape[0], n_repeat, nx)
                W_condition[:, n_repeat*3:n_repeat*4, :nx] = Yh0[0][:, [1]].expand(ret.shape[0], n_repeat, nx)
            
            ret = W_condition
        else:
            xfm = DWTForward(J=1, mode=args.pad_mode, wave=args.wave_type).cuda(device) 
            ret = ret[:, :, :ori_shape[-1]].unsqueeze(1)
            Yl, Yh = xfm(ret)
            ret = coef_to_tensor(Yl, Yh)[:, 0]
            if N_upsample != 0:
                ret = ret
            ret = nn.functional.pad(ret, (0, 64*2**N_upsample - ret.shape[-1], 0, 64*2**N_upsample - ret.shape[-2]), 'constant', 0)

    return ret



def load_2dconv_base_model(model_i, args, RESCALER):
    if args.is_wavelet:
        dataset = load_burgers_dataset_wavelet(args, RESCALER)
    else:
        dataset = load_burgers_dataset(args, RESCALER)
    shape = dataset.shape
    ori_shape = dataset.ori_shape
    ddpm = get_2d_ddpm(shape, ori_shape, args, RESCALER, is_super_model=False, upsample_t=0, upsample_x=0)

    trainer = Trainer(
        ddpm, 
        dataset, 
        is_super_model=False,
        wave_type=args.wave_type,
        pad_mode=args.pad_mode,
        rescaler=RESCALER, 
        results_folder=os.path.join(args.results_folder, model_i), 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    return ddpm

def load_2dconv_super_model(model_i, args, RESCALER):
    assert args.is_super_model
    if args.is_wavelet:
        dataset = load_burgers_dataset_wavelet(args, RESCALER)
    else:
        dataset = load_burgers_dataset(args, RESCALER)
    shape_init = dataset.shape
    ori_shape_init = dataset.ori_shape
    shape, ori_shape = [], []
    for i in range(1, args.upsample_x+1):
        shape.append([(2**i)*2*(shape_init[-2]//2) + shape_init[-2]%2, \
            (2**i)*2*(shape_init[-1]//2) + shape_init[-1]%2])
        ori_shape.append([(2**i)*2*(ori_shape_init[-2]//2) + ori_shape_init[-2]%2, \
            (2**i)*2*(ori_shape_init[-1]//2) + ori_shape_init[-1]%2])
    args.dim = args.super_dim
    ddpm = get_2d_ddpm(shape, ori_shape, args, RESCALER, is_super_model=True, upsample_t=args.upsample_t, upsample_x=args.upsample_x)

    trainer = Trainer(
        ddpm, 
        dataset, 
        is_super_model=True,
        wave_type=args.wave_type,
        pad_mode=args.pad_mode,
        rescaler=RESCALER, 
        results_folder=os.path.join(args.results_folder, model_i), 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.super_checkpoint if 'checkpoint' in args.__dict__ else 10)
    return ddpm
    
