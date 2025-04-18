import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse 

from ddpm_burgers.model_utils import get_scheduler, get_nablaJ, sigmoid_schedule
from ddpm_burgers.generate_burgers import burgers_numeric_solve_free
from ddpm_burgers.wavelet_utils import get_wt_T, upsample_coef
from ddpm_burgers.test_util import ddpm_guidance_loss, mse_deviation, metric, get_target, load_2dconv_base_model, load_2dconv_super_model
from ddpm_burgers.result_io import save_acc
from wave_trans import tensor_to_coef, coef_to_tensor, tensor_to_coef_super

import argparse
from IPython import embed

SOLVER_N_UPSAMPLE = 4 # presicion of solver: 2**4=16

parser = argparse.ArgumentParser(description='Eval EBM model')
parser.add_argument('--exp_id', type=str,
                    help='trained base model id')
parser.add_argument('--super_exp_id', default='', type=str,
                    help='trained super model id')
parser.add_argument('--results_folder', default='./results/', type=str,
                    help='results folder')
parser.add_argument('--save_file', default='results/evaluate/result.yaml', type=str,
                    help='file to save')
parser.add_argument('--dataset', default='1d', type=str,
                    help='dataset name for evaluation (eval samples drawn from)')
parser.add_argument('--model_str', default='', type=str,
                    help='description (where is this used?)')
parser.add_argument('--report_all', default=True, type=eval,
                    help='report all metrics')

# experiment settings
parser.add_argument('--Ntest', default=50, type=int,
                    help='total number of samples for testing')
parser.add_argument('--batch_size', default=50, type=int,
                    help='batch size for testing')
parser.add_argument('--upsample_t', default=0, type=int,
                    help='ONLY IMPLEMENT upsample_t==upsample_x. Times of upsampling time, nt *= 2**upsample_t.')
parser.add_argument('--upsample_x', default=0, type=int,
                    help='ONLY IMPLEMENT upsample_t==upsample_x. Times of upsampling space, nx *= 2**upsample_x.')

# wavelet
parser.add_argument('--is_wavelet', default=True, type=eval,
                    help='If learning wavelet coefficients')
parser.add_argument('--is_super_model', default=False, type=eval,
                    help='If testing the super resolution model')
parser.add_argument('--wave_type', default='bior2.4', type=str,
                    help='type of wavelet: bior2.4, bior4.4, db4, db5, sym4, ...')
parser.add_argument('--pad_mode', default='periodization', type=str,
                    help='padding mode for wavelet transform: zero')
                    
# p(u, w) model training
parser.add_argument('--checkpoint', default=10, type=int,
                    help='which checkpoint base model to load')
parser.add_argument('--super_checkpoint', default=10, type=int,
                    help='which checkpoint super model to load')
parser.add_argument('--checkpoint_interval', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')
parser.add_argument('--train_num_steps', default=100000, type=int,
                    help='train_num_steps')

# sampling, not used in training
parser.add_argument('--using_ddim', default=False, type=eval,
                    help='If using DDIM')
parser.add_argument('--ddim_eta', default=0., type=float, help='eta in DDIM')
parser.add_argument('--ddim_sampling_steps', default=100, type=int, 
                    help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
parser.add_argument('--J_scheduler', default=None, type=str,
                    help='which J_scheduler to use. None means no scheduling.')
parser.add_argument('--wfs', nargs='+', default=[0], type=float,
                    help='guidance intensity of energy')
parser.add_argument('--wus', nargs='+', default=[0], type=float,
                    help='guidance intensity of state deviation')

# ddpm and unet
parser.add_argument('--timesteps', default=1000, type=int, help='timesteps of diffusion model')
parser.add_argument('--beta_schedule', default='cosine', type=str, 
                    help='beta schedule: cosine | linear')
                    
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
                    help='')
parser.add_argument('--dim_muls', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')
parser.add_argument('--super_dim', default=64, type=int,
                    help='first layer feature dim num in Unet')


# loss, guidance and utils
    
def get_loss_fn_2dconv(
        args, shape, ori_shape, RESCALER,
        is_super_model=False, low=0, N_upsample=0,
        wf=0, wu=0, target_i=0, device=0, 
        dataset='1d', condition_f=False, 
):
    """
    Gets target data and returns a function that computes guidance loss
    """
    u_target = get_target(
        args, is_wave=False, target_i=target_i, 
        N_upsample=N_upsample, device=device, dataset=dataset
    )
    
    def loss_fn_2dconv(x):
        # Apply rescaling based on model type
        x = x[:, :8] * RESCALER[:, :8] if is_super_model and args.is_wavelet else x * RESCALER
        
        if not args.is_wavelet:
            return ddpm_guidance_loss(
                u_target[:, :shape[-2], :shape[-1]], 
                x[:, 0, :shape[-2], :shape[-1]], 
                x[:, 1, :shape[-2]-1, :shape[-1]], 
                wu=wu, wf=wf, condition_f=condition_f
            )
        else:
            ifm = DWTInverse(mode=args.pad_mode, wave=args.wave_type).to(x.device)
            Yl, Yh = tensor_to_coef(x, shape)
            u_f = ifm((Yl, Yh))[:, :, :ori_shape[-2], :ori_shape[-1]]
            u, f = u_f[:, 0], u_f[:, 1, :ori_shape[-2]-1]
            
            return ddpm_guidance_loss(
                u_target[:, :ori_shape[-2], :ori_shape[-1]], 
                u, f, wu=wu, wf=wf, condition_f=condition_f
            )
    
    return loss_fn_2dconv

def get_nablaJ_2dconv(**kwargs):
    return get_nablaJ(get_loss_fn_2dconv(**kwargs))


# run exp

def diffuse_2dconv(ddpm, args, custom_metric, ret_ls=False, RESCALER=1, **kwargs):
    '''
    Samples from diffusion model and evaluates results.
    
    Args:
        ddpm: Diffusion model
        args: Configuration arguments
        custom_metric: Function to calculate metrics
        RESCALER: Scaling factor for outputs
        
    Returns:
        x: Model output 
        ddpm_mse: Mean squared error
        J_diffused: objective predicted by diffusion model 
        J_actual: objective from solver
        energy: Energy values
        total_J: Total objective values
    '''
    # Determine shapes based on upsampling
    if 'N_upsample' not in kwargs:
        shape = ddpm.padded_shape
        ori_shape = ddpm.ori_shape
        N_upsample = 0
    else:
        N_upsample = kwargs['N_upsample']
        shape = ddpm.padded_shape[N_upsample-1]
        ori_shape = ddpm.ori_shape[N_upsample-1]
    
    if not ddpm.is_wavelet:
        u_from_x = lambda x: x[:, 0, :shape[-2], :shape[-1]]
        f_from_x = lambda x: x[:, 1, :shape[-2]-1, :shape[-1]]

    # Sampling
    x = ddpm.sample(**kwargs) * RESCALER
    if ddpm.is_wavelet:
        # Process wavelet coefficients
        ifm = DWTInverse(mode=args.pad_mode, wave=args.wave_type).to(x.device)
        if 'low' in kwargs:
            Yl, Yh = tensor_to_coef_super(x, shape)
        else:
            Yl, Yh = tensor_to_coef(x, shape)
        x = x[:, :, :shape[-2], :shape[-1]]
        u_f = ifm((Yl, Yh))[:, :, :ori_shape[-2], :ori_shape[-1]]
        u, f = u_f[:, 0], u_f[:, 1, :ori_shape[-2]-1]
    else:
        u, f = u_from_x(x), f_from_x(x)
    
    # Solver
    if ddpm.is_condition_f:
        x_gt = kwargs['x_gt'][:, :, :ori_shape[-1]*2**(args.upsample_x-N_upsample)]
    else:
        x_gt = burgers_numeric_solve_free(u[:, 0], f, visc=0.01, T=8.0, num_t=80*2**N_upsample, output_space_downsample=False)\
                [:, :, ::2**(SOLVER_N_UPSAMPLE-args.upsample_x)]
    
    # Calculate MSE for different cases
    if args.is_super_model and args.is_condition_f:
        # Calculate MSE for different interpolation methods
        u_upsample = F.interpolate(u.unsqueeze(1), size=(x_gt.shape[-2], x_gt.shape[-1]), mode='bilinear', align_corners=False)[:, 0]
        ddpm_mse1 = mse_deviation(u_upsample[:, 1:], x_gt[:, 1:]).cpu()
        
        u_upsample = F.interpolate(u.unsqueeze(1), size=(x_gt.shape[-2], x_gt.shape[-1]), mode='nearest')[:, 0]
        ddpm_mse2 = mse_deviation(u_upsample[:, 1:], x_gt[:, 1:]).cpu()
        
        sub_N = int(x_gt.shape[-1]/u.shape[-1])
        ddpm_mse3 = mse_deviation(u[:, 1:], x_gt[:, ::sub_N, ::sub_N][:, 1:]).cpu()
        
        ddpm_mse = torch.cat((ddpm_mse1.unsqueeze(1), ddpm_mse2.unsqueeze(1), ddpm_mse3.unsqueeze(1)), dim=1)
    else:
        sub_N = int(x_gt.shape[-1]/u.shape[-1])
        ddpm_mse = mse_deviation(u[:, 1:], x_gt[:, 1:, ::sub_N]).cpu()

    # Calculate metrics
    sub_N = int(x_gt.shape[-1]/u.shape[-1])
    u_repeat = u.unsqueeze(-1).expand(u.shape[0], u.shape[1], u.shape[2], sub_N).reshape(u.shape[0], u.shape[1], u.shape[2]*sub_N)[:, :, :x_gt.shape[2]]
    J_diffused, _, _ = custom_metric(f, upsample_t=N_upsample, u=u_repeat, evaluate=True)
    if ddpm.is_condition_f:
        J_actual, energy, total_J = custom_metric(f, upsample_t=N_upsample, u=u_repeat, evaluate=True)
    else:
        J_actual, energy, total_J = custom_metric(f, upsample_t=N_upsample, u=x_gt, evaluate=True)
    
    # Convert results to CPU numpy arrays
    elems_to_cpu_numpy_if_tuple = lambda x: x.detach().cpu().numpy() if type(x) is not tuple else np.array([xi.detach().cpu().numpy() for xi in x])
    J_diffused = elems_to_cpu_numpy_if_tuple(J_diffused)
    J_actual = elems_to_cpu_numpy_if_tuple(J_actual)
    energy = energy.detach().cpu().numpy()
    total_J = total_J.detach().cpu().numpy()

    if args.is_wavelet:
        return x[:, :8], ddpm_mse, J_diffused, J_actual, energy, total_J
    else:
        return x[:, :2], ddpm_mse, J_diffused, J_actual, energy, total_J

def evaluate(
        model_i, super_model_i, args, 
        # exp settings
        Ntest=50, batch_size=25,
        # guidance choices
        wu=0, wf=0, 
        # model choice
        RESCALER=1,
):
    ddpm = load_2dconv_base_model(model_i, args, RESCALER)
    if args.is_super_model:
        ddpm_super = load_2dconv_super_model(super_model_i, args, RESCALER)
    RESCALER_base, RESCALER = RESCALER.cuda(), RESCALER.cuda()
    if args.is_super_model:
        RESCALER_base = RESCALER[:, 8:17] if args.is_wavelet else RESCALER
    
    rep = math.ceil(Ntest / batch_size)

    logs = {k: {'mses': [], 'l_gts': [], 'l_dfs': [], 'energies': [], 'l_total': []} 
           for k in range(args.upsample_x+1)}
    sampled_coefs = []

    for i in range(rep):
        print("Batch No.", i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(0)

        if (i + 1) * batch_size <= Ntest:
            target_idx = list(range(i * batch_size, (i + 1) * batch_size)) # should also work if being an iterable
        else:
            target_idx = list(range(i * batch_size, Ntest)) # should also work if being an iterable
            batch_size = Ntest - i * batch_size

        # diffuse on base resolution    
        u_target_ori = get_target(args, is_wave=False, target_i=target_idx, N_upsample=args.upsample_x, dataset=args.dataset)\
                        [:, :, :ddpm.ori_shape[-1]*2**args.upsample_x] 
        u_target = u_target_ori[:, :, ::2**args.upsample_x]
        u_condition = get_target(args, is_wave=args.is_wavelet, target_i=target_idx, dataset=args.dataset)

        sampled_coef, ddpm_mse, J_diffused, J_actual, energy, total_J = diffuse_2dconv(
            ddpm, args, RESCALER=RESCALER_base, batch_size=batch_size,
            J_scheduler=get_scheduler(args.J_scheduler), 
            custom_metric=lambda f, **kwargs: metric(u_target_ori, f, args.upsample_t, wf=wf, report_all=args.report_all, **kwargs), 
            u_init=u_condition[:, 0, :] / RESCALER if not args.is_wavelet else u_condition[:, :32] / RESCALER_base.squeeze()[-1], 
            u_final=u_condition[:, -1, :] / RESCALER if not args.is_wavelet else u_condition[:, -32:] / RESCALER_base.squeeze()[-1], 
            f=get_target(args, f=True, is_wave=args.is_wavelet, target_i=target_idx, dataset=args.dataset) / 
              (RESCALER if not args.is_wavelet else RESCALER_base[:, 4:8]), 
            x_gt=u_target_ori,
            nablaJ=get_nablaJ_2dconv(
                args=args, shape=ddpm.padded_shape, ori_shape=ddpm.ori_shape,
                RESCALER=RESCALER_base, target_i=target_idx, 
                wu=wu, wf=wf, dataset=args.dataset, condition_f=args.is_condition_f,
            ),  
        )
        logs[0]['mses'].append(ddpm_mse)
        logs[0]['l_gts'].append(J_actual)
        logs[0]['l_dfs'].append(J_diffused)
        logs[0]['energies'].append(energy)
        logs[0]['l_total'].append(total_J)
        sampled_coefs.append(sampled_coef.cpu().numpy())
        
        # zero-shot super-resolution
        if args.is_super_model:
            for k in range(1, args.upsample_x+1):
                sampled_coef = upsample_coef(sampled_coef, ddpm_super.padded_shape[k-1]) 
                pad_size = 64*2**k if args.is_wavelet else 128*2**k
                sampled_coef = nn.functional.pad(sampled_coef, (0, pad_size - sampled_coef.shape[-1], 0, 
                                pad_size - sampled_coef.shape[-2]), 'constant', 0)
                
                u_target = u_target_ori[:, :, ::2**(args.upsample_x-k)]
                u_condition = get_target(args, is_wave=args.is_wavelet, target_i=target_idx, 
                                        N_upsample=k, dataset=args.dataset)

                sampled_coef, ddpm_mse, J_diffused, J_actual, energy, total_J = diffuse_2dconv(
                    ddpm_super, args, N_upsample=k, RESCALER=RESCALER, batch_size=batch_size,
                    J_scheduler=get_scheduler(args.J_scheduler), 
                    custom_metric=lambda f, **kwargs: metric(u_target_ori, f, args.upsample_t, wf=wf, report_all=args.report_all, **kwargs), 
                    low=sampled_coef / (RESCALER[:, 8:16] if args.is_wavelet else RESCALER),
                    u_init=u_condition[:, 0, :] / RESCALER if not args.is_wavelet else u_condition[:, :32*2**k] / RESCALER.squeeze()[-1], 
                    u_final=u_condition[:, -1, :] / RESCALER if not args.is_wavelet else u_condition[:, -32*2**k:] / RESCALER.squeeze()[-1], 
                    f=get_target(args, f=True, is_wave=args.is_wavelet, target_i=target_idx, N_upsample=k, dataset=args.dataset) / 
                        (RESCALER if not args.is_wavelet else RESCALER[:, 4:8]), 
                    x_gt=u_target_ori,
                    nablaJ=get_nablaJ_2dconv(
                        args=args, shape=ddpm_super.padded_shape[k-1], ori_shape=ddpm_super.ori_shape[k-1],
                        RESCALER=RESCALER, is_super_model=True, N_upsample=k,
                        low=sampled_coef / (RESCALER[:, 8:16] if args.is_wavelet else RESCALER),
                        target_i=target_idx, wu=0, wf=0, dataset=args.dataset, condition_f=args.is_condition_f,
                    ),  
                )
                logs[k]['mses'].append(ddpm_mse)
                logs[k]['l_gts'].append(J_actual)
                logs[k]['l_dfs'].append(J_diffused)
                logs[k]['energies'].append(energy)
                logs[k]['l_total'].append(total_J)

    return [[np.concatenate(logs[k]['mses']), \
            *(np.concatenate(logs[k]['l_dfs'], axis=1)[i] for i in range(logs[k]['l_dfs'][0].shape[0])), \
            *(np.concatenate(logs[k]['l_gts'], axis=1)[i] for i in range(logs[k]['l_dfs'][0].shape[0])), \
            np.concatenate(logs[k]['energies']), \
            np.concatenate(logs[k]['l_total'])] for k in range(args.upsample_x+1)]


def save_eval_results(save_values, model_i, fname, model_str=''):
    print('Evaluated on super, current and base resolution。')
    attr = ""
    for k in range(len(save_values)): 
        names = [
            f'mse_gt_{k}_{attr}, linear sr/nearest sr/without sr:',  
            f'J_diffused_mse_{k}_{attr}', f'J_diffused_mse_median_{k}_{attr}', f'J_diffused_mae_{k}_{attr}', f'J_diffused_mae_median_{k}_{attr}', 
            f'J_diffused_nmse_{k}_{attr}', f'J_diffused_nmae_{k}_{attr}', f'J_actual_mse_{k}_{attr}', f'J_actual_mse_median_{k}_{attr}', 
            f'J_actual_mae_{k}_{attr}', f'J_actual_mae_median_{k}_{attr}', f'J_actual_nmse_{k}_{attr}', f'J_actual_nmae_{k}_{attr}', 
            f'energy_{k}_{attr}', f'totalJ_{k}_{attr}'
        ]
        save_value = save_values[k]

        for acc, inner_str in zip(save_value, names):
            print(k, inner_str, acc.mean(0))
            save_acc(
                acc, 
                fname, 
                make_dict_path=lambda acc, dict_args: {
                    dict_args['model_name']: {
                        'model_description': dict_args['model_str'], 
                        dict_args['guidance_str']: {
                            inner_str: acc
                        }
                    }
                }, 
                model_name=f'{model_i}', 
                model_str=model_str, 
                guidance_str=f'wu={wu:.1f}, wf={wf:.1f}' # these values are from the outer scope
            )

if __name__ == '__main__':
    args = parser.parse_args()

    if args.is_super_model:
        assert args.upsample_t == args.upsample_x, "super model only support the same upsample_t and upsample_x"

    model_i, super_model_i, cp, super_cp, model_str = args.exp_id, args.super_exp_id, args.checkpoint, args.super_checkpoint, args.model_str

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

    print(args)
    print('Results saved in: ', args.save_file)
    for wf in args.wfs:
        for wu in args.wus:
            print(f'wu={wu:.2f}, wf={wf:.7f}')
            results = evaluate(
                model_i = model_i,
                super_model_i = super_model_i,
                args = args,
                Ntest = args.Ntest,
                batch_size = args.batch_size, 
                wu = wu, 
                wf = wf, 
                RESCALER = RESCALER,
            )
            save_eval_results(
                results, 
                model_i = model_i + super_model_i + ' checkpoint=' + str(cp) + ' super checkpoint=' \
                        + str(super_cp) + ' ddim_sampling_steps=' + str(args.ddim_sampling_steps) + ' ddim_eta=' + str(args.ddim_eta), 
                fname = args.save_file, 
                model_str = model_str + f'u0 guidance, rescaler {RESCALER.squeeze()}', 
            )
