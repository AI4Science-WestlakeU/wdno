import math
from pathlib import Path
from random import random
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR  

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from IPython import embed
from tensorboardX import SummaryWriter
import datetime
from collections import namedtuple
from functools import partial

from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse

from ddpm_burgers.model_utils import linear_beta_schedule, cosine_beta_schedule, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, extract
from ddpm_burgers.wavelet_utils import get_wt_T
from wave_trans import coef_to_tensor, tensor_to_coef


# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# gaussian diffusion trainer class


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length, # define sampling size
        # wavelet
        is_wavelet = True,
        pad_mode = None,
        wave_type = None,
        padded_shape=None,
        ori_shape = torch.tensor([81, 128], device='cuda'),
        # super model
        is_super_model = False,
        upsample_t = 1,
        upsample_x = 1,
        # diffusion
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        loss_layer_weight = 1,
        # condition of diffusion
        is_condition_pad = True,
        is_condition_u0 = False, 
        is_condition_uT = False, 
        is_condition_f = False, 
        train_on_padded_locations=True, # true: mimic faulty behavior. in principle it should be false.
    ):
        # NOTE: perhaps we cannot always normalize the dataset? 
        # May need to fix this problem.
        super().__init__()

        self.is_wavelet = is_wavelet
        self.is_super_model = is_super_model
        self.pad_mode = pad_mode
        self.wave_type = wave_type

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.traj_size = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) \
            or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_prev = F.pad(alphas[:-1], (1, 0), value = 1.)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        self.alphas = alphas.to(torch.float32).clone() # to make compatible with previous trained models
        self.alphas_prev = alphas_prev.to(torch.float32).clone() # to make compatible with previous trained models
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.loss_layer_weight = loss_layer_weight 
        self.upsample_t = upsample_t
        self.upsample_x = upsample_x
        self.is_condition_pad = is_condition_pad
        self.is_condition_u0 = is_condition_u0 # condition on u_{t=0}
        self.is_condition_uT = is_condition_uT # condition on u_{t=T}
        self.is_condition_f = is_condition_f # condition on f
        self.train_on_padded_locations = train_on_padded_locations
        self.padded_shape = padded_shape
        self.ori_shape = ori_shape

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, **kwargs):
        model_output = self.model(x, t, x_self_cond)
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        nablaJ, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        
        if self.objective == 'pred_noise':
            if 'pred_noise' in kwargs and kwargs['pred_noise'] is not None:
                pred_noise = kwargs['pred_noise']
            else:
                pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            
            # guidance
            with torch.enable_grad():
                pred_noise = may_proj_guidance(pred_noise, nablaJ(x_start) * nablaJ_scheduler(t[0].item()))
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, **kwargs):
        preds = self.model_predictions(x, t, x_self_cond, **kwargs)
        x_start = preds.pred_x_start

        x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, preds.pred_noise

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, **kwargs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, **kwargs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start, pred_noise


    def get_guidance_options(self, **kwargs):
        if 'nablaJ' in kwargs and kwargs['nablaJ'] is not None: # guidance
            nabla_J = kwargs['nablaJ']
            assert not self.self_condition, 'self condition not tested with guidance'
        else:
            nabla_J = lambda x: 0
        nablaJ_scheduler = kwargs['J_scheduler'] if ('J_scheduler' in kwargs and kwargs['J_scheduler'] is not None) else lambda t: 1.
        if 'proj_guidance' in kwargs and kwargs['proj_guidance'] is not None:
            may_proj_guidance = kwargs['proj_guidance']
        else:
            # no proj
            # NOTE: well I am not sure what sign nabla_J should take....
            may_proj_guidance = lambda ep, nabla_J: ep + nabla_J
        return nabla_J, nablaJ_scheduler, may_proj_guidance

    def set_condition(self, img, u: torch.Tensor, shape, condition_type):
        if self.is_wavelet:
            if condition_type == 'u0':
                img[:, -1, :u.shape[-2], :shape[-1]] = u[:, :, :shape[-1]]
            elif condition_type == 'uT':
                img[:, -1, -u.shape[-2]:, :shape[-1]] = u[:, :, :shape[-1]]
            elif condition_type == 'f':
                img[:, 4:8, :shape[-2], :shape[-1]] = u[:, :, :shape[-2], :shape[-1]]
            elif condition_type == 'low':
                img[:, 8:16, :shape[-2], :shape[-1]] = u[:, :, :shape[-2], :shape[-1]]
            elif condition_type == 'pad':
                img[:, :-1, shape[-2]:] = 0
                img[:, :, :, shape[-1]:] = 0

        else:
            if condition_type == 'uT':
                if len(u.shape) == 3: # super model
                    img[:, 0, shape[-2]-2:shape[-2], :shape[-1]] = u.squeeze()[:, :, :shape[-1]]
                else:  # base model
                    img[:, 0, shape[-2]-1, :shape[-1]] = u.squeeze()[:, :shape[-1]]
            elif condition_type == 'u0':
                img[:, 0, 0, :shape[-1]] = u.squeeze()[:, :shape[-1]]
            elif condition_type == 'f':
                img[:, 1, :shape[-2]-1, :shape[-1]] = u.squeeze()[:, :shape[-2]-1, :shape[-1]]
            elif condition_type == 'pad':
                img[:, 0, shape[-2]:, :] = 0
                img[:, 1, shape[-2]-1:, :] = 0
                img[:, :, :, shape[-1]:] = 0
            elif condition_type == 'low':
                img[:, 2:4, :shape[-2], :shape[-1]] = u[:, :, :shape[-2], :shape[-1]]
            else:
                assert False


    @torch.no_grad()
    def p_sample_loop(self, shape, **kwargs):
        nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        batch, device = shape[0], self.betas.device
        if not self.is_super_model:
            coef_shape = self.padded_shape
        else:
            coef_shape = [self.padded_shape[kwargs['N_upsample']-1][0] + 1, self.padded_shape[kwargs['N_upsample']-1][1]]# repeat the last timestep due to the odd number of timesteps

        img = torch.randn(shape, device=device)
        x_start = None
        for t in reversed(range(0, self.num_timesteps)):
            # fill u0 into cur sample
            if self.is_condition_pad:
                self.set_condition(img, 0, coef_shape, "pad")
                
            if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                u0 = kwargs['u_init'] 
                self.set_condition(img, u0, coef_shape, 'u0')

            if self.is_condition_uT: # NOTE: uT here means physical time t=T
                if self.is_super_model and not self.is_wavelet:
                    uT = kwargs['u_final'].unsqueeze(1).expand(-1, 2, -1)
                else:
                    uT = kwargs['u_final']  
                self.set_condition(img, uT, coef_shape, 'uT')

            if self.is_condition_f: 
                f = kwargs['f'] 
                self.set_condition(img, f, coef_shape, 'f')
                
            if self.is_super_model:
                low = kwargs['low'] 
                self.set_condition(img, low, coef_shape, 'low')

            self_cond = x_start if self.self_condition else None
            # calculates \hat{u_0} for better guidance calculation
            img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, **kwargs)
            img = img_curr.detach()
            
        if self.is_condition_pad:
            self.set_condition(img, 0, coef_shape, "pad")

        if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
            u0 = kwargs['u_init'] 
            self.set_condition(img, u0, coef_shape, 'u0')

        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            if self.is_super_model and not self.is_wavelet:
                uT = kwargs['u_final'].unsqueeze(1).expand(-1, 2, -1)
            else:
                uT = kwargs['u_final']  
            self.set_condition(img, uT, coef_shape, 'uT')

        if self.is_condition_f: 
            f = kwargs['f'] 
            self.set_condition(img, f, coef_shape, 'f')

        if self.is_super_model:
            low = kwargs['low'] 
            self.set_condition(img, low, coef_shape, 'low')

        img = self.unnormalize(img)
        return img

    # @torch.inference_mode()
    @torch.no_grad()    
    def ddim_sample(self, shape, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device,\
                        self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if not self.is_super_model:
            coef_shape = self.padded_shape
        else:
            coef_shape = [self.padded_shape[kwargs['N_upsample']-1][0] + 1, self.padded_shape[kwargs['N_upsample']-1][1]]# repeat the last timestep due to the odd number of timesteps

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            if self.is_condition_pad:
                self.set_condition(img, 0, coef_shape, "pad")            

            if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                u0 = kwargs['u_init'] 
                self.set_condition(img, u0, coef_shape, 'u0')

            if self.is_condition_uT: # NOTE: uT here means physical time t=T
                if self.is_super_model and not self.is_wavelet:
                    uT = kwargs['u_final'].unsqueeze(1).expand(-1, 2, -1)
                else:
                    uT = kwargs['u_final']  
                self.set_condition(img, uT, coef_shape, 'uT')

            if self.is_condition_f: 
                f = kwargs['f'] 
                self.set_condition(img, f, coef_shape, 'f')

            if self.is_super_model:
                low = kwargs['low'] 
                self.set_condition(img, low, coef_shape, 'low')

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, **kwargs)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
        if self.is_condition_pad:
            self.set_condition(img, 0, coef_shape, "pad")

        if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
            u0 = kwargs['u_init'] 
            self.set_condition(img, u0, coef_shape, 'u0')

        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            if self.is_super_model and not self.is_wavelet:
                uT = kwargs['u_final'].unsqueeze(1).expand(-1, 2, -1)
            else:
                uT = kwargs['u_final']  
            self.set_condition(img, uT, coef_shape, 'uT')

        if self.is_condition_f: 
            f = kwargs['f'] 
            self.set_condition(img, f, coef_shape, 'f')

        if self.is_super_model:
            low = kwargs['low'] 
            self.set_condition(img, low, coef_shape, 'low')

        ret = self.unnormalize(img)
        return ret

    def sample(self, batch_size=16, **kwargs):
        '''
        Kwargs:
            nablaJ: 
                a gradient function returning nablaJ for diffusion guidance. 
                Can use the function get_nablaJ to construct the gradient function.
            J_scheduler: 
                Optional callable, scheduler for J, returns stepsize given t
            proj_guidance:
                Optional callable, postprocess guidance for better diffusion. 
                E.g., project nabla_J to the orthogonal direction of epsilon_theta
            u_init:
                Optional, torch.Tensor of size (batch, Nx). u at time = 0, applies when self.is_condition_u0 == True
        '''
        if self.is_condition_u0:
            assert 'is_condition_u0' not in kwargs, 'specify this value in the model. not during sampling.'
            assert 'u_init' in kwargs and kwargs['u_init'] is not None
        if self.is_condition_uT:
            assert 'is_condition_uT' not in kwargs, 'specify this value in the model. not during sampling.'
            assert 'u_final' in kwargs and kwargs['u_final'] is not None
        if self.is_condition_f:
            assert 'is_condition_f' not in kwargs, 'specify this value in the model. not during sampling.'
            assert 'f' in kwargs and kwargs['f'] is not None
        if self.is_super_model:
            assert 'N_upsample' in kwargs and kwargs['N_upsample'] is not None
            assert 'low' in kwargs and kwargs['low'] is not None

        # determine sampling size
        if not self.is_super_model:
            sample_size = (batch_size, self.channels, *self.traj_size)
        else:
            sample_size = (batch_size, self.channels, *kwargs['low'].shape[-2:])

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(sample_size, **kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
        for i in reversed(range(0, t)):
            self_cond = x_start if self.self_condition else None
            img, x_start, pred_noise = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, nt, nx = x_start.shape
        if self.is_super_model:
            if self.is_wavelet:
                N_downsample = int(math.log2(64 / nx))
            else:
                N_downsample = int(math.log2(128 / nx))
            coef_shape = [self.padded_shape[N_downsample][0] + 1, self.padded_shape[N_downsample][1]]# repeat the last timestep due to the odd number of timesteps
        else:
            coef_shape = self.padded_shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        # 1. BEFORE MODEL_PREDICTION: SET INPUT
        if self.is_condition_pad:
            self.set_condition(x, 0, coef_shape, "pad")

        if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
            if self.is_wavelet :
                self.set_condition(x, x_start[:, -1, :int(nt/2), :], coef_shape, 'u0')
            else:
                self.set_condition(x, x_start[:, 0, 0, :], coef_shape, 'u0')
                if len(x.shape) == 4:
                    pass
                else:
                    raise ValueError('Bad sample shape')
                
        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            if self.is_wavelet :
                self.set_condition(x, x_start[:, -1, int(nt/2):, :], coef_shape, 'uT')
            else:
                if self.is_super_model:
                    self.set_condition(x, x_start[:, 0, coef_shape[0]-2:coef_shape[0], :], coef_shape, 'uT')
                else:
                    self.set_condition(x, x_start[:, 0, coef_shape[0]-1, :], coef_shape, 'uT')

        if self.is_condition_f:
            if self.is_wavelet:
                self.set_condition(x, x_start[:, 4:8], coef_shape, 'f')
            else:
                self.set_condition(x, x_start[:, 1], coef_shape, 'f')
        
        if self.is_super_model:
            # assert self.is_wavelet
            if self.is_wavelet:
                self.set_condition(x, x_start[:, 8:16], coef_shape, 'low')
            else:
                self.set_condition(x, x_start[:, 2:4], coef_shape, 'low')

        # 2. MODEL PREDICTION
        model_out = self.model(x, t, x_self_cond)

        # 3. AFTER MODEL_PREDICTION: SET OUTPUT AND TARGET
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        if self.is_condition_pad:
            self.set_condition(noise, 0, coef_shape, "pad")

        if self.is_condition_u0:
            # not computing loss for the diffused state!
            if self.is_wavelet:
                self.set_condition(noise, torch.zeros_like(x[:, -1, :int(nt/2), :]), coef_shape, 'u0')
            else:
                self.set_condition(noise, torch.zeros_like(x[:, 0, 0, :]), x.shape, 'u0')
        
        if self.is_condition_uT:
            # not computing loss for the diffused state!
            if self.is_wavelet:
                self.set_condition(noise, torch.zeros_like(x[:, -1, int(nt/2):, :]), coef_shape, 'uT')
            else:
                if self.is_super_model:
                    self.set_condition(noise, torch.zeros_like(x[:, 0, :2, :]), coef_shape, 'uT')
                else:
                    self.set_condition(noise, torch.zeros_like(x[:, 0, 0, :]), coef_shape, 'uT')

        if self.is_condition_f:
            # not computing loss for the diffused state!
            if self.is_wavelet:
                self.set_condition(noise, torch.zeros_like(x[:, 4:8]), coef_shape, 'f')
            else:
                self.set_condition(noise, torch.zeros_like(x[:, 1]), coef_shape, 'f')

        if self.is_super_model:
            if self.is_wavelet:
                self.set_condition(noise, torch.zeros_like(x_start[:, 8:16]), coef_shape, 'low')
            else:
                self.set_condition(noise, torch.zeros_like(x_start[:, 2:4]), coef_shape, 'low')

        # 4. COMPUTE LOSS
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = loss * self.loss_layer_weight.to(loss.device)
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, nt, nx, device, traj_size = *img.shape, img.device, self.traj_size
        # assert (nt, nx) == traj_size, f'traj size must be (nt, nx) of ({nt, nx})'
        # diffusion timestep
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

class GaussianDiffusion1D(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
