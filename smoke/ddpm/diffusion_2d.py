import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import datetime
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
import logging
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def get_device():
    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Simulator(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # # time embeddings

        # time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = None),
                block_klass(dim_in, dim_in, time_emb_dim = None),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = None)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = None),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = None),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = None)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time=None, x_self_cond = None):        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()
                
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        return self.final_conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    
# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# clamp a tensor with states, mask and offsets of several boundaries
def asynchronous_clamp(x):
    if x.dim() == 5:
        # batch_size x number_boundary x (3*frames + 3) x 64 x 64
        x[:, :, :-3].clamp_(-1., 1.) # clamp states (pressure and velocity of frames time steps) of number_boundary boundaries
        x[:, :, -3].clamp_(0., 1.) # clamp boundary masks of number_boundary boundaries
        x[:, :, -2:].clamp_(-0.5, 0.5) # clamp boundary offsets of number_boundary boundaries
    elif x.dim() == 4:
        # (batch_size * number_boundary) x (3*frames + 3) x 64 x 64
        x[:, :-3].clamp_(-1., 1.) # clamp states (pressure and velocity of frames time steps) of number_boundary boundaries
        x[:, -3].clamp_(0., 1.) # clamp boundary masks of number_boundary boundaries
        x[:, -2:].clamp_(-0.5, 0.5) # clamp boundary offsets of number_boundary boundaries

    else:
        raise ValueError("invalid x dim, must be 4 or 5")
        
    return x


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        loss_layer_weight,
        is_condition_control,
        is_condition_pad,
        is_wavelet,
        is_super_model,
        wave_type,
        pad_mode,
        padded_shape,
        ori_shape,
        *,
        image_size,
        frames,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        standard_fixed_ratio=0.01, # used in standard sampling
        coeff_ratio = 0.1, # used in standard-alpha sampling
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not (type(self) == GaussianDiffusion)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.loss_layer_weight = loss_layer_weight
        self.is_condition_control = is_condition_control
        self.is_condition_pad = is_condition_pad
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.frames = frames
        self.is_wavelet = is_wavelet
        self.is_super_model = is_super_model
        self.wave_type = wave_type
        self.pad_mode = pad_mode
        self.padded_shape = padded_shape
        self.ori_shape = ori_shape
        self.standard_fixed_ratio = standard_fixed_ratio
        self.coeff_ratio = coeff_ratio


        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
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

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr / snr)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

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
    
    def model_predictions(self, shape, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, design_fn = None, design_guidance = "standard", low=None, init=None, init_u=None):

        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        # guidance
        with torch.enable_grad():
            if design_fn is not None:
                if design_guidance.startswith("standard"):
                    with torch.enable_grad():
                        x_clone = x_start.clone().detach().requires_grad_()
                        g = design_fn(x_clone, low=low, init=init, init_u=init_u)
                    if design_guidance == "standard":
                        grad_final = self.standard_fixed_ratio * g
                    elif design_guidance == "standard-alpha":
                        coeff_schedule = self.coeff_ratio * (self.betas).clone().flip(0)
                        eta = extract(coeff_schedule, t, x.shape)
                        grad_final = eta * g
                    else:
                        raise
                pred_noise = pred_noise + grad_final

        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    

    def p_mean_variance(self, shape, x, t, x_self_cond = None, clip_denoised = True, design_fn = None, design_guidance = "standard", low=None, init=None, init_u=None):
        preds = self.model_predictions(shape, x, t, x_self_cond, design_fn = design_fn, design_guidance = design_guidance, low=low, init=init, init_u=init_u)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.) 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def sample_noise(self, shape, device):
        return torch.randn(shape, device = device)

    @torch.no_grad()
    def p_sample(self, shape, x, t: int, x_self_cond = None, clip_denoised = True, design_fn = None, design_guidance = "standard", low=None, init=None, init_u=None):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"
        """
        b, *_, device = *x.shape, x.device 
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        coeff_design_schedual = self.coeff_ratio * (self.betas).clone().flip(0)
        eta = extract(coeff_design_schedual, batched_times, x.shape)
        p_mean_variance = self.p_mean_variance(shape, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, design_fn = design_fn, design_guidance = design_guidance, low=low, init=init, init_u=init_u)

        if "recurrence" not in design_guidance:
            model_mean, _, model_log_variance, x_start = p_mean_variance
            noise = self.sample_noise(model_mean.shape, device) if t > 0 else 0
            pred = model_mean + (0.5 * model_log_variance).exp() * noise

            return pred, x_start
            

    @torch.no_grad()
    def p_sample_loop(self, shape, N_upsample=0, design_fn=None, design_guidance="standard", return_all_timesteps=None, init=None, init_u=None, control=None, low=None, device=None):
        b, f, c, h, w = shape 
        batch, device = shape[0], self.betas.device
        if not self.is_super_model:
            coef_shape = self.padded_shape
        else:
            if self.is_condition_control:
                coef_shape = [self.padded_shape[N_upsample][0], self.padded_shape[N_upsample][1]+2, self.padded_shape[N_upsample][2]+2]
            else:
                coef_shape = [self.padded_shape[N_upsample][0]+2, self.padded_shape[N_upsample][1], self.padded_shape[N_upsample][2]]

        # self.betas = self.betas.to(device)
        noise_state = self.sample_noise([b, f, c, h, w], device) 
        # condition
        assert init is not None
        init = init.to(device)
        if self.is_wavelet:
            noise_state[:, :, -2] = init
        else:
            noise_state[:, 0, 0] = init
        if self.is_condition_control:
            control = control.to(device)
            if self.is_wavelet:
                noise_state[:, :, 24:40] = control
            else:
                noise_state[:, :, 3:5] = control
        if self.is_condition_pad:
            if self.is_wavelet:
                noise_state[:, coef_shape[-3]:, :-2] = 0
                noise_state[:, coef_shape[-3]:, -1] = 0
                noise_state[:, :, :-1, coef_shape[-2]:] = 0
                noise_state[:, :, :-1, :, coef_shape[-1]:] = 0
        if self.is_super_model:
            low = low.to(device)
            noise_state[:, :, 40:80] = low

        x = noise_state
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(shape, x, t, self_cond, design_fn=design_fn, design_guidance=design_guidance, low=low, init=init, init_u=init_u)
            if self.is_wavelet:
                x[:, :, -2] = init
            else:
                x[:, 0, 0] = init
            if self.is_condition_control:
                if self.is_wavelet:
                    x[:, :, 24:40] = control
                else:
                    x[:, :, 3:5] = control
            if self.is_condition_pad:
                if self.is_wavelet:
                    x[:, coef_shape[-3]:, :-2] = 0
                    x[:, coef_shape[-3]:, -1] = 0
                    x[:, :, :-1, coef_shape[-2]:] = 0
                    x[:, :, :-1, :, coef_shape[-1]:] = 0
            if self.is_super_model:
                x[:, :, 40:80] = low
            final_result = x

        return final_result

    @torch.no_grad()
    def ddim_sample(self, shape, N_upsample=0, design_fn=None, design_guidance="standard", init=None, init_u=None, control=None, low=None, device=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        if not self.is_super_model:
            coef_shape = self.padded_shape
        else:
            if self.is_condition_control:
                coef_shape = [self.padded_shape[N_upsample][0], self.padded_shape[N_upsample][1]+2, self.padded_shape[N_upsample][2]+2]
            else:
                coef_shape = [self.padded_shape[N_upsample][0]+2, self.padded_shape[N_upsample][1], self.padded_shape[N_upsample][2]]
            
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        
        # condition
        init = init.to(device)
        if self.is_wavelet:
            img[:, :, -2] = init
        else:
            img[:, 0, 0] = init
        if self.is_condition_control:
            control = control.to(device)
            if self.is_wavelet:
                img[:, :, 24:40] = control
            else:
                img[:, :, 3:5] = control
        if self.is_condition_pad:
            if self.is_wavelet:
                img[:, coef_shape[-3]:, :-2] = 0
                img[:, coef_shape[-3]:, -1] = 0
                img[:, :, :-1, coef_shape[-2]:] = 0
                img[:, :, :-1, :, coef_shape[-1]:] = 0
        if self.is_super_model:
            low = low.to(device)
            img[:, :, 40:80] = low

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(shape, img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, \
                                            design_fn = design_fn, design_guidance = design_guidance, init=init, init_u=init_u, low=low)

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

            if self.is_wavelet:
                img[:, :, -2] = init
            else:
                img[:, 0, 0] = init
            if self.is_condition_control:
                if self.is_wavelet:
                    img[:, :, 24:40] = control
                else:
                    img[:, :, 3:5] = control
            if self.is_condition_pad:
                if self.is_wavelet:
                    img[:, coef_shape[-3]:, :-2] = 0
                    img[:, coef_shape[-3]:, -1] = 0
                    img[:, :, :-1, coef_shape[-2]:] = 0
                    img[:, :, :-1, :, coef_shape[-1]:] = 0
            if self.is_super_model:
                img[:, :, 40:80] = low

        ret = img 

        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, N_upsample=0, design_fn = None, design_guidance="standard", init=None, init_u=None, control=None, low=None, device = None):
        if not self.is_super_model:
            image_size, channels, frames = self.image_size, self.channels, self.frames
        else:
            image_size, channels, frames = low.shape[-1], self.channels, low.shape[1]
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        assert batch_size == init.shape[0]

        if not self.is_super_model:
            sample_size = (batch_size, frames, channels, image_size, image_size)
        else:
            sample_size = (batch_size, low.shape[1], channels, low.shape[-2], low.shape[-1])
        return sample_fn(sample_size, N_upsample, design_fn, design_guidance, init=init, init_u=init_u, control=control, low=low, device = device)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )        


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, state_start, t, noise = None):
        b, f, c, h, w = state_start.shape 
        if self.is_super_model:
            if self.is_condition_control: # simulation, downsample space
                N_downsample = int(math.log2(40 / w))
                coef_shape = [self.padded_shape[N_downsample][0], self.padded_shape[N_downsample][1]+2, self.padded_shape[N_downsample][2]+2]
            else: # control, downsample time
                N_downsample = int(math.log2(24 / f))
                coef_shape = [self.padded_shape[N_downsample][0]+2, self.padded_shape[N_downsample][1], self.padded_shape[N_downsample][2]]
        else:
            coef_shape = self.padded_shape
            
        noise_state = default(noise, lambda: torch.randn_like(state_start))
        
        # noisy sample
        state = self.q_sample(x_start = state_start, t = t, noise = noise_state)
        
        # condition on initial state (and reward)
        # print("conditional ... ")
        
        if self.is_wavelet: # condition on initial density
            state[:, :, -2] = state_start[:, :, -2]
            noise_state[:, :, -2] = torch.zeros_like(noise_state[:, :, -2])
        else:
            state[:, 0, 0] = state_start[:, 0, 0]
            noise_state[:, 0, 0] = torch.zeros_like(noise_state[:, 0, 0])
        if self.is_condition_control:
            if self.is_wavelet:
                state[:, :, 24:40] = state_start[:, :, 24:40]
                noise_state[:, :, 24:40] = torch.zeros_like(noise_state[:, :, 24:40])
            else:
                state[:, :, 3:5] = state_start[:, :, 3:5]
                noise_state[:, :, 3:5] = torch.zeros_like(noise_state[:, :, 3:5])
        if self.is_condition_pad:
            if self.is_wavelet:
                state[:, coef_shape[-3]:, :-2] = 0
                state[:, coef_shape[-3]:, -1] = 0
                state[:, :, :-1, coef_shape[-2]:] = 0
                state[:, :, :-1, :, coef_shape[-1]:] = 0
                noise_state[:, coef_shape[-3]:, :-2] = 0
                noise_state[:, coef_shape[-3]:, -1] = 0
                noise_state[:, :, :-1, coef_shape[-2]:] = 0
                noise_state[:, :, :-1, :, coef_shape[-1]:] = 0
        if self.is_super_model:
            state[:, :, 40:80] = state_start[:, :, 40:80]
            noise_state[:, :, 40:80] = torch.zeros_like(noise_state[:, :, 40:80])
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(state, t).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(state, t, x_self_cond)
        
        loss = self.loss_fn(model_out, noise_state, reduction = 'mean')
        # loss = self.loss_fn(model_out, noise_state, reduction = 'none')
        # loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * self.loss_layer_weight.to(loss.device)

        return loss.mean()

    def forward(self, state, *args, **kwargs):
        #pdb.set_trace()
        b, f, c, h, w, device, img_size, = *state.shape, state.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(state, t, *args, **kwargs)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        dataset_path,
        *,
        N_downsample = 0,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_path = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        is_schedule = True,
        resume = False,
        resume_step = 0,
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp
        self.dataset = dataset

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        from ddpm.data_2d import Smoke, Smoke_wave, SuperDataLoader
        if dataset == "Smoke":
            if self.model.is_wavelet:
                if not self.model.is_super_model:
                    self.ds = Smoke_wave(
                        dataset_path,
                        self.model.wave_type,
                        self.model.pad_mode,
                        is_super_model=self.model.is_super_model,
                        N_downsample=0,
                    )
                else:
                    self.ds = []
                    for i in range(N_downsample):
                        self.ds.append(Smoke_wave(
                            dataset_path,
                            self.model.wave_type,
                            self.model.pad_mode,
                            is_super_model=self.model.is_super_model,
                            downsample_type="space" if self.model.is_condition_control else "time",
                            N_downsample=i,
                        ))
            else:
                self.ds = Smoke(
                    dataset_path,
                    is_train=True,
                )
        else:
            assert False

        if not self.model.is_super_model:
            dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 16)
        else:
            dl = SuperDataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.resume = resume
        self.resume_step = resume_step
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        if is_schedule == True:
            self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=[50000, 150000, 300000], gamma=0.1)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        # if self.resume:
        #     self.load(self.resume_step // self.save_and_sample_every)
        #     self.step = self.resume_step
        
        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_path / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        print("model path: ", str(self.results_path / f'model-{milestone}.pt'))
        data = torch.load(str(self.results_path / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
#         model_state_dict = data['state_dict']

#         # Now you can use the model_state_dict for various purposes, such as loading it into a model
#         # For example, if you have a model instance:
#         # your_model.load_state_dict(model_state_dict)

#         # Accessing keys in the state_dict
#         for key, value in model_state_dict.items():
#             print(f"{key}: {value.shape}")
    
        model.load_state_dict(data['model'])
        print("model loaded: ", str(self.results_path / f'model-{milestone}.pt'))
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        # current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = os.path.join(self.results_path, "info.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    state, _, _, _ = next(self.dl)
                    state = state.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(state)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.step != 0 and self.step % 10 == 0:
                    pbar.set_description(f'loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                    logging.info(f'step: {self.step}, loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    # if True:
                        self.ema.ema_model.eval()

                        milestone = self.step // self.save_and_sample_every
                        
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')