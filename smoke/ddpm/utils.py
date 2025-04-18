import torch
from torch.utils.data import Sampler
import numpy as np
import os, sys
from datetime import datetime
from copy import deepcopy
from collections import deque
from numbers import Number

from video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
from ddpm.diffusion_2d import GaussianDiffusion, Trainer

def load_data(args):
    from ddpm.data_2d import Smoke, Smoke_wave
    # get shape, RESCALER
    if args.dataset == "Smoke":
        if args.is_wavelet:
            dataset = Smoke_wave(
                dataset_path=args.dataset_path,
                wave_type=args.wave_type,
                pad_mode=args.pad_mode,
                is_super_model=args.is_super_model,
                N_downsample=0,
            )
            _, shape_init, ori_shape_init, _ = dataset[0]
            if not args.is_super_model:
                shape = shape_init
                ori_shape = ori_shape_init
            else:
                shape, ori_shape = [], []
                shape.append(shape_init)
                ori_shape.append(ori_shape_init)
                for i in range(args.upsample):
                    if not args.is_condition_control:
                        shape.append([2*shape[-1][-3]-2, shape[-1][-2], shape[-1][-1]])
                        ori_shape.append([2*ori_shape[-1][-3], ori_shape[-1][-2], ori_shape[-1][-1]])
                    else:
                        shape.append([shape[-1][-3], 2*shape[-1][-2]-2, 2*shape[-1][-1]-2])
                        ori_shape.append([ori_shape[-1][-3], 2*ori_shape[-1][-2], 2*ori_shape[-1][-1]])
        else:
            dataset = Smoke(
                dataset_path=args.dataset_path,
                is_train=True,
            )
            _, shape, ori_shape, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0).to(args.device)

    dataset = Smoke(
        dataset_path=args.dataset_path,
        is_train=False,
        test_mode='control' if not args.is_condition_control else 'simulation', # control / simulation
        upsample=args.is_super_model # change Ndata
    ) # the super resolution data, super_nt=8*base_nt, super_nx=2*base_nx, not rescaled
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    print("number of batch in test_loader: ", len(test_loader))
    return test_loader, shape, ori_shape, RESCALER


def load_ddpm_super_model(args, shape, ori_shape, RESCALER):
    if args.is_wavelet:
        channels = 82
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=channels
    )
    print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(args.device)
    diffusion = GaussianDiffusion(
        model,
        RESCALER,
        args.is_condition_control,
        args.is_condition_pad,
        args.is_wavelet,
        True, # is_super_model
        args.wave_type,
        args.pad_mode,
        shape,
        ori_shape,
        image_size = args.image_size if not args.is_wavelet else 40,
        frames = 32 if not args.is_wavelet else 24,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
    )
    diffusion.eval()
    # load trainer
    trainer = Trainer(
        diffusion,
        dataset = args.dataset,
        dataset_path = args.dataset_path,
        results_path = os.path.join(args.super_diffusion_model_path, args.super_exp_id), 
        N_downsample = 1, # not used
        amp = False,      # turn on mixed precision
    )
    trainer.load(args.super_diffusion_checkpoint) 
    return diffusion, trainer.device


def load_ddpm_base_model(args, shape, ori_shape, RESCALER):
    if args.is_wavelet:
        channels = 42
    else:
        channels = 6
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=channels
    )
    print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(args.device)
    diffusion = GaussianDiffusion(
        model,
        RESCALER,
        args.is_condition_control,
        args.is_condition_pad,
        args.is_wavelet,
        False, # is_super_model
        args.wave_type,
        args.pad_mode,
        shape,
        ori_shape,
        image_size = args.image_size if not args.is_wavelet else 40,
        frames = 32 if not args.is_wavelet else 24,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
    )
    diffusion.eval()
    # load trainer
    trainer = Trainer(
        diffusion,
        dataset = args.dataset,
        dataset_path = args.dataset_path,
        results_path = os.path.join(args.diffusion_model_path, args.exp_id), 
        amp = False,   # turn on mixed precision
    )
    trainer.load(args.diffusion_checkpoint) 
    return diffusion, trainer.device


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))

def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string

p = Printer(n_digits=6)