import torch
import os
import datetime

from ddpm.diffusion_2d import Unet, GaussianDiffusion, Trainer
from ddpm.data_2d import Smoke, Smoke_wave
from video_diffusion_pytorch.video_diffusion_pytorch import Unet3D
from video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D

import argparse


parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--dataset', default='Smoke', type=str,
                    help='dataset to evaluate')
parser.add_argument('--dataset_path', default="./data/2d/", type=str,
                    help='path to dataset')
parser.add_argument('--is_condition_control', default=False, type=eval,
                    help='If condition on control')
parser.add_argument('--is_condition_pad', default=True, type=eval,
                    help='If condition on padded state')

# Wavelet
parser.add_argument('--is_wavelet', default=True, type=eval,
                    help='If learning wavelet coefficients')
parser.add_argument('--is_super_model', default=False, type=eval,
                    help='If training the super resolution model')
parser.add_argument('--wave_type', default='bior1.3', type=str,
                    help='type of wavelet: bior1.3, bior2.2 ...')
parser.add_argument('--pad_mode', default='zero', type=str,
                    help='padding mode for wavelet transform: zero')
parser.add_argument('--N_downsample', default=0, type=int,
                    help='number of times of subsampling for training of super resolution model,\
                         no more than 2')

parser.add_argument('--batch_size', default=6, type=int,
                    help='size of batch of input to use')
parser.add_argument('--train_num_steps', default=200000, type=int,
                    help='total training steps')
parser.add_argument('--results_path', default="./results/train/", type=str,
                    help='folder to save training checkpoints')
parser.add_argument('--exp_id', default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str,
                    help='experiment name')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    print(FLAGS)
    
    # get shape, RESCALER
    if FLAGS.dataset == "Smoke":
        if FLAGS.is_super_model:
            shape, ori_shape = [], []
            for i in range(FLAGS.N_downsample):
                dataset = Smoke_wave(
                    dataset_path=FLAGS.dataset_path,
                    wave_type=FLAGS.wave_type,
                    pad_mode=FLAGS.pad_mode,
                    is_super_model=FLAGS.is_super_model,
                    downsample_type="space" if FLAGS.is_condition_control else "time",
                    N_downsample=i,
                )
                _, shape_i, ori_shape_i, _ = dataset[0]
                shape.append(shape_i)
                ori_shape.append(ori_shape_i)
        else:
            if FLAGS.is_wavelet:
                dataset = Smoke_wave(
                    dataset_path=FLAGS.dataset_path,
                    wave_type=FLAGS.wave_type,
                    pad_mode=FLAGS.pad_mode,
                    is_super_model=FLAGS.is_super_model,
                    N_downsample=0,
                )
            else:
                dataset = Smoke(
                    dataset_path=FLAGS.dataset_path,
                    is_train=True,
                )
            _, shape, ori_shape, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0)

    if FLAGS.is_super_model:
        assert FLAGS.is_wavelet
    if FLAGS.is_wavelet:
        channels = 42
        if FLAGS.is_super_model:
            channels += 40
    else:
        channels = 6

    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = channels,
    )
    print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    os.makedirs(FLAGS.results_path, exist_ok=True)
    FLAGS.results_path = os.path.join(FLAGS.results_path, FLAGS.exp_id)
    print("Saved at: ", FLAGS.results_path)

    diffusion = GaussianDiffusion(
        model,
        RESCALER,
        FLAGS.is_condition_control,
        FLAGS.is_condition_pad,
        FLAGS.is_wavelet,
        FLAGS.is_super_model,
        FLAGS.wave_type,
        FLAGS.pad_mode,
        shape,
        ori_shape,
        image_size = 40 if FLAGS.is_wavelet else 64,
        frames = 24 if FLAGS.is_wavelet else 32,
        timesteps = 1000,           # number of diffusion steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        objective = "pred_noise",
    )

    trainer = Trainer(
        diffusion,
        FLAGS.dataset,
        FLAGS.dataset_path,
        N_downsample = FLAGS.N_downsample,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-3, 
        train_num_steps = FLAGS.train_num_steps, # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 1, # 4000
        results_path = FLAGS.results_path,
        amp = False,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
    )

    trainer.train()
