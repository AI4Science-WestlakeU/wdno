import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import grad

import numpy as np
import sys, os
from pathlib import Path
from collections import OrderedDict
import datetime, time
from numbers import Number
import math as math_package
import multiprocess as mp

import pywt, ptwt
from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse

from ddpm.data_2d import Smoke, Smoke_wave
from ddpm.utils import load_ddpm_base_model, load_ddpm_super_model, load_data
from ddpm.wave_utils import upsample_coef
from wave_trans_2d import tensor_to_coef, coef_to_tensor
from dataset.evaluate_solver import *

import argparse

from IPython import embed


def guidance_fn(x, args, shape, ori_shape, RESCALER, w_energy=0, w_init=0, low=None, init=None, init_u=None):
    '''
    low, init: rescaled
    init_u: not rescaled
    '''
    x = x * RESCALER
    if args.is_wavelet:
        wavelet = pywt.Wavelet(args.wave_type)
        if low != None:
            low = low * RESCALER[:, :, 40:80]
        coef = tensor_to_coef(x[:,:,:-2].permute(0,2,1,3,4), shape) # no smoke out
        state = ptwt.waverec3(coef, wavelet)[:, :ori_shape[0], :ori_shape[1], :ori_shape[2]]\
                                        .reshape(-1, 5, ori_shape[0], ori_shape[1], ori_shape[2])
        ifm1d = DWT1DInverse(mode=args.pad_mode, wave=args.wave_type).to(x.device)
        Yl_s = x[:,:shape[0],-1,:int(40/2)].mean((-2,-1)).unsqueeze(1)
        Yh_s = [x[:,:shape[0],-1,int(40/2):].mean((-2,-1)).unsqueeze(1)]
        smoke_out = ifm1d((Yl_s, Yh_s))[:,0] # 25, 32

        guidance_success = smoke_out[:,ori_shape[0]-1].sum()
        guidance_init = (state[:,0,0]-init_u.to(state.device)).square().mean((-1,-2)).sum()
        guidance_energy = state[:,3:5].square().mean((1,2,3,4)).sum()
        if args.is_condition_control:
            guidance = w_init * guidance_init
        else:
            guidance = -guidance_success + w_energy * guidance_energy + w_init * guidance_init

    else:
        state = x
        if args.is_condition_control:
            guidance = 0
        else:
            guidance_success = state[:,-1,-1].mean((-1,-2)).sum()
            guidance_energy = state[:,3:5].square().mean((1,2,3,4)).sum()
            guidance = -guidance_success + w_energy * guidance_energy

    grad_x = grad(guidance, x, grad_outputs=torch.ones_like(guidance))[0]
    return grad_x
     

def load_model(args, shape, ori_shape, RESCALER):
    w_energy = args.w_energy
    w_init = args.w_init
    if args.is_super_model:
        assert args.is_wavelet
        diffusion, device = load_ddpm_base_model(args, shape[0], ori_shape[0], RESCALER[:, :, 40:])
        diffusion_super, device = load_ddpm_super_model(args, shape, ori_shape, RESCALER)
    else:
        diffusion, device = load_ddpm_base_model(args, shape, ori_shape, RESCALER)
    RESCALER = RESCALER.to(device)
    
    # define design function
    def design_fn(x, low=None, init=None, init_u=None):
        if args.is_super_model:
            if low != None:
                upsample = int(math_package.log2(low.shape[-1] / 40))
                grad_x = guidance_fn(x, args, shape[upsample], ori_shape[upsample], RESCALER, w_energy=w_energy, w_init=w_init, low=low, init=init, init_u=init_u)
            else:
                grad_x = guidance_fn(x, args, shape[0], ori_shape[0], RESCALER[:, :, 40:], w_energy=w_energy, w_init=w_init, low=low, init=init, init_u=init_u)
        else:
            grad_x = guidance_fn(x, args, shape, ori_shape, RESCALER, w_energy=w_energy, w_init=w_init, low=low, init=init, init_u=init_u)
    
        return grad_x
    
    if args.is_super_model:
        return [diffusion, diffusion_super], design_fn
    else:
        return [diffusion], design_fn


class InferencePipeline(object):
    def __init__(
        self,
        model,
        args=None,
        RESCALER=1,
        results_path=None,
        args_general=None,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.results_path = results_path
        self.args_general = args_general
        self.is_wavelet = args_general.is_wavelet
        self.is_condition_control = args_general.is_condition_control
        self.image_size = self.args_general.image_size
        self.device = self.args_general.device
        self.upsample = args_general.upsample
        self.RESCALER = RESCALER
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)


    def run_base_model(self, state, wave_init, wave_control):
        output = self.model[0].sample(
            batch_size = state.shape[0],
            design_fn=self.args["design_fn"],
            design_guidance=self.args["design_guidance"],
            low=None, 
            init=wave_init/self.RESCALER[:,:,-2] if self.is_wavelet else state[:,0,0]/self.RESCALER[:,0,0], 
            init_u=state[:,0,0], # only used in guidance if is_wavelet
            control=wave_control/self.RESCALER[:,:,24:40] if self.is_wavelet else state[:,:,3:5]/self.RESCALER[:,:,3:5]
        )
        if not self.is_wavelet:
            output = output * self.RESCALER
            output[:,:,-1] = output[:,:,-1].mean((-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)
            return output
        else: # inverse wavelet transform
            shape, ori_shape = self.model[0].padded_shape, self.model[0].ori_shape
            wave_output = output * self.RESCALER
            coef = tensor_to_coef(wave_output[:,:,:-2].permute(0,2,1,3,4), shape)
            ori_output = ptwt.waverec3(coef, pywt.Wavelet(self.model[0].wave_type))[:, :ori_shape[0], :ori_shape[1], :ori_shape[2]]\
                        .reshape(-1, 5, ori_shape[0], ori_shape[1], ori_shape[2]).permute(0,2,1,3,4)

            ifm1d = DWT1DInverse(mode=self.model[0].pad_mode, wave=self.model[0].wave_type).to(output.device)
            Yl_s = wave_output[:,:shape[0],-1,:int(40/2)].mean((-2,-1)).unsqueeze(1)
            Yh_s = [wave_output[:,:shape[0],-1,int(40/2):].mean((-2,-1)).unsqueeze(1)]
            smoke_out = ifm1d((Yl_s, Yh_s))[:,0] # 25, 32
            smoke_out = smoke_out.reshape(smoke_out.shape[0], smoke_out.shape[1], 1, 1, 1)\
                        .expand(-1, -1, -1, 64, 64)

            ori_output = torch.cat((ori_output, smoke_out), dim=2)
            return ori_output
 

    def run_super_model(self, state, state_ori, wave_init, wave_control):
        shape, ori_shape = self.model[1].padded_shape, self.model[1].ori_shape
        assert self.args_general.is_condition_control
        upsample_type = 'space'
        rets, wave_outputs = [], []

        # base resolution
        wave_output = self.model[0].sample(
            batch_size = state.shape[0],
            design_fn=self.args["design_fn"],
            design_guidance=self.args["design_guidance"],
            low=None, 
            init=wave_init/self.RESCALER[:,:,-2], 
            init_u=state[:,0,0], # only used in guidance if is_wavelet
            control=wave_control/self.RESCALER[:,:,24:40]
        ) # rescaled
        wave_outputs.append(wave_output) # rescaled
        coef = tensor_to_coef(wave_output[:,:,:40].permute(0,2,1,3,4), shape[0])
        rets.append(coef_to_tensor(coef).reshape(-1,5,8,*shape[0]).reshape(-1,40,*shape[0]).permute(0,2,1,3,4)) # rescaled

        for i in range(1, self.upsample+1): # upsampling
            pad_t, pad_x, nx = 24, 40*2**i, self.model[1].padded_shape[i][-2]
            state = state_ori[:, ::2**(3-i)] if not self.args_general.is_condition_control else state_ori[:,:,:,::2**(1-i),::2**(1-i)]
            xfm2d = DWTForward(J=1, mode=self.model[1].pad_mode, wave=self.model[1].wave_type).to(state.device)
            Yl0, Yh0 = xfm2d(state[:, 0, [0]])
            wave_init =torch.cat((Yl0, Yh0[0][:,0]), dim=1)
            wave_init = wave_init.unsqueeze(1).expand(wave_init.shape[0], int(pad_t/4), 4, nx, nx)\
                                .reshape(-1, pad_t, nx, nx) # (repeated b, W_d0, pad_x, pad_x)
            wave_init = F.pad(wave_init, (0, pad_x - nx, 0, pad_x - nx), 'constant', 0)
            wave_control = ptwt.wavedec3(state[:,:,3:5].permute(0,2,1,3,4).reshape(-1, state.shape[1], state.shape[3], state.shape[4]),\
                            pywt.Wavelet(self.model[1].wave_type), mode=self.model[1].pad_mode, level=1) 
            wave_control = coef_to_tensor(wave_control)
            # repeat boundary
            if not self.args_general.is_condition_control:
                wave_control = torch.cat((wave_control[:,:,:1], wave_control[:,:,:shape[i][0]], wave_control[:,:,[shape[i][0]-1]]), dim=2)
            else:
                wave_control = F.pad(wave_control.reshape(-1, *wave_control.shape[-3:]), (1, 1, 1, 1), mode='replicate')
            wave_control = F.pad(wave_control, (0, pad_x-wave_control.shape[-1], 0, pad_x-wave_control.shape[-2], 0, pad_t-wave_control.shape[-3]),\
                            'constant', 0).reshape(-1,2,8,pad_t,pad_x,pad_x).reshape(-1,16,pad_t,pad_x,pad_x).permute(0,2,1,3,4)
            sampled_coef = upsample_coef(rets[-1], shape[i], type=upsample_type)
            sampled_coef = nn.functional.pad(sampled_coef, (0, pad_x - sampled_coef.shape[-1], 0, \
                            pad_x - sampled_coef.shape[-2], 0, 0, 0, pad_t - sampled_coef.shape[-4]), 'constant', 0)

            wave_output = self.model[1].sample(
                batch_size = state.shape[0],
                design_fn=self.args["design_fn"],
                design_guidance=self.args["design_guidance"],
                N_upsample=i,
                low=sampled_coef, 
                init=wave_init/self.RESCALER[:,:,-2], 
                init_u=state[:,0,0], # only used in guidance if is_wavelet
                control=wave_control/self.RESCALER[:,:,24:40]
            )
            wave_outputs.append(wave_output)
            coef = tensor_to_coef(wave_output[:,:,:40].permute(0,2,1,3,4), shape[i], upsample_type=upsample_type) # upsample_type!!
            rets.append(coef_to_tensor(coef).reshape(-1,5,8,*shape[i]).reshape(-1,40,*shape[i]).permute(0,2,1,3,4))

        # inverse wavelet transform
        ori_rets = []
        for i in range(len(rets)):
            wave_output = wave_outputs[i][:,:,:40] * self.RESCALER[:,:,:40]
            if i == 0:
                coef = tensor_to_coef(wave_output.permute(0,2,1,3,4), shape[i])
            else:
                coef = tensor_to_coef(wave_output.permute(0,2,1,3,4), shape[i], upsample_type=upsample_type) # upsample_type!!
            ori_output = ptwt.waverec3(coef, pywt.Wavelet(self.model[0].wave_type))\
                        [:, :ori_shape[i][0], :ori_shape[i][1], :ori_shape[i][2]]\
                        .reshape(-1, 5, ori_shape[i][0], ori_shape[i][1], ori_shape[i][2]).permute(0,2,1,3,4)
            ifm1d = DWT1DInverse(mode=self.model[0].pad_mode, wave=self.model[0].wave_type).to(wave_output.device)
            wave_smoke_out = wave_outputs[i][:,:,[-1]] * self.RESCALER[:,:,[-1]]
            Yl_s = wave_smoke_out[:,:shape[i][0],-1,:int(wave_output.shape[-2]/2)].mean((-2,-1)).unsqueeze(1)
            Yh_s = [wave_smoke_out[:,:shape[i][0],-1,int(wave_output.shape[-2]/2):].mean((-2,-1)).unsqueeze(1)]
            smoke_out = ifm1d((Yl_s, Yh_s))[:,0] # 25, 32
            smoke_out = smoke_out.reshape(smoke_out.shape[0], smoke_out.shape[1], 1, 1, 1)\
                        .expand(-1, -1, -1, ori_shape[i][1], ori_shape[i][2])
            ori_output = torch.cat((ori_output, smoke_out), dim=2)
            ori_rets.append(ori_output)
        return ori_rets


    def run_model(self, state):
        '''
        state: not rescaled
        '''
        state_ori = state.to(self.device)
        state = state_ori[:, ::8] if not self.args_general.is_condition_control else state_ori[:, :, :, ::2, ::2] # base resolution

        if self.is_wavelet: # prepare wavelet conditions
            pad_t, pad_x, nt, nx = 24, 40, self.model[0].padded_shape[-3], self.model[0].padded_shape[-2]
            xfm2d = DWTForward(J=1, mode=self.model[0].pad_mode, wave=self.model[0].wave_type).to(state.device)
            Yl0, Yh0 = xfm2d(state[:, 0, [0]])
            wave_init = torch.cat((Yl0, Yh0[0][:,0]), dim=1)
            wave_init = wave_init.unsqueeze(2).expand(wave_init.shape[0], 4, int(pad_t/4), nx, nx)\
                                        .reshape(-1, pad_t, nx, nx) # (repeated b, W_d0, pad_x, pad_x)
            wave_init = F.pad(wave_init, (0, pad_x - nx, 0, pad_x - nx), 'constant', 0)
            wave_control = ptwt.wavedec3(state[:,:,3:5].permute(0,2,1,3,4).reshape(-1, state.shape[1], state.shape[3], state.shape[4]),\
                            pywt.Wavelet(self.model[0].wave_type), mode=self.model[0].pad_mode, level=1)
            wave_control = coef_to_tensor(wave_control)
            wave_control = F.pad(wave_control, (0, pad_x-wave_control.shape[-1], 0, pad_x-wave_control.shape[-2], 0, pad_t-wave_control.shape[-3]),\
                                'constant', 0).reshape(-1,2,8,pad_t,pad_x,pad_x).reshape(-1,16,pad_t,pad_x,pad_x).permute(0,2,1,3,4)
        
        if len(self.model) == 1: # base resolution
            return self.run_base_model(state, wave_init, wave_control)
        else: # super resolution
            return self.run_super_model(state, state_ori, wave_init, wave_control)

    def run(self, dataloader):
        preds = []
        J_totals, J_targets, J_energys, mses, n_l2s = {}, {}, {}, {}, {}
        for i in range(self.upsample+1): # super resolution
            J_totals[i], J_targets[i], J_energys[i], mses[i], n_l2s[i] = [], [], [], [], []
        for i, data in enumerate(dataloader):
            print(f"Batch No.{i}")
            state, shape, ori_shape, sim_id = data # shape and ori_shape are not neeeded for baselines
            pred = self.run_model(state)
            preds.append(pred)
            if not self.args_general.is_super_model:
                J_total, J_target, J_energy, mse, n_l2 = self.multi_evaluate(pred, state, plot=False)
                J_totals[0].append(J_total)
                J_targets[0].append(J_target)
                J_energys[0].append(J_energy)
                mses[0].append(mse)
                n_l2s[0].append(n_l2)
            else:
                for i in range(len(pred)):
                    print('Number of upsampling times:', i)
                    J_total, J_target, J_energy, mse, n_l2 = self.multi_evaluate(pred[i], state, plot=False)
                    J_totals[i].append(J_total)
                    J_targets[i].append(J_target)
                    J_energys[i].append(J_energy)
                    mses[i].append(mse)
                    n_l2s[i].append(n_l2)
            
        print("Final results!")
        for i in range(self.upsample+1):
            print(f"Number of upsampling times: {i}")
            print(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: " +
                    f"{np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}")

        self.save_results(J_totals, J_targets, J_energys, mses, n_l2s)


    def save_results(self, J_totals, J_targets, J_energys, mses, n_l2s):
        save_file = 'results_sim.txt' if self.args_general.is_condition_control else 'results.txt'
        results_path = self.results_path
        with open(os.path.join(results_path, save_file), 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\n')
            f.write(str(self.args_general)+'\n')
            for i in range(self.upsample+1):
                f.write(f"Number of upsampling times: {i}\n")
                f.write(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: " +
                        f"{np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}\n")
            f.write("-----------------------------------------------------------------------------------------\n")
    

    def per_evaluate(self, sim, eval_no, pred, data, output_queue):
        '''
        eval_no: No of multi-process
        pred: torch.Tensor, [nt, 1, nx, nx]
        '''
        if not self.args_general.is_condition_control: # control        
            # print(f'Evaluate No.{eval_no}')
            init_velocity = init_velocity_() # 1, 128, 128, 2
            init_density = data[0,0,:,:] # nx, nx
            c1 = pred[:,3,:,:] # nt, nx, nx
            c2 = pred[:,4,:,:] # nt, nx, nx
            per_solver_out = solver(sim, init_velocity, init_density, c1, c2)
            # print(f'Evaluate No.{eval_no} down!')

        try:
            output_queue.put({eval_no:per_solver_out})
            # print(f"Queue Put down {eval_no}")
        except Exception as e:
            print(f"Error in process {eval_no}: {e}")

    def multi_evaluate_control(self, pred, data, plot=False):
        print("Start solving...")
        start = time.time()
        pool_num = pred.shape[0]
        solver_out = np.zeros((pred.shape[0],256,6,128,128), dtype=float)
        pred_ = pred.detach().cpu().numpy().copy()
        data_ = data.detach().cpu().numpy().copy()
        pred_[:,:,3:5,8:56,8:56] = 0 # indirect control
        sim = init_sim()
        output_queue = mp.Queue()

        processes = []
        args_list = [(sim, i, pred_[i,:,:,:,:].copy(), data_[i,:,:,:,:].copy(),output_queue) for i in range(pool_num)]
        for args in args_list:
            process = mp.Process(target=self.per_evaluate, args=args)
            processes.append(process)
            process.start()
        # print(f"Total processes started: {len(processes)}")

        multi_results_list = []
        for i in range(len(processes)):
            multi_results_list.append(output_queue.get())
        multi_results_sorted = dict()
        for eval_no in range(len(processes)): # process no.
            for item in multi_results_list: 
                if list(item.keys())[0] == eval_no:
                    multi_results_sorted[f'{eval_no}']=list(item.values())[0]
                    continue

        for process in processes:
            process.join()
            # print(f"{process} down!")

        for i in range(len(multi_results_sorted)):
            solver_out[i,:,0,:,:] = multi_results_sorted[f'{i}'][0] # density
            # solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][1] # zero_density
            solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,0] # vel_x
            solver_out[i,:,2,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,1] # vel_x
            solver_out[i,:,3,:,:] = multi_results_sorted[f'{i}'][3] # control_x
            solver_out[i,:,4,:,:] = multi_results_sorted[f'{i}'][4] # control_y
            solver_out[i,:,5,:,:] = multi_results_sorted[f'{i}'][5] # smoke_portion

        if plot:
            print("Start Generating GIFs...")
            """
            Generate GIF
            """
            for i in range(10,50):
                gif_density(solver_out[i,:,0,:,:],zero=False,name=f'WDNO{i}')
            
        return solver_out


    def multi_evaluate(self, pred, data, plot=False):
        '''
        pred: torch.Tensor, [B, nt, 6, nx, nx] 
        data: torch.Tensor, control: [B, 256, 1, 64, 64], simulation: [B, 32, 1, 128, 128]
        '''
        pred[:, 0, 0] = data[:, 0, 0, ::int(data.shape[-1]/pred.shape[-1]), ::int(data.shape[-1]/pred.shape[-1])] # initial condition
        if not self.args_general.is_condition_control: # control 
            start = time.time()
            solver_out = self.multi_evaluate_control(pred, data, plot=plot)
            end = time.time()
            print(f"Time cost: {end-start}")

            data_super = torch.tensor(solver_out, device=pred.device)[:, :, :, ::2, ::2] # no space super resolution
            data_current = data_super[:, ::int(data_super.shape[1]/pred.shape[1])]
            data_base = data_super[:, ::8]
            pred_current = pred
            if self.args_general.is_super_model:
                pred_base = pred[:, ::int(pred.shape[1]/32)]
                pred_super = F.interpolate(pred.permute(0,2,3,4,1).reshape(pred.shape[0],-1,pred.shape[1]), size=(256), mode='linear', align_corners=False)\
                            .permute(0,2,1).reshape(pred.shape[0],256,pred.shape[2],pred.shape[3],pred.shape[4])
                pred_super2 = F.interpolate(pred.permute(0,2,3,4,1).reshape(pred.shape[0],-1,pred.shape[1]), size=(256), mode='nearest')\
                            .permute(0,2,1).reshape(pred.shape[0],256,pred.shape[2],pred.shape[3],pred.shape[4])
        else: # simlulation
            data_super = data.to(pred.device)
            data_current = data_super[:, :, :, ::int(data_super.shape[-1]/pred.shape[-1]), ::int(data_super.shape[-1]/pred.shape[-1])]
            data_base = data_super[:, :, :, ::2, ::2]

            pred_current = pred
            if self.args_general.is_super_model:
                pred_base = pred[:, :, :, ::int(pred.shape[-1]/64), ::int(pred.shape[-1]/64)]
                pred_super = F.interpolate(pred.reshape(pred.shape[0],-1,*pred.shape[-2:]), size=(128,128), mode='bilinear', align_corners=False)\
                            .reshape(pred.shape[0],pred.shape[1],pred.shape[2],128,128)
                pred_super2 = F.interpolate(pred.reshape(pred.shape[0],-1,*pred.shape[-2:]), size=(128,128), mode='nearest')\
                            .reshape(pred.shape[0],pred.shape[1],pred.shape[2],128,128)

        if self.args_general.is_super_model: # super resolution
            message = ['base resolution', 'current resolution', 'linear super resolution', 'nearest super resolution']
            data = [data_base, data_current, data_super, data_super]
            pred = [pred_base, pred_current, pred_super, pred_super2]
            print('current resolution:', pred[1].shape[1], pred[1].shape[3], pred[1].shape[4])
        else:
            data = [data_current]
            pred = [pred_current]
        J_totals, J_targets, J_energys, mses, n_l2s = [], [], [], [], []
        for i in range(len(pred)):
            mask = torch.ones_like(pred[i], device = pred[i].device)
            mask[:, 0, 0] = False
            pred[i] = pred[i] * mask 
            data[i] = data[i] * mask

            mse = (torch.cat(((pred[i] - data[i])[:,:,:3], (pred[i] - data[i])[:,:,[-1]]), dim=2).square().mean((1, 2, 3, 4))).detach().cpu().numpy()
            mse_wo_smoke = ((pred[i] - data[i])[:,:,:3].square().mean((1, 2, 3, 4))).detach().cpu().numpy()
            n_l2 = (((pred[i] - data[i])[:,:,:3].square().sum((1, 2, 3, 4))).sqrt()/data[i][:,:,:3].square().sum((1, 2, 3, 4)).sqrt()).detach().cpu().numpy()
        
            J_target = - data[i][:, -1, -1, 0, 0].detach().cpu().numpy()
            J_energy = data[i][:, :, 3:5].square().mean((1, 2, 3, 4)).detach().cpu().numpy()
            J_total = J_target + self.args_general.w_energy * J_energy

            print(message[i], ', evaluate shape:', pred[i].shape[1], pred[i].shape[3], pred[i].shape[4])
            if not self.args_general.is_condition_control: 
                print('J_total=J_target+w*J_energy=', J_target.mean(), '+', self.args_general.w_energy, '*', J_energy.mean(), '=', J_total.mean())
                # print('J_total=', J_total)
            print('mse =', mse.mean(), 'mse_wo_smoke =', mse_wo_smoke.mean(), 'normalized_l2 =', n_l2.mean())

            if self.args_general.is_super_model:
                mses.append(mse_wo_smoke.mean()) # not include smoke out
            else:
                mses.append(mse.mean())
            n_l2s.append(n_l2.mean())
            J_totals.append(J_total.mean())
            J_targets.append(J_target.mean())
            J_energys.append(J_energy.mean())

        return np.array(J_totals), np.array(J_targets), np.array(J_energys), np.array(mses), np.array(n_l2s) # [4,]
    

def inference(dataloader, diffusion, design_fn, args, RESCALER):
    model = diffusion # may vary according to different control methods
    model_args = {
        "design_fn": design_fn,
        "design_guidance": args.design_guidance,
    } # may vary according to different control methods

    inferencePPL = InferencePipeline(
        model, 
        model_args,
        RESCALER,
        results_path = args.inference_result_subpath,
        args_general=args
    )
    inferencePPL.run(dataloader)
    

def main(args):
    dataloader, shape, ori_shape, RESCALER = load_data(args)
    diffusion, design_fn = load_model(args, shape, ori_shape, RESCALER) 
    inference(dataloader, diffusion, design_fn, args, RESCALER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    # setting
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--dataset', default='Smoke', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default="./data/2d", type=str,
                        help='path to dataset')
    parser.add_argument('--w_energy', default=0., type=float,
                        help='weight of energy in the objective function')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--is_super_model', default=False, type=eval,
                        help='If training the super resolution model')
    parser.add_argument('--upsample', default=0, type=int,
                        help='number of times of upsampling with super resolution model, n *= 2**upsample')
    parser.add_argument('--inference_result_path', default="./results/test/", type=str,
                        help='path to save inference result')
    parser.add_argument('--batch_size', default=50, type=int,
                        help='size of batch of input to use')

    # load model
    parser.add_argument('--exp_id', default="base_sim", type=str,
                        help='experiment id')
    parser.add_argument('--diffusion_model_path', default="./results/train/", type=str,
                        help='directory of trained diffusion model (Unet)')
    parser.add_argument('--diffusion_checkpoint', default=50, type=int,
                        help='index of checkpoint of trained diffusion model (Unet)')
    parser.add_argument('--super_exp_id', default="super_sim", type=str,
                        help='experiment id of super model')
    parser.add_argument('--super_diffusion_model_path', default="./results/train/", type=str,
                        help='directory of trained super diffusion model (Unet)')
    parser.add_argument('--super_diffusion_checkpoint', default=75, type=int,
                        help='index of checkpoint of trained super diffusion model (Unet)')
    parser.add_argument('--is_condition_control', default=False, type=eval,
                        help='If condition on control')
    parser.add_argument('--is_condition_pad', default=True, type=eval,
                        help='If condition on padded state')

    # sampling
    parser.add_argument('--using_ddim', default=False, type=eval,
                        help='If using DDIM')
    parser.add_argument('--ddim_eta', default=1., type=float, help='eta in DDIM')
    parser.add_argument('--ddim_sampling_steps', default=100, type=int, 
                        help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
    parser.add_argument('--design_guidance', default='standard', type=str,
                        help='design_guidance')
    parser.add_argument('--standard_fixed_ratio_list', nargs='+', default=[0], type=float,
                        help='standard_fixed_ratio for standard sampling')
    parser.add_argument('--coeff_ratio_list', nargs='+', default=[0], type=float,
                        help='coeff_ratio for standard-alpha sampling')
    parser.add_argument('--w_init_list', nargs='+', default=[0], type=float,
                        help='guidance intensity of initial condition')

    # wavelet
    parser.add_argument('--is_wavelet', default=False, type=eval,
                        help='If learning wavelet coefficients')
    parser.add_argument('--wave_type', default='bior1.3', type=str,
                        help='type of wavelet: bior1.3, bior2.2 ...')
    parser.add_argument('--pad_mode', default='zero', type=str,
                        help='padding mode for wavelet transform: zero')

    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for w_init in args.w_init_list:
        for standard_fixed_ratio in args.standard_fixed_ratio_list:
            for coeff_ratio in args.coeff_ratio_list:
                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                args.standard_fixed_ratio = standard_fixed_ratio
                args.coeff_ratio = coeff_ratio
                args.w_init = w_init
                args.inference_result_subpath = os.path.join(
                    args.inference_result_path,
                    current_time + "_standard_fixed_ratio_{}".format(args.standard_fixed_ratio)\
                                + "_coeff_ratio_{}".format(coeff_ratio)\
                                + "_w_init_{}".format(w_init) + "_w_energy_{}".format(args.w_energy),
                    args.inference_result_path, 
                )
                print("args: ", args)
                main(args)
    
