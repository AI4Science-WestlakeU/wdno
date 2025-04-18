import sys
sys.path.append("../")

from phi.fluidformat import *
from phi.flow import FluidSimulation, DomainBoundary
import random
import numpy as np
from phi.math.nd import *
import matplotlib.pyplot as plt
from phi.solver.sparse import SparseCGPressureSolver
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from phi.fluidformat import *
import os
import pdb
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz
from numpy.random import default_rng
import argparse
import multiprocessing

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def plot_initial_velocity(sim):
    """
    Function: Plot Obstacles of sim
    Input: sim
    """
    fig_ob, ax_ob = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable_ob0 = ax_ob[0].imshow(sim._active_mask[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    # mappable_ob1 = ax_ob[1].imshow(sim._active_mask[0,:,:,1], cmap='viridis',
    #                          #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
    #                          interpolation="bicubic",
    #                          aspect='auto',
    #                          origin='lower')
    fig_ob.colorbar(mappable_ob0, ax=ax_ob[0])
    #fig_ob.colorbar(mappable_ob1, ax=ax_ob[1])
    fig_ob.tight_layout()


def plot_init_op_velocity(init_op_velocity):
    """
    Function: Plot initial velocity
    Input: StaggeredGrid type velocity
    """
    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable0 = ax[0].imshow(init_op_velocity.staggered[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    mappable1 = ax[1].imshow(init_op_velocity.staggered[0,:,:,1], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('Heatmap of initial velocity in x-dirction')
    ax[1].set_title('Heatmap of initial velocity in y-dirction')
    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()


def plot_velocity_with_mask(divergent_velocity):
    """
    Function: Plot the heatmap of velocity
    Input: StaggeredGrid type divergent_velocity
    """
    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable0 = ax[0].imshow(divergent_velocity.staggered[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    mappable1 = ax[1].imshow(divergent_velocity.staggered[0,:,:,1], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('Heatmap of initial velocity \n in x-dirction (with mask)')
    ax[1].set_title('Heatmap of initial velocity \n in y-dirction (with mask)')

    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()


###Vector Field Representation of Velocity###
def plot_vector_field_128(velocity):
    """
    Function: Plot velocity field
    Input: StaggeredGrid type velocity
    """
    fig = plt.figure()
    x,y = np.meshgrid(np.linspace(0,127,128),np.linspace(0, 127, 128))

    xvel = np.zeros([128]*2)
    yvel = np.zeros([128]*2)

    xvel[1::4,1::4] = velocity.staggered[0,1::4,1::4,0]
    yvel[1::4,1::4] = velocity.staggered[0,1::4,1::4,1]

    plt.quiver(x,y,xvel,yvel,scale=2.5, scale_units='inches')
    plt.title('Vector Field Plot')



def plot_velocity_boundary_effect2(velocity):
    hor_velocity_array = np.empty([128, 128], np.float32)
    hor_velocity_array = velocity.staggered[0,:,:,0]

    ver_velocity_array = np.empty([128, 128], np.float32)
    ver_velocity_array = velocity.staggered[0,:,:,1]


    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of velocity meating equation in x-dirction###
    mappable0 = ax[0].imshow(hor_velocity_array, cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of velocity meating equation in y-dirction###
    mappable1 = ax[1].imshow(ver_velocity_array, cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('x-axis velocity boundary effect')
    ax[1].set_title('y-axis velocity boundary effect')
    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()



def plot_loop(loop_advected_density, loop_velocity, target_des_array, frame=None):
    """
    Function: Plot density field & velocity field
    Input:
        loop_advected_density: numpy array
        loop_velocity: staggeredgrid
        target_des_array: (optional) numpy array
    """
    fig, ax = plt.subplots()
    ax.imshow(loop_advected_density[0,:,:,0], origin='lower')
    """
    xvel = np.zeros([129]*2)
    yvel = np.zeros([129]*2)
    xvel[1::8,1::8] = loop_velocity.staggered[0,1::8,1::8,0]
    yvel[1::8,1::8] = loop_velocity.staggered[0,1::8,1::8,1]
    ax.quiver(x,y,xvel,yvel,scale=3., scale_units='inches', color="white")
    """
    ax.scatter(hor_bound, ver_bound, color="grey", marker=",")

    fig.savefig(f'dens_sample/{frame}.png', dpi=50)

    velocity_array = np.empty([128, 128, 2], np.float32)
    velocity_array[...,0] = loop_velocity.staggered[0,:,:,0]
    velocity_array[...,1] = loop_velocity.staggered[0,:,:,1]
    
    fig1, ax1 = plt.subplots(figsize=(8,4), ncols=2)
    ###Heatmap of velocity meating equation in x-dirction###
    mappable0 = ax1[0].imshow(velocity_array[:,:,0], cmap='viridis',
                                #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                aspect='auto',
                                origin='lower')
    ###Heatmap of velocity meating equation in y-dirction###
    mappable1 = ax1[1].imshow(velocity_array[:,:,1], cmap='viridis',
                                #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                interpolation="bicubic",
                                aspect='auto',
                                origin='lower')
    fig1.colorbar(mappable0, ax=ax1[0])
    fig1.colorbar(mappable1, ax=ax1[1])
    fig1.tight_layout()


def get_real_vel(vel):
    """
    Function: Get Real Velocity from Nomral Distribution
    Input: 
        vel: float
    Output:
        real_vel: float
    """
    std = abs(vel / 4)
    real_vel = np.random.normal(vel, std)
    return real_vel


def build_obstacles_pi_128(sim):
    """
    Function: Set obstacles
    Input: 
        sim: FluidSimulation object
    """

    sim.set_obstacle((1, 96), (16, 16)) # Bottom

    sim.set_obstacle((8, 1), (16, 16)) # Left Down
    sim.set_obstacle((16, 1), (40, 16)) # Left Medium
    sim.set_obstacle((40, 1), (72, 16)) # Left Up

    sim.set_obstacle((8, 1), (16, 112)) # Right Down
    sim.set_obstacle((16, 1), (40, 112)) # Right Medium
    sim.set_obstacle((40, 1), (72, 112)) # Right Up

    # Buckets
    sim.set_obstacle((1, 8), (112, 16)) # [16-24] # [24-40(16)]
    sim.set_obstacle((1, 16), (112, 40)) # [40-56] # [56-72(16)]
    sim.set_obstacle((1, 16),(112, 72)) # [72-88] # [88-104(16)]
    sim.set_obstacle((1, 8),(112, 104)) # [104-113]


    # y-axis obstacle
    sim.set_obstacle((16, 1), (64, 48))
    sim.set_obstacle((16, 1), (96, 48))
    sim.set_obstacle((16, 1), (64, 80))
    sim.set_obstacle((16, 1), (96, 80))
    
    # Should Change
    sim.set_obstacle((1, 128-40-40), (40, 40)) # x-axis


def apply_mask(sim, optimizable_velocity):
    ###Set Initial Condition for Velocity###
    control_mask = sim.ones("staggered")
    control_mask.staggered[:, 16:112, 16:112, :] = 0
    divergent_velocity = optimizable_velocity * control_mask.staggered
    divergent_velocity = StaggeredGrid(divergent_velocity)
    return divergent_velocity 


def initialize_field_128():
    """
    Function: initialize fluid field
    Output:
        sim: FluidSimulation Object
    """
    sim = FluidSimulation([127]*2, DomainBoundary([(True, True), (True, True)]), force_use_masks=True)
    build_obstacles_pi_128(sim)
    return sim


def get_per_vel(xs, ys):
    """
    Function: Calculate vague velocity
    Input:
        xs: random x-position for turn
        ys: random y-position for turn
    Output:
        vxs: vx list
        vys: vy list
        intervals: frame num for each interval
    """
    distance = ((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)**(0.5) + ((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)**(0.5) + ((xs[3]-xs[2])**2+(ys[3]-ys[2])**2)**(0.5) + ((xs[4]-xs[3])**2+(ys[4]-ys[3])**2)**(0.5)
    distance1 = ((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)**(0.5)
    distance2 = ((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)**(0.5)
    distance3 = ((xs[3]-xs[2])**2+(ys[3]-ys[2])**2)**(0.5)
    distance4 = ((xs[4]-xs[3])**2+(ys[4]-ys[3])**2)**(0.5)


    v = distance / float(scenelength)

    vx1 = v * (xs[1]-xs[0]) / distance1 
    vy1 = v * (ys[1]-ys[0]) / distance1 
    vx2 = v * (xs[2]-xs[1]) / distance2
    vy2 = v * (ys[2]-ys[1]) / distance2
    vx3 = v * (xs[3]-xs[2]) / distance3 
    vy3 = v * (ys[3]-ys[2]) / distance3
    vx4 = v * (xs[4]-xs[3]) / distance4 
    vy4 = v * (ys[4]-ys[3]) / distance4

    scale = np.random.uniform(2, 5)

    vxs = [get_real_vel(scale*vx1), get_real_vel(scale*vx2), get_real_vel(scale*vx3), get_real_vel(scale*vx4)]
    vys = [get_real_vel(5*vy1), get_real_vel(5*vy2), get_real_vel(5*vy3), get_real_vel(5*vy4)]


    interval1 = int(scenelength * distance1 / distance)
    interval2 = int(scenelength * distance2 / distance)
    interval3 = int(scenelength * distance3 / distance)

    intervals = [interval1, interval2, interval3]

    return vxs, vys, intervals


def exp2_target_128():
    """
    Function: Get x,y for turns
    Output:
        xs: list x-position for each turn
        ys: list y-position for each turn
    """
    m = 5
    start_x = np.random.randint(16+1+m, 112-10-m)
    start_y = np.random.randint(16+1+m, 40-10-m)
    if start_x < (64-10):
        a = 0
    else:
        a = 1
    target1_x = np.random.randint(16+m, 64-10) if a == 0 else np.random.randint(64, 112-10-m)
    target2_x = np.random.randint(16+m, 64-10) if a == 0 else np.random.randint(64, 112-10-m)
    target3_x = np.random.randint(50, 80-1-10)
    end_x = np.random.randint(64-8, 64+8-10)

    target1_y = 40
    target2_y = 50
    target3_y = 64
    end_y = 112
    
    xs = [int(start_x), int(target1_x), int(target2_x), int(target3_x), int(end_x)]
    ys = [int(start_y), int(target1_y), int(target2_y), int(target3_y), int(end_y)]
    
    return xs, ys


def initialize_gas_exp2_128(xs, ys):
    """
    Function: Intialize density field
    Input:
        xs: x-postion list
        ys: y-postion list
    Output:
        array: numpy array density field
    """
    array = np.zeros([127, 127, 1], np.float32)
    start_x = xs[0]
    start_y = ys[0]
    array[start_y:start_y+11, start_x:start_x+11, :] = 1
    return array


def initialize_velocity_128(vx, vy):
    """
    Function: Initialize velocity field
    Input:
        vx, vy: float velocity-x, velocity-y
    Output:
        init_op_velocity: StaggeredGrid velocity
        optimizable_velocity: numpy array velocity
    """
    velocity_array = np.empty([128, 128, 2], np.float32)
    velocity_array[...,0] = vx
    velocity_array[...,1] = vy
    init_op_velocity = StaggeredGrid(velocity_array.reshape((1,)+velocity_array.shape))
    optimizable_velocity = init_op_velocity.staggered
    return init_op_velocity, optimizable_velocity


def get_envolve(sim,pre_velocity,frame,control_write,vx=None,vy=None):
    """
    Function: get next step velocity with indirect control
    Input:
        sim: FluidSimulation Object
        pre_velocity: StaggeredGrid previous velocity
        frame: int
        control_write: numpy array
        vx: float
        vy: float
    Output:
        velocity: StaggeredGrid next velocity
        control_write: numpy array
    """
    if(vx==None and vy==None):
        current_vel_field = np.zeros_like(pre_velocity.staggered)
        
        # Add noise # noise_arr.shape = [1,128,128,2]
        noise_arr = np.random.normal(loc=0,scale=0.1,size=pre_velocity.staggered.shape)
        
        # Calculate Current Controlled Velocity # current_vel_field.shape = [1,128,128,2]
        current_vel_field[:,:,:16,:] = pre_velocity.staggered[:,:,:16,:] + noise_arr[:,:,:16,:]
        current_vel_field[:,:,112:,:] = pre_velocity.staggered[:,:,112:,:] + noise_arr[:,:,112:,:]
        current_vel_field[:,112:,16:112,:] = pre_velocity.staggered[:,112:,16:112,:] + noise_arr[:,112:,16:112,:]
        current_vel_field[:,:16,16:112,:] = pre_velocity.staggered[:,:16,16:112,:] + noise_arr[:,:16,16:112,:]
        
        divergent_velocity =  current_vel_field.copy()

        if frame % record_scale == 0:
            control_write[:,:,0,int(frame/record_scale)] = divergent_velocity[0,::2,::2,0]
            control_write[:,:,1,int(frame/record_scale)] = divergent_velocity[0,::2,::2,1]

        current_vel_field[:,16:112,16:112,:] = pre_velocity.staggered[:,16:112,16:112,:]

        Current_vel_field = StaggeredGrid(current_vel_field)

        velocity = sim.divergence_free(Current_vel_field, solver=SparseCGPressureSolver(), accuracy=1e-8)
        velocity = sim.with_boundary_conditions(velocity)

        return velocity, control_write
    else:
        divergent_velocity = np.zeros((1,128,128,2), dtype=np.float32)

        divergent_velocity[:,:,:,0] = np.random.normal(loc=vx,scale=abs(vx/10),size=(1,128,128))
        divergent_velocity[:,:,:,1] = np.random.normal(loc=vy,scale=abs(vy/10),size=(1,128,128))

        divergent_velocity[:, 16:112, 16:112, :] = 0
        divergent_velocity_ = StaggeredGrid(divergent_velocity)
        
        if frame % record_scale == 0:
            control_write[:,:,0,int(frame/record_scale)] = divergent_velocity[0,::2,::2,0]
            control_write[:,:,1,int(frame/record_scale)] = divergent_velocity[0,::2,::2,1]

        current_vel_field = math.zeros_like(divergent_velocity_.staggered)
        current_vel_field[:,16:112,16:112,:] = pre_velocity.staggered[:,16:112,16:112,:]

        current_vel_field[:,:,:16,:] = divergent_velocity_.staggered[:,:,:16,:]
        current_vel_field[:,:,112:,:] = divergent_velocity_.staggered[:,:,112:,:]
        current_vel_field[:,112:,16:112,:] = divergent_velocity_.staggered[:,112:,16:112,:]
        current_vel_field[:,:16,16:112,:] = divergent_velocity_.staggered[:,:16,16:112,:]

        Current_vel_field = StaggeredGrid(current_vel_field)
        
        velocity = sim.divergence_free(Current_vel_field, solver=SparseCGPressureSolver(), accuracy=1e-8)
        velocity = sim.with_boundary_conditions(velocity)

        return velocity, control_write


def get_intial_state(sim,xs,ys,vxs,vys,density_write,density_set_zero_write,velocity_write,control_write):
    """
    Function: get initial state
    """
    # initialize velocity
    init_op_velocity, optimizable_velocity = initialize_velocity_128(vx=0, vy=0.2)

    velocity, control_write = get_envolve(sim=sim,pre_velocity=init_op_velocity,frame=0,vx=vxs[0],vy=vys[0],control_write=control_write)
    
    array = initialize_gas_exp2_128(xs=xs, ys=ys)
    init_op_density = StaggeredGrid(array)
    init_op_density = init_op_density.staggered.reshape((1,)+init_op_density.staggered.shape)

    advected_density = velocity.advect(init_op_density, dt=dt)

    # write initial state
    loop_velocity = velocity

    density_write[:,:,:,0] = advected_density[0,::2,::2,:]
    density_set_zero_write[:,:,:,0] = advected_density[0,::2,::2,:]
    
    velocity_write[:,:,0,0] = loop_velocity.staggered[0,::2,::2,0]
    velocity_write[:,:,1,0] = loop_velocity.staggered[0,::2,::2,0]

    return advected_density, velocity, density_write, density_set_zero_write, velocity_write, control_write


def get_bucket_mask():
    """
    Function: get absorb area
    """
    bucket_pos = [(112,24-2,127-112,16+4),(112,56-2,127-112,16+4),(112,88-2,127-112,16+4)]
    bucket_pos_y = [(24-2,0,16+4,16),(56-2,0,16+4,16),(24-2,112,16+4,127-112),(56-2,112,16+4,127-112)]
    cal_smoke_list = [] 
    set_zero_matrix = np.ones((128,128))
    cal_smoke_concat = np.zeros((128,128))
    for pos in bucket_pos:
        cal_smoke_matrix = np.zeros((128,128)) 
        y,x,len_y,len_x = pos[0], pos[1], pos[2], pos[3]
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix)
    for pos in bucket_pos_y:
        cal_smoke_matrix = np.zeros((128,128)) 
        y,x,len_y,len_x = pos[0], pos[1], pos[2], pos[3]
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix)
 
    return cal_smoke_list, cal_smoke_concat, set_zero_matrix #, absorb_matrix, cal_inside_smoke


def get_save_name():
    """
    Function: get save name
    """
    des_name_no_zero = f'Density.npy'
    vel_name = f'Velocity.npy'
    control_name = f'Control.npy'
    smoke_cal_name = f'Smoke.npy'

    return des_name_no_zero, vel_name, control_name, smoke_cal_name


def get_domain_name():
    return f'domain.npy'


def write_vel_density(loop_velocity,loop_advected_density,loop_density_no_set,density_write,density_set_zero_write,velocity_write,frame,smoke_outs_128_record,smoke_outs_128):
    """
    Function: write velocity density field for turns
    """
    density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::2,::2,:] 
    density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::2,::2,:] 
    velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,0]
    velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,1]

    array = np.zeros((128,128,1), dtype=float)
    array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 
    
    if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
        for i in range(len(cal_smoke_list)):
            smoke_outs_128_record[i] += np.sum(array[::2,::2,0] * cal_smoke_list[i][::2,::2])
        loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

    smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
    smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_density_no_set[0,::2,::2,:])
    return loop_advected_density,density_write,density_set_zero_write,smoke_outs_128_record,smoke_outs_128


def plot_narray(matrix):
    plt.imshow(matrix, cmap='gray')  # cmap 参数设置颜色映射表，可选
    plt.colorbar()  # 显示颜色条
    # plt.savefig('smoke_concat.png')
    # plt.show()


def loop_write_0423(sim,loop_advected_density,loop_velocity,smoke_outs_128,save_sim_path,vxs,vys,intervals,xs,ys,density_write, \
                    density_set_zero_write,velocity_write,control_write,record_scale):
    """
    Function: Write loop 
    """
    print_list = [1, scenelength/16, scenelength/8, scenelength/4, scenelength/2, scenelength]
    loop_density_no_set = loop_advected_density.copy()
    loop_advected_density = loop_advected_density
    control_write = control_write
    density_write = density_write
    velocity_write = velocity_write
    smoke_outs_128_record = np.zeros((7,), dtype=float)

    smoke_outs_128[0,-1] = np.sum(loop_density_no_set[0,::2,::2,:])

    # print("step 1")
    for frame in range(1, intervals[0]):
        # print(frame)

        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,control_write=control_write)
        
        # Solver Attention
        # using advect function to get current density field movement under velocity field
        # loop_advected_density - numpy array - shape [1,255,255,1]

        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[::2,::2,0] * cal_smoke_list[i][::2,::2])
            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        # print(np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,::2,::2,0]))
        
        if frame % record_scale == 0:
            density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::2,::2,:] 
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,::2,::2,:])
        
    
    frame += 1
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,vx=vxs[1],vy=vys[1],control_write=control_write)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)
    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128)


    # print("step 2")
    for frame in range(intervals[0]+1, intervals[0]+intervals[1]):
        
        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,control_write=control_write)

        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 
        
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[::2,::2,0] * cal_smoke_list[i][::2,::2])
            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        # print(np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,::2,::2,0]))
        
        if frame % record_scale == 0:
            density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::2,::2,:] 
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,::2,::2,:])

    # get extreme point control
    frame += 1
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,vx=vxs[2],vy=vys[2],control_write=control_write)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)
    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128)

    # print("step 3")
    for frame in range(intervals[0]+intervals[1]+1, intervals[0]+intervals[1]+intervals[2]):

        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,control_write=control_write)

        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 
        
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[::2,::2,0] * cal_smoke_list[i][::2,::2])
            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        # print(np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,::2,::2,0]))
        
        if frame % record_scale == 0:
            density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::2,::2,:] 
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,::2,::2,:])
        
    
    frame += 1
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,vx=vxs[3],vy=vys[3],control_write=control_write)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)
    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128)

    # print("step 4")
    for frame in range(intervals[0]+intervals[1]+intervals[2]+1, scenelength+1):

        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame,control_write=control_write)
        
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 
        
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[::2,::2,0] * cal_smoke_list[i][::2,::2])
            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        # print(np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,::2,::2,0]))
        
        if frame % record_scale == 0:
            density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::2,::2,:] 
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::2,::2,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,::2,::2,:])
            
    
    des_name, vel_name,control_name, smoke_cal_name = get_save_name()[0],get_save_name()[1],get_save_name()[2], get_save_name()[3]
    des_path = os.path.join(save_sim_path,des_name)
    vel_path = os.path.join(save_sim_path,vel_name)
    control_path = os.path.join(save_sim_path,control_name)
    smoke_path = os.path.join(save_sim_path, smoke_cal_name)

    np.save(des_path, density_write)
    np.save(vel_path, velocity_write)
    np.save(control_path, control_write)
    np.save(smoke_path, smoke_outs_128)
    save_txt_path = os.path.join(save_sim_path, 'smoke_out.csv')
    np.savetxt(save_txt_path, smoke_outs_128, delimiter=',')

    return True


def exp2_same_side_128(is_train_, fix_velocity_, Test_, branch_num, data_savepath):
    pid = os.getpid()
    np.random.seed(pid)

    Test_ = Test_

    is_train, fix_velocity = is_train_, fix_velocity_

    if(Test_):
        scenecount = 5
    elif(is_train):
        scenecount = 2
    else:
        scenecount = 40
    
    # Universal Parameters
    global scenelength, dt, cal_smoke_list, cal_smoke_concat, set_zero_matrix, record_scale
    scenelength = 256
    dt = 1
    record_scale = 8
    cal_smoke_list, cal_smoke_concat, set_zero_matrix= get_bucket_mask()
    save_branch_name = f'branch{branch_num}'


    if(Test_):
            save_path = 'test_0501'
    else:
        if is_train:
            if fix_velocity:
                save_path = f'./{data_savepath}/'
            else:
                save_path = f'./{data_savepath}/'

        else:
            if fix_velocity:
                save_path = f'./{data_savepath}/'
            else:
                save_path = f'./{data_savepath}/'

    # makedirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    contents = os.listdir(save_path)


    begin_sim_set = scenecount * int(branch_num)

    for scene_index in range(begin_sim_set, scenecount+begin_sim_set):
        smoke_outs_128 = np.zeros(7)

        print("SCENE"+str(scene_index))
        
        sim = initialize_field_128()

        res_sim = sim._fluid_mask.reshape((127,127))
        boundaries = np.argwhere(res_sim==0)
        global ver_bound, hor_bound
        ver_bound = boundaries[:,0]
        hor_bound = boundaries[:,1]

        xs, ys = exp2_target_128()

        if scene_index < 10:
            sim_path = f'sim_00000{scene_index}'
        elif scene_index < 100:
            sim_path = f'sim_0000{scene_index}'
        elif scene_index < 1000:
            sim_path = f'sim_000{scene_index}'
        elif scene_index < 10000:
            sim_path = f'sim_00{scene_index}'
        elif scene_index < 100000:
            sim_path = f'sim_0{scene_index}'

        save_sim_path = os.path.join(save_path, sim_path)

        if not os.path.exists(save_sim_path):
            os.makedirs(save_sim_path)

        domain_name = get_domain_name()
        save_domain_path = os.path.join(save_sim_path, domain_name)
        np.save(save_domain_path, sim._active_mask)

        vxs, vys, intervals = get_per_vel(xs=xs, ys=ys)
        record_frame_len = 33
        
        density_write = np.zeros((64,64,1,record_frame_len), dtype=float)
        density_set_zero_write = np.zeros((64,64,1,record_frame_len), dtype=float)
        velocity_write = np.zeros((64,64,2,record_frame_len), dtype=float)
        control_write = np.zeros((64,64,2,record_frame_len), dtype=float)
        smoke_outs_128 = np.zeros((record_frame_len, 8))

        loop_advected_density,loop_velocity,density_write,density_set_zero_write,velocity_write,control_write = get_intial_state(xs=xs, ys=ys, \
                                        sim=sim, vxs=vxs, vys=vys,density_write=density_write,density_set_zero_write=density_set_zero_write,\
                                        velocity_write=velocity_write,control_write=control_write)
        
        flag = loop_write_0423(sim=sim, loop_advected_density=loop_advected_density, loop_velocity=loop_velocity,smoke_outs_128=smoke_outs_128,\
                                save_sim_path=save_sim_path,vxs=vxs,vys=vys,intervals=intervals,xs=xs,ys=ys,density_write=density_write, \
                                density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,control_write=control_write, \
                                record_scale=record_scale)
        



    print("DATA GENERATION DOWN!")


# if __name__ == "__main__":
#     exp2_same_side_128(is_train_=False, fix_velocity_=False,Test_=True, branch_num=0, data_savepath='test0514')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_or_train", type=str, help="(test:input test or train)")
    parser.add_argument("--data_savepath", type=str, help='dataset location')
    parser.add_argument("--branch_begin", type=str, help='branch begin number')
    parser.add_argument("--branch_end", type=str, help='branch end number')

    args = parser.parse_args()
    data_savepath = args.data_savepath
    if args.test_or_train == 'test':
        Test_ = True
        is_train = False
    elif args.test_or_train == 'train':
        Test_ = False
        is_train = True

    begin_no = int(args.branch_begin)
    end_no = int(args.branch_end)
    branch_list = np.arange(begin_no,end_no)
    fix_velocity_ = False

    with multiprocessing.Pool(len(branch_list)) as pool:
        args_func = [(is_train,fix_velocity_,Test_, str(branch_num), data_savepath) for branch_num in branch_list]
        pool.starmap(exp2_same_side_128, args_func)