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
import multiprocessing
import tqdm

from PIL import Image
import imageio

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# init

def build_obstacles_pi_128(sim):

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

def init_sim():
    sim = FluidSimulation([127]*2, DomainBoundary([(True, True), (True, True)]), force_use_masks=True)
    build_obstacles_pi_128(sim)
    return sim


def initialize_velocity_128(vx, vy):
    velocity_array = np.empty([128, 128, 2], np.float32)
    velocity_array[...,0] = vx
    velocity_array[...,1] = vy
    init_op_velocity = StaggeredGrid(velocity_array.reshape((1,)+velocity_array.shape))
    optimizable_velocity = init_op_velocity.staggered
    return init_op_velocity, optimizable_velocity

def init_velocity_():
    init_op_velocity, optimizable_velocity = initialize_velocity_128(vx=0, vy=0.2)
    return optimizable_velocity


def get_envolve(sim,pre_velocity,c1,c2,frame):
    '''
    Input:
        sim: environment of the fluid
        pre_velocity: numpy array, [1,128,128,2]
        c1: numpy array, [nt,nx,nx]
        c2: numpy array, [nt,nx,nx]
    Output:
        next_velocity: numpy array, [1,128,128,2]
    '''
    divergent_velocity = np.zeros((1,128,128,2), dtype=np.float32)
    divergent_velocity[0,:,:,0] = c1[frame,:,:]
    divergent_velocity[0,:,:,1] = c2[frame,:,:]

    divergent_velocity[:, 16:112, 16:112, :] = 0
    divergent_velocity_ = StaggeredGrid(divergent_velocity)

    current_vel_field = math.zeros_like(divergent_velocity)
    current_vel_field[:,16:112,16:112,:] = pre_velocity.staggered[:,16:112,16:112,:]
    current_vel_field[:,:,:16,:] = divergent_velocity_.staggered[:,:,:16,:]
    current_vel_field[:,:,112:,:] = divergent_velocity_.staggered[:,:,112:,:]
    current_vel_field[:,112:,16:112,:] = divergent_velocity_.staggered[:,112:,16:112,:]
    current_vel_field[:,:16,16:112,:] = divergent_velocity_.staggered[:,:16,16:112,:]

    Current_vel_field = StaggeredGrid(current_vel_field)
    
    velocity = sim.divergence_free(Current_vel_field, solver=SparseCGPressureSolver(), accuracy=1e-8)
    velocity = sim.with_boundary_conditions(velocity)

    return velocity


def get_bucket_mask():
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
 
    return cal_smoke_list, cal_smoke_concat, set_zero_matrix


def solver(sim, init_velocity, init_density, c1, c2, dt=1):
    '''
    Input:
        sim: environment of the fluid
        init_velocity: numpy array, [128,128,2]
        init_density: numpy array, [nx,nx]
        c1: numpy array, [nt,nx,nx]
        c2: numpy array, [nt,nx,nx]
    Output:
        densitys: numpy array, [256,128,128]
        zero_densitys: numpy array, [256,128,128]
        velocitys: numpy array, [256,128,128,2]
        smoke_outs: numpy array, [256,128,128], the second is the target
    '''
    num_t = 256
    nt, nx = c1.shape[0], c1.shape[1]
    time_interval, space_interval = int(num_t/nt), int(128/nx)
    init_density = np.tile(init_density.reshape(nx,1,nx,1), (1,space_interval,1,space_interval)).reshape(128,128,1)
    c1 = np.tile(c1.reshape(nt,1,nx,1,nx,1), (1,time_interval,1,space_interval,1,space_interval)).reshape(256,128,128)
    c2 = np.tile(c2.reshape(nt,1,nx,1,nx,1), (1,time_interval,1,space_interval,1,space_interval)).reshape(256,128,128)
    loop_advected_density = init_density[:-1, :-1].reshape(1, 127, 127, 1)
    loop_velocity = StaggeredGrid(init_velocity)

    cal_smoke_list, cal_smoke_concat, set_zero_matrix = get_bucket_mask()

    densitys, zero_densitys, velocitys, smoke_out_record = [], [], [], []
    smoke_outs = np.zeros((7,), dtype=float)
    density_set_zero = loop_advected_density.copy()
    for frame in range(num_t):
        loop_velocity = get_envolve(sim=sim,pre_velocity=loop_velocity,c1=c1,c2=c2,frame=frame)
        
        # using advect function to get current density field movement under velocity field
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
        density_set_zero = loop_velocity.advect(density_set_zero, dt=dt)

        array = np.zeros((128, 128), dtype=float)
        array[:-1,:-1] = loop_advected_density[0,:,:,0] 

        if(np.sum((array[:,:]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs[i] += np.sum(array[:,:] * cal_smoke_list[i][:,:])
            density_set_zero[0,:,:,0] = density_set_zero[0,:,:,0] * set_zero_matrix[:-1,:-1]
    
        velocity_array = np.empty([128, 128, 2], np.float32)
        velocity_array[...,0] = loop_velocity.staggered[0,:,:,0]
        velocity_array[...,1] = loop_velocity.staggered[0,:,:,1]

        array_set_zero = np.zeros((128, 128), dtype=float)
        array_set_zero[:-1,:-1] = density_set_zero[0,:,:,0]
        
        # velocity_array = np.transpose(velocity_array, (2, 0, 1))
        densitys.append(array)
        zero_densitys.append(array_set_zero)
        velocitys.append(velocity_array)
        smoke_out_value = smoke_outs[1]/(np.sum(smoke_outs)+np.sum(array_set_zero))
        smoke_out_record.append(smoke_out_value)

    smoke_out_record = np.stack(smoke_out_record)
    smoke_out_record = np.tile(smoke_out_record[:, None, None], (1, 128, 128))
    # print(f"smoke_out_record.shape: {smoke_out_record.shape}")

    return np.stack(densitys), np.stack(zero_densitys), np.stack(velocitys), c1, c2, np.stack(smoke_out_record)


def get_bucket_mask_torch(device):
    bucket_pos = [(112, 24-2, 127-112, 16+4), (112, 56-2, 127-112, 16+4), (112, 88-2, 127-112, 16+4)]
    bucket_pos_y = [(24-2, 0, 16+4, 16), (56-2, 0, 16+4, 16), (24-2, 112, 16+4, 127-112), (56-2, 112, 16+4, 127-112)]
    
    cal_smoke_list = [] 
    set_zero_matrix = torch.ones((128, 128), device=device)
    cal_smoke_concat = torch.zeros((128, 128), device=device)
    
    for pos in bucket_pos:
        cal_smoke_matrix = torch.zeros((128, 128), device=device) 
        y, x, len_y, len_x = pos
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix.unsqueeze(0))
    
    for pos in bucket_pos_y:
        cal_smoke_matrix = torch.zeros((128, 128), device=device)
        y, x, len_y, len_x = pos
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix.unsqueeze(0))

    return cal_smoke_list, cal_smoke_concat.unsqueeze(0), set_zero_matrix.unsqueeze(0)


# plot

def draw_pic(des, ver_bound, hor_bound, frame, save_pic_path, name=None):
    fig, ax = plt.subplots()
    ax.imshow(des[frame,:,:], origin='lower')
    ax.scatter(hor_bound, ver_bound, color="grey", marker=",")
    fig.savefig(os.path.join(save_pic_path, f'density_{name}_{frame}.png'), dpi=300)
    plt.close(fig)
    return


def get_bound(sim):
    res_sim = sim._fluid_mask.reshape((127,127))
    boundaries = np.argwhere(res_sim==0)
    global ver_bound, hor_bound
    ver_bound = boundaries[:,0]
    hor_bound = boundaries[:,1]
    return ver_bound, hor_bound


def load_and_sort_images2(save_pic_path):
    paths = []
    for filename in save_pic_path:
        paths.append(filename)
    sorted_paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x).group()))
    return sorted_paths



def plot_vector_field_128(velocity, frame, pic_dir):
    velocity = velocity[frame,:,:,:]
    fig = plt.figure(dpi=600)
    x,y = np.meshgrid(np.linspace(0,127,128),np.linspace(0, 127, 128))

    xvel = np.zeros([128]*2)
    yvel = np.zeros([128]*2)

    xvel[1::4,1::4] = velocity[1::4,1::4,0]
    yvel[1::4,1::4] = velocity[1::4,1::4,1]

    plt.quiver(x,y,xvel,yvel,scale=2.5, scale_units='inches')
    plt.title('Vector Field Plot')
    plt.savefig(os.path.join(pic_dir, f'field_{frame}.png'), dpi=300)
    # plt.show()


def plot_control_field_128(c1, c2, frame, pic_dir):
    fig = plt.figure(dpi=600)
    x,y = np.meshgrid(np.linspace(0,127,128),np.linspace(0, 127, 128))

    xvel = np.zeros([128]*2)
    yvel = np.zeros([128]*2)

    xvel[1::4,1::4] = c1[frame,1::4,1::4]
    yvel[1::4,1::4] = c2[frame,1::4,1::4]

    plt.quiver(x,y,xvel,yvel,scale=2.5, scale_units='inches')
    plt.title('Field Plot')
    plt.savefig(os.path.join(pic_dir, f'{frame}.png'), dpi=300)
    # plt.show()

def gif_density(densitys,zero,pic_dir='./dens_sample/',gif_dir='./gifs', name='0'):
    """
    Function:
        Generate densitys or zero_densitys
        gif saved at gif_dir
    Input: 
        densitys: numpy array [256,128,128]
        zero: when density->False, when zero_densitys->True
    """
    sim = init_sim()
    ver_bound, hor_bound = get_bound(sim)
    if(not os.path.exists(pic_dir)):
        os.makedirs(pic_dir)
    if(not os.path.exists(gif_dir)):
        os.makedirs(gif_dir)
    for frame in range(densitys.shape[0]):
        draw_pic(des=densitys,ver_bound=ver_bound,hor_bound=hor_bound,frame=frame,save_pic_path=pic_dir,name=name)
    sorted_pic_path = load_and_sort_images2(os.listdir(pic_dir))
    images = [imageio.imread(os.path.join(pic_dir, file)) for file in sorted_pic_path]
    # if zero==False:
    #     gif_save_path = os.path.join(gif_dir, f'density_{name}.gif')
    # else:
    #     gif_save_path = os.path.join(gif_dir, f'zero_density{name}.gif')
    # imageio.mimsave(gif_save_path, images, duration=0.05)
    for file in os.listdir(pic_dir):
        file_path = os.path.join(pic_dir, file)
        # os.remove(file_path)


def gif_vel(velocitys, pic_dir='./dens_sample/',gif_dir='./gifs', name='0'):
    """
    Function:
        Generate velocitys or control
        gif saved at gif_dir
    Input: 
        velocitys: numpy array, [256,128,128,2]
    """
    sim = init_sim()
    ver_bound, hor_bound = get_bound(sim)
    if(not os.path.exists(pic_dir)):
        os.makedirs(pic_dir)
    if(not os.path.exists(gif_dir)):
        os.makedirs(gif_dir)
    for frame in range(velocitys.shape[0]):
        plot_vector_field_128(velocity=velocitys, frame=frame, pic_dir=pic_dir)
    sorted_pic_path = load_and_sort_images2(os.listdir(pic_dir))
    images = [imageio.imread(os.path.join(pic_dir, file)) for file in sorted_pic_path]
    gif_save_path = os.path.join(gif_dir, f'velocity_{name}.gif')
    imageio.mimsave(gif_save_path, images, duration=0.05)
    for file in os.listdir(pic_dir):
        file_path = os.path.join(pic_dir, file)
        os.remove(file_path)


def gif_control(c1, c2, pic_dir='./dens_sample/',gif_dir='./gifs', control_bool=False, name='0'):
    """
    Function:
        Generate velocitys or control
        gif saved at gif_dir
    Input: 
        velocitys: numpy array, [256,128,128,2]
    """
    sim = init_sim()
    ver_bound, hor_bound = get_bound(sim)
    if(not os.path.exists(pic_dir)):
        os.makedirs(pic_dir)
    if(not os.path.exists(gif_dir)):
        os.makedirs(gif_dir)
    for frame in range(c1.shape[0]):
        plot_control_field_128(c1, c2, frame, pic_dir)
    sorted_pic_path = load_and_sort_images2(os.listdir(pic_dir))
    images = [imageio.imread(os.path.join(pic_dir, file)) for file in sorted_pic_path]
    if control_bool== True:
        gif_save_path = os.path.join(gif_dir, f'control_{name}.gif')
    else:
        gif_save_path = os.path.join(gif_dir, f'velocity_{name}.gif')
    imageio.mimsave(gif_save_path, images, duration=0.05)
    for file in os.listdir(pic_dir):
        file_path = os.path.join(pic_dir, file)
        os.remove(file_path)


    

if __name__ == "__main__":
    # Fluid Simulation
    '''
    Input:
        sim: environment of the fluid
        init_velocity: numpy array, [128,128,2]
        init_density: numpy array, [128,128]
        c1: numpy array, [nt,nx,nx]
        c2: numpy array, [nt,nx,nx]
    Output:
        densitys: numpy array, [256,128,128]
        zero_densitys: numpy array, [256,128,128]
        velocitys: numpy array, [256,128,128,2]
    '''
    init_velocity = init_velocity_()
    # init_density = np.zeros((128,128,1),dtype=float)
    # c1 = np.zeros((256,128,128),dtype=float)
    # c2 = np.zeros((256,128,128),dtype=float)
    init_density = np.random.rand(128, 128, 1)
    c1 = np.random.rand(256, 128, 128)
    c2 = np.random.rand(256, 128, 128)
    sim = init_sim()
    densitys, zero_densitys, velocitys, smoke_out = solver(sim, init_velocity, init_density, c1, c2)
    
    # GIF Generation
    ver_bound, hor_bound = get_bound(sim)
    # control
    gif_control(c1, c2,control_bool=True)
    print("control gif down!")
    # densitys
    gif_density(densitys,zero=False)
    print("densitys gif down!")
    # zero_densitys
    gif_density(zero_densitys,zero=True)
    print("zero_densitys gif down!")
    # velocitys
    gif_vel(velocitys)
    print("velocitys gif down!")