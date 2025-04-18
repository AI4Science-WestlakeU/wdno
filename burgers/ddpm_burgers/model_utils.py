import torch
import math


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def exists(x):
    return x is not None

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

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# guidance helper functions

def get_nablaJ(loss_fn: callable):
    '''Use explicit loss for guided inference in diffusion.
    J is the loss here, not Jacobian.

    Arguments:
        loss_fn: callable, calculates the loss.
            Arguments: 
                x: state + control
            Returns: loss (requires_grad)
    '''
    def nablaJ(x: torch.TensorType):
        x.requires_grad_(True)
        J = loss_fn(x) # vec of size of batch
        grad = torch.autograd.grad(J, x, grad_outputs=torch.ones_like(J), retain_graph = True, create_graph=True, allow_unused=True)[0]
        return grad.detach()
    return nablaJ

def get_scheduler(scheduler):
    if scheduler is None:
        return None
    # decreasing schedules
    elif scheduler == 'linear':
        raise NotImplementedError
    elif scheduler == 'cosine':
        return cosine_beta_J_schedule
    elif scheduler == 'plain_cosine':
        return plain_cosine_schedule
    elif scheduler == 'sigmoid':
        return sigmoid_schedule
    # increasing step (eta[t=0] is the largest)
    elif scheduler == 'sigmoid_flip':
        return sigmoid_schedule_flip
    else:
        raise ValueError

def get_proj_ep_orthogonal_func(norm='F'):
    # well,for 1D case there is no ambiguity but it seems less straightforward
    # for highr-dimensional embedding (even for Burgers' the data is essentailly 2D)
    # The inner product of two matrices are their F norm though.

    if norm == 'F':
        def proj_ep_orthogonal(ep, nabla_J):
            return ep + nabla_J - (nabla_J * ep).sum() * ep / ep.square().sum((-2, -1)).sqrt().unsqueeze(-1).unsqueeze(-1)
    elif norm == '1D_x':
        def proj_ep_orthogonal(ep, nabla_J):
            return ep + nabla_J - (nabla_J * ep).sum(-1).unsqueeze(-1) * ep / ep.square().sum(-1).sqrt().unsqueeze(-1)
    elif norm == '1D_t':
        def proj_ep_orthogonal(ep, nabla_J):
            return ep + nabla_J - (nabla_J * ep).sum(-2) * ep / ep.square().sum(-2).sqrt()
    else:
        raise NotImplementedError

    return proj_ep_orthogonal
    

def cosine_beta_J_schedule(t, s = 0.008):
    """
    cosine schedule (returns beta = 1 - cos^2 (x / N), which is increasing.)
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    timesteps = 1000 # NOTE: 1000 steps in sampling
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)[t]

def plain_cosine_schedule(t, s = 0.0):
    """
    cosine schedule, which is decreasing...
    """
    timesteps = 1000 # NOTE: 1000 steps in sampling
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    eta = torch.cos((x + s) / (timesteps + s))
    return eta.flip()[t] # t=0 should be zero (small step size)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    timesteps = 1000
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((x * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)[t]

def sigmoid_schedule_flip(t):
    return sigmoid_schedule(999 - t)

def linear_schedule(t):
    timesteps = 1
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)[t]


# NOTE: These are returning the full array
def linear_beta_schedule(timesteps):
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
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# helpers functions


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
