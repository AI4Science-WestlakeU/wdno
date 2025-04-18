import torch
import torch.nn.functional as F


def upsample_coef(w_sub, shape):
    '''
    upsample the wavelet coefficients to the target shape
    (old) w_sub: [N, layer, nt(maybe padded), nx(maybe padded)]
    (old) w: [N, layer, shape[-2], shape[-1]]
    w_sub: [N, layer, nt, nx]
    w: [N, layer, 2*nt, 2*nx]
    '''
    N, l, nt, nx = w_sub.shape[0], w_sub.shape[1], w_sub.shape[2], w_sub.shape[3]
    w_sub = w_sub.unsqueeze(-2).unsqueeze(-1).expand(N, l, nt, 2, nx, 2)
    # return w_sub.reshape(N, l, nt*2, nx*2)[..., :shape[-2], :shape[-1]]
    return w_sub.reshape(N, l, nt*2, nx*2)
    # return F.interpolate(w_sub, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)

def get_wt_T(test_data, shape):
    '''
    test_data: [batch_size, 1+3*len(shape) or 2*(1+3*len(shape)), nt, nx]
    return: list [batch_size, 64], [batch_size, 3, 64] * 3 layers
    '''
    wt_T = []
    wt_T.append(test_data[:,0,shape[-1][-2]-1])
    for i in range(len(shape)):
        wt_T.append(test_data[:,1+3*i:1+3*(i+1),shape[i][-2]-1])
    return wt_T

