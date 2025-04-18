def upsample_coef(w_sub, shape, type):
    '''
    upsample the wavelet coefficients to the target shape
    w_sub: [N, nt, layer, nx, nx]
    tyoe: 'time' or 'space'
    w: [N, 2*nt, layer, nx, nx] or [N, nt, layer, 2*nx, 2*nx]
    '''
    N, nt, l, nx = w_sub.shape[0], w_sub.shape[1], w_sub.shape[2], w_sub.shape[3]
    if type == 'time':
        w_sub = w_sub.unsqueeze(-4).unsqueeze(-2).unsqueeze(-1).expand(N, nt, 2, l, nx, 1, nx, 1)
        return w_sub.reshape(N, nt*2, l, nx, nx)
    else:
        w_sub = w_sub.unsqueeze(-4).unsqueeze(-2).unsqueeze(-1).expand(N, nt, 1, l, nx, 2, nx, 2)
        return w_sub.reshape(N, nt, l, nx*2, nx*2)