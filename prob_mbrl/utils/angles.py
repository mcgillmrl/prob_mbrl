import numpy as np
import torch

ODIMS = {}


def to_complex(x, dims):
    global ODIMS
    if len(dims) == 0:
        return x
    else:
        # check cache for indices
        odims_id = (x.dtype, getattr(x, 'device',
                                     'cpu'), x.shape[-1], tuple(dims))
        if odims_id in ODIMS:
            odims, dims = ODIMS[odims_id]
        else:
            odims, dims = build_odims_(x, dims)
            ODIMS[odims_id] = (odims, dims)

        if isinstance(x, torch.Tensor):
            return to_complex_(x, dims, odims)
        else:
            angles = np.atleast_1d(x[..., dims])
            others = np.atleast_1d(x[..., odims])
            return np.concatenate([others, np.sin(angles), np.cos(angles)], -1)


def build_odims_(x, dims):
    # build indices for other dimensions
    if not isinstance(dims, torch.Tensor):
        dims = torch.tensor(dims).to(x.device)
    odims = torch.arange(0, x.shape[-1]).long().to(dims.device)
    odims = torch.bitwise_not(torch.eq(dims,
                                       odims[:, None])).prod(1).nonzero()[:, 0]
    return odims.detach(), dims


def to_complex_(x, dims, odims):
    angles = x.index_select(-1, dims)
    others = x.index_select(-1, odims)
    return torch.cat([others, angles.sin(), angles.cos()], -1)