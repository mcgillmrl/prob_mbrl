import torch

ODIMS = {}


def to_complex(x, dims=[]):
    global ODIMS
    if len(dims) == 0:
        return x
    else:
        dims = torch.tensor(dims).to(x.device)

        # check cache for indices
        odims_id = (x.shape[-1], dims)
        if odims_id in ODIMS:
            odims = ODIMS[odims_id]
        else:
            # build indices for other dimensions
            odims = torch.arange(0, x.shape[-1]).long().to(dims.device)
            odims = (
                1 - torch.eq(dims, odims[:, None])).prod(1).nonzero()[:, 0]
            ODIMS[odims_id] = odims

        angles = x.index_select(-1, dims)
        others = x.index_select(-1, odims)

        return torch.cat([others, angles.sin(), angles.cos()], -1)
