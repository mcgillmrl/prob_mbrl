import numpy as np


def gaussian_log_likelihood(targets, pred_means, pred_stds=None):
    ''' Computes the log likelihood for gaussian distributed predictions.
        This assumes diagonal covariances
    '''
    deltas = pred_means - targets
    # note that if noise is a 1xD vector, broadcasting
    # rules apply
    if pred_stds is not None:
        lml = -((deltas/pred_stds)**2).sum(-1)*0.5\
              - pred_stds.log().sum(-1)\
              - np.log(2*np.pi)*0.5
    else:
        lml = -(deltas**2).sum(-1)*0.5

    return lml


def quadratic_loss(states, target, Q):
    deltas = states - target
    return (deltas.mm(Q)*deltas).sum(-1)[:, None]


def quadratic_saturating_loss(states, target, Q):
    return 1 - (-0.5*quadratic_loss(states, target, Q)).exp()
