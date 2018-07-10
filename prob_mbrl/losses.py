import numpy as np
import torch

PI = {'default': torch.tensor(np.pi)}
TWO_PI = {'default': 2*PI['default']}
LOG_TWO_PI = {'default': torch.log(TWO_PI['default'])}
HALF_LOG_TWO_PI = {'default': 0.5*LOG_TWO_PI['default']}


def gaussian_log_likelihood(targets, pred_means, pred_stds=None):
    ''' Computes the log likelihood for gaussian distributed predictions.
        This assumes diagonal covariances
    '''
    global HALF_LOG_TWO_PI
    deltas = pred_means - targets
    # note that if noise is a 1xD vector, broadcasting
    # rules apply
    if pred_stds is not None:
        device_id = str(targets.device.type)+str(targets.device.index)
        if device_id not in HALF_LOG_TWO_PI:
            HALF_LOG_TWO_PI[device_id] = HALF_LOG_TWO_PI['default'].to(
                targets.device)

        lml = -((deltas*pred_stds.reciprocal())**2).sum(-1)*0.5\
              - pred_stds.log().sum(-1)\
              - HALF_LOG_TWO_PI[device_id]
    else:
        lml = -(deltas**2).sum(-1)*0.5

    return lml


def gaussian_mixture_log_likelihood(targets, means, stds, pi):
    global TWO_PI
    device_id = str(targets.device.type)+str(targets.device.index)
    if device_id not in TWO_PI:
        TWO_PI[device_id] = TWO_PI['default'].to(targets.device)
    # get deltas wrt each mixture component
    deltas = means - targets[:, None, :]

    # weighted probabilities
    norm = ((TWO_PI[device_id]*(stds**2).prod(-1))**0.5).reciprocal()
    probs = pi*(norm*(-0.5*((deltas*stds.reciprocal())**2).sum(-1)).exp())

    # total probability
    probs = probs.sum(-1)
    return probs.log()


def quadratic_loss(states, target, Q):
    deltas = states - target
    return (deltas.mm(Q)*deltas).sum(-1)[:, None]


def quadratic_saturating_loss(states, target, Q):
    return 1 - (-0.5*quadratic_loss(states, target, Q)).exp()
