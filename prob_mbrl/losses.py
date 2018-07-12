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
    '''
        Returns the log probability of targets under the mixture
        distribution parametrized by means, stds and pi.
        This assumes a mixture of gaussians we diagonal covariance.
        The expected shape for mean and std is
            [batch_size, output_dimensions, n_components]
    '''
    global HALF_LOG_TWO_PI
    device_id = str(targets.device.type)+str(targets.device.index)
    if device_id not in HALF_LOG_TWO_PI:
        HALF_LOG_TWO_PI[device_id] = HALF_LOG_TWO_PI['default'].to(
            targets.device)
    # get deltas wrt each mixture component
    deltas = means - targets[:, :, None]

    # weighted probabilities
    log_stds = stds.log()
    log_norm = -HALF_LOG_TWO_PI[device_id] - (log_stds).sum(-2)
    dists = -0.5*((deltas*stds.reciprocal())**2).sum(-2)
    log_probs = pi.log() + log_norm + dists

    # total log probability
    return torch.distributions.utils.log_sum_exp(log_probs, keepdim=True)


def quadratic_loss(states, target, Q):
    target = target.to(states.device)
    Q = Q.to(states.device)
    deltas = states - target
    return (deltas.mm(Q)*deltas).sum(-1)[:, None]


def quadratic_saturating_loss(states, target, Q):
    return 1 - (-0.5*quadratic_loss(states, target, Q)).exp()
