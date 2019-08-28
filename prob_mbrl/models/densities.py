import numpy as np
import torch
import warnings

from torch.nn.functional import log_softmax
try:
    from torch.distributions.utils import log_sum_exp as logsumexp
except ImportError:
    logsumexp = torch.logsumexp
from prob_mbrl.models.modules import StochasticModule

PI = {'default': torch.tensor(np.pi)}
TWO_PI = {'default': 2 * PI['default']}
LOG_TWO_PI = {'default': torch.log(TWO_PI['default'])}
HALF_LOG_TWO_PI = {'default': 0.5 * LOG_TWO_PI['default']}


class CategoricalDensity(StochasticModule):
    def __init__(self, output_dims):
        super(CategoricalDensity, self).__init__()
        self.output_dims = output_dims
        self.register_buffer('z', torch.ones([1, 1]))

    def resample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.distributions.utils.clamp_probs(self.z)
        self.z.data = -(-u.log()).log()

    def forward(self,
                x,
                scaling_params=None,
                return_samples=False,
                output_noise=True,
                resample_output_noise=True,
                sampling_temperature=0.1,
                seed=None,
                **kwargs):
        D = int(self.output_dims)
        outs = x.split(D, -1)
        if len(outs) == 2:
            logits, log_temperature = outs
        else:
            logits = outs[0][:D]
        if return_samples:
            if (x.shape != self.z.shape) or resample_output_noise:
                if seed is not None:
                    torch.manual_seed(seed)
                u = torch.distributions.utils.clamp_probs(torch.rand_like(x))
                self.z.data = -(-u.log()).log()
            z = self.z
            # sample from gumbel softmax
            y_soft = ((log_softmax(x, -1) + z) /
                      sampling_temperature).softmax(-1)
            # sample from resulting categorical
            y_idx = torch.distributions.Categorical(y_soft).sample().view(
                -1, 1)
            y_hard = torch.zeros_like(y_soft).scatter(1, y_idx, 1)
            # get hard max (but backprop through softmax)
            y = ((y_hard - y_soft).detach() + y_soft)
            return y
        else:
            return logits

        def log_prob(self, z, logits):
            torch.distributions
            return


class DiagGaussianDensity(StochasticModule):
    '''
        Rearranges the incoming dimensions to correspond to the parameters
        of a Gaussian distribution, with diagonal covariance.
    '''

    def __init__(self, output_dims, max_noise_std=1.0):
        super(DiagGaussianDensity, self).__init__()
        self.output_dims = output_dims
        self.register_buffer('z', torch.ones([1, 1]))
        self.register_buffer('max_log_std', torch.tensor(max_noise_std).log())

    def resample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.z.data = torch.randn_like(self.z)

    def forward(self,
                x,
                scaling_params=None,
                return_samples=False,
                output_noise=True,
                resample_output_noise=True,
                seed=None,
                **kwargs):
        D = int(self.output_dims)
        mean, log_std = x.split(D, -1)

        log_std = -torch.nn.functional.softplus(-log_std) + self.max_log_std

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0]
                Sy = scaling_params[1]
                log_std = log_std + Sy.log()
                mean = mean * Sy + my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")

        if return_samples:
            samples = mean
            if output_noise:
                if (mean.shape != self.z.shape) or resample_output_noise:
                    if seed is not None:
                        torch.manual_seed(seed)
                    self.z.data = torch.randn_like(mean)
                z = self.z
                noise = z * log_std.clamp(-15, 15).exp()
                return samples, noise
            return samples
        else:
            return mean, log_std

    def log_prob(self, z, mean, log_std=None):
        ''' Computes the log likelihood for gaussian distributed predictions.
            This assumes diagonal covariances
        '''
        global HALF_LOG_TWO_PI
        D = self.output_dims
        deltas = mean - z
        # note that if noise is a 1xD vector, broadcasting
        # rules apply
        if log_std is not None:
            device_id = str(z.device.type) + str(z.device.index)
            if device_id not in HALF_LOG_TWO_PI:
                HALF_LOG_TWO_PI[device_id] = HALF_LOG_TWO_PI['default'].to(
                    z.device)
            stds = log_std.clamp(-15, 15).exp()
            lml = - 0.5 * ((deltas*stds.reciprocal())**2).sum(-1)\
                - log_std.sum(-1)\
                - D * HALF_LOG_TWO_PI[device_id]
        else:
            lml = -(deltas**2).sum(-1) * 0.5

        return lml

    def __repr__(self):
        return self.__class__.__name__ + '(output_dims=%d)' % (
            self.output_dims)


class GaussianMixtureDensity(StochasticModule):
    '''
     Mixture of Gaussian Density Network model. The components have diagonal
     covariance.
    '''

    def __init__(self, output_dims, n_components, max_noise_std=1.0, **kwargs):
        super(GaussianMixtureDensity, self).__init__(**kwargs)
        self.n_components = n_components
        self.output_dims = output_dims
        self.register_buffer('z_normal', torch.ones([1, 1]))
        self.register_buffer('z_pi', torch.ones([1, 1]))
        self.register_buffer('max_log_std', torch.tensor(max_noise_std).log())

    def resample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.distributions.utils.clamp_probs(torch.rand_like(self.z_pi))
        self.z_pi.data = -(-u.log()).log()
        self.z_normal.data = torch.randn_like(self.z_normal)

    def forward(self,
                x,
                scaling_params=None,
                return_samples=False,
                output_noise=True,
                resample_output_noise=True,
                sampling_temperature=0.1,
                seed=None,
                **kwargs):
        D = int(self.output_dims)
        nD = D * self.n_components
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outs = x.split(nD, -1)
        if len(outs) == 4:
            mean, log_std, logit_pi, log_temperature = outs
        else:
            mean, log_std, extras = outs
            logit_pi, log_temperature = extras.split(self.n_components, -1)

        log_std = -torch.nn.functional.softplus(-log_std) + self.max_log_std

        # the output shape is [batch_size, output_dimensions, n_components]
        mean = mean.view(-1, D, self.n_components)
        log_std = log_std.view(-1, D, self.n_components)
        temp = 1e-1 + torch.nn.functional.softplus(log_temperature)
        logit_pi = logit_pi / temp

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0].unsqueeze(-1)
                Sy = scaling_params[1].unsqueeze(-1)
                log_std = log_std + Sy.log()
                mean = mean * Sy + my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")

        if return_samples:
            if seed is not None:
                torch.manual_seed(seed)
            if (logit_pi.shape != self.z_pi.shape) or resample_output_noise:
                u = torch.distributions.utils.clamp_probs(
                    torch.rand_like(logit_pi))
                self.z_pi.data = -(-u.log()).log()
            z1 = self.z_pi
            # sample from gumbel softmax
            k_soft = ((log_softmax(logit_pi, -1) + z1) /
                      sampling_temperature).softmax(-1)
            # k_idx = k_soft.argmax(-1).view(-1, 1)
            k_idx = torch.distributions.Categorical(k_soft).sample().view(
                -1, 1)
            k_hard = torch.zeros_like(k_soft).scatter(1, k_idx, 1)
            # get hard max (but backprop through softmax)
            k = ((k_hard - k_soft).detach() + k_soft)[:, None, :]
            samples = (mean * k).sum(-1)
            if output_noise:
                if (mean[:-1].shape != self.z_pi.shape)\
                        or resample_output_noise:
                    self.z_normal.data = torch.randn(*mean.shape[:-1],
                                                     device=mean.device)
                z2 = self.z_normal
                noise = z2 * (log_std * k).sum(-1).clamp(-15, 15).exp()

                return samples, noise
            return samples
        else:
            return mean, log_std, logit_pi

    def log_prob(self, z, mean, log_std, logit_pi):
        global HALF_LOG_TWO_PI
        D = self.output_dims
        device_id = str(z.device.type) + str(z.device.index)
        if device_id not in HALF_LOG_TWO_PI:
            HALF_LOG_TWO_PI[device_id] = HALF_LOG_TWO_PI['default'].to(
                z.device)
        # get deltas wrt each mixture component
        deltas = mean - z.unsqueeze(-1)

        # weighted probabilities
        stds = log_std.clamp(-15, 15).exp()
        log_norm = -D * HALF_LOG_TWO_PI[device_id] - log_std.sum(-2)
        dists = -0.5 * ((deltas * stds.reciprocal())**2).sum(-2)
        log_probs = log_softmax(logit_pi, -1) + log_norm + dists

        # total log probability
        return logsumexp(log_probs, dim=-1, keepdim=True)

    def __repr__(self):
        desc = '(output_dims=%d, n_components=%d)' % (self.output_dims,
                                                      self.n_components)
        return self.__class__.__name__ + desc
