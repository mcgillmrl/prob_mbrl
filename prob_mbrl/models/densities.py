import torch
import warnings

from torch.nn.functional import log_softmax
from prob_mbrl.models.modules import StochasticModule


class DiagGaussianDensity(StochasticModule):
    '''
        Rearranges the incoming dimensions to correspond to the parameters
        of a Gaussian Density distribution.
    '''

    def __init__(self, output_dims):
        super(DiagGaussianDensity, self).__init__()
        self.output_dims = output_dims
        self.register_buffer('z', torch.ones([1, 1]))

    def resample(self, *args, **kwargs):
        self.z.data = torch.randn_like(self.z)

    def forward(self,
                x,
                scaling_params=None,
                return_samples=False,
                output_noise=True,
                resample_output_noise=True,
                **kwargs):
        D = int(self.output_dims)
        mean, log_std = x.split(D, -1)

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
                    self.z.data = torch.randn_like(mean)
                z = self.z
                noise = z * log_std.clamp(-15, 15).exp()
                return samples, noise
            return samples
        else:
            return mean, log_std


class MixtureDensity(StochasticModule):
    '''
     Mixture of Gaussian Densities Network model. The components have diagonal
     covariance.
    '''

    def __init__(self, output_dims, n_components, **kwargs):
        super(MixtureDensity, self).__init__(**kwargs)
        self.n_components = n_components
        self.output_dims = output_dims
        self.register_buffer('z_normal', torch.ones([1, 1]))
        self.register_buffer('z_pi', torch.ones([1, 1]))

    def resample(self, *args, **kwargs):
        u = torch.distributions.utils.clamp_probs(torch.rand_like(self.z_pi))
        self.z_pi.data = -(-u.log()).log()
        self.z_normal.data = torch.randn_like(self.z_normal)

    def forward(self,
                x,
                scaling_params=None,
                return_samples=False,
                output_noise=True,
                resample_output_noise=True,
                temperature=1.0,
                **kwargs):
        D = int(self.output_dims)
        nD = D * self.n_components

        outs = x.split(nD, -1)
        if len(outs) == 4:
            mean, log_std, logit_pi, log_temperature = outs
        else:
            mean, log_std, extras = outs
            logit_pi, log_temperature = extras.split(self.n_components, -1)

        # the output shape is [batch_size, output_dimensions, n_components]
        mean = mean.view(-1, D, self.n_components)
        log_std = log_std.view(-1, D, self.n_components)
        logit_pi = logit_pi / (temperature +
                               (log_temperature.clamp(-15, 15)).exp())

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
            if (logit_pi.shape != self.z_pi.shape) or resample_output_noise:
                u = torch.distributions.utils.clamp_probs(
                    torch.rand_like(logit_pi))
                self.z_pi.data = -(-u.log()).log()
            z1 = self.z_pi
            # replace this sampling operation
            k = (log_softmax(logit_pi, -1) + z1).argmax(-1)
            k = k[:, None, None].repeat(1, mean.shape[-2], 1)
            samples = mean.gather(-1, k).squeeze(-1)
            if output_noise:
                if (mean[:-1].shape != self.z_pi.shape)\
                        or resample_output_noise:
                    self.z_normal.data = torch.randn(*mean.shape[:-1],
                                                     device=mean.device)
                z2 = self.z_normal
                noise = z2 * log_std.gather(-1, k).squeeze(-1).clamp(-15,
                                                                     15).exp()

                return samples, noise
            return samples
        else:
            return mean, log_std, logit_pi
