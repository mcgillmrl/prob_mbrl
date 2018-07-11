import torch
import warnings

from torch.nn import Parameter

from prob_mbrl.models import StochasticModule


class DiagGaussianDensity(StochasticModule):
    '''
        Rearranges the incoming dimensions to correspond to the parameters
        of a Gaussian Density distribution.
    '''
    def __init__(self, output_dims):
        super(DiagGaussianDensity, self).__init__()
        self.output_dims = output_dims
        self.z = Parameter(
            torch.ones([1, 1]), requires_grad=False)

    def forward(self, x, scaling_params=None, return_samples=False,
                output_noise=True, resample_output_noise=True, **kwargs):
        D = self.output_dims
        idx = torch.range(0, 2*D-1, dtype=torch.long, device=x.device)
        mean = x.index_select(-1, idx[:D])
        std = x.index_select(-1, idx[D:]).exp()

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0]
                Sy = scaling_params[1]
                std = std*Sy
                mean = mean*Sy + my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")
        if return_samples:
            samples = mean
            if output_noise:
                if (mean.shape != self.z.shape) or resample_output_noise:
                    self.z.data = torch.randn_like(mean)
                z = self.z
                samples = samples + z*std
            return samples
        else:
            return mean, std


class MixtureDensity(StochasticModule):
    '''
     Mixture of Gaussian Densities Network model. The components have diagonal
     covariance.
    '''
    def __init__(self, output_dims, n_components, **kwargs):
        super(MixtureDensity, self).__init__(**kwargs)
        self.n_components = n_components
        self.output_dims = output_dims
        self.z_normal = Parameter(
            torch.ones([1, 1]), requires_grad=False)
        self.z_pi = Parameter(
            torch.ones([1, 1]), requires_grad=False)

    def forward(self, x, scaling_params=None, return_samples=False,
                output_noise=True, resample_output_noise=True, **kwargs):
        D = self.output_dims
        nD = D*self.n_components
        # the output shape is [batch_size, output_dimensions, n_components]
        idx = torch.range(
            0, 2*nD+self.n_components-1, dtype=torch.long, device=x.device)
        mean = x.index_select(-1, idx[:nD]).view(-1, D, self.n_components)
        std = x.index_select(
            -1, idx[nD:2*nD]).view(-1, D, self.n_components).exp()
        logit_pi = x.index_select(-1, idx[2*nD:])
        pi = torch.nn.functional.softmax(logit_pi, -1)

        # scale and center outputs
        if scaling_params is not None and len(scaling_params) > 0:
            if len(scaling_params) == 2:
                my = scaling_params[0].unsqueeze(-1)
                Sy = scaling_params[1].unsqueeze(-1)
                std = std*Sy
                mean = mean*Sy + my
            else:
                warnings.warn(
                    "Expected scaling_params as tuple or list with 2 elements")
        if return_samples:
            if (pi.shape != self.z_pi.shape) or resample_output_noise:
                self.z_pi.data = torch.rand_like(pi)
            z1 = self.z_pi
            k = (torch.log(pi) + z1).argmax(-1)
            k = k[:, None, None].repeat(1, mean.shape[-2], 1)
            samples = mean.gather(-1, k).squeeze()
            if output_noise:
                if (mean[:-1].shape != self.z_pi.shape) or resample_output_noise:
                    self.z_normal.data = torch.randn(*mean.shape[:-1])
                z2 = self.z_normal
                samples = samples + z2*std.gather(-1, k).squeeze()
            return samples
        else:
            return mean, std, pi