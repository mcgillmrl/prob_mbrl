import torch
import numpy as np
import numbers

from . import core
from . import modules
from . import activations


class ScalingTransform(torch.distributions.transforms.Transform):
    r"""
    Transform with a scaling transform :math:`y = \text{loc} + \text{L}x`.
    where L is a 

    Args:
        loc (Tensor or float): Location parameter.
        L (Tensor): Lower triangular matrix corresponding to the cholesky factorization.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    domain = torch.distributions.transforms.constraints.real
    codomain = torch.distributions.transforms.constraints.real
    bijective = True

    def __init__(self, loc, L, event_dim=0, cache_size=0):
        super(ScalingTransform, self).__init__(cache_size=cache_size)
        self.loc = loc
        self.L = L
        self.event_dim = event_dim

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ScalingTransform(self.loc,
                                self.L,
                                self.event_dim,
                                cache_size=cache_size)

    def __eq__(self, other):
        if not isinstance(other, ScalingTransform):
            return False

        if isinstance(self.loc, numbers.Number) and isinstance(
                other.loc, numbers.Number):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():
                return False

        if isinstance(self.L, numbers.Number) and isinstance(
                other.L, numbers.Number):
            if self.L != other.L:
                return False
        else:
            if not (self.L == other.L).all().item():
                return False
        return True

    @property
    def sign(self):
        if isinstance(self.L, numbers.Number):
            return 1 if self.L > 0 else -1 if self.L < 0 else 0
        return self.L.diagonal().sign()

    def _call(self, x):
        return x.matmul(self.L) + self.loc

    def _inverse(self, y):
        return torch.triangular_solve((y - self.loc)[..., :, None],
                                      self.L,
                                      upper=False,
                                      transpose=True)[0].squeeze(-1)

    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        L = self.L
        if isinstance(L, numbers.Number):
            result = torch.full_like(x, math.log(abs(L)))
        else:
            result = torch.abs(L.diagonal(0, -2, -1).prod(-1,
                                                          keepdim=True)).log()
        if self.event_dim:
            result_size = result.size()[:-self.event_dim] + (-1, )
            result = result.view(result_size).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)


class ConditionalDensityModel(torch.nn.Module):
    '''
        Implements conditional density models, consisting of a base model
        that predicts parameters for an output distribution. This can
        be used to generate samples or evaluate the density.

        Subclasses need to implement:
            * the `n_params` method: returns the total number of parameters
                                   output by the base model
            * the `get_dist` method: returns a torch.distribution object
                                     instantiated with the parameters predicted
                                     by the base model
    '''
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.X_mean = None
        self.LX = None
        self.iLX = None
        self.Y_mean = None
        self.LY = None
        self.iLY = None

    @staticmethod
    def n_params(D):
        return D

    def get_dist(self, params, temperature):
        ones = torch.ones_like(params)
        dist = torch.distributions.Normal(params, ones)
        dist_params = dict(mu=params, sqrtSigma=ones * temperature)
        return dist, dist_params

    def resample(self):
        if hasattr(self.base_model, 'resample'):
            self.base_model.resample()

    def set_scaling(self, X, Y):
        self.set_input_scaling(X)
        self.set_output_scaling(Y)

    def set_input_scaling(self, X):
        X_mean = X.mean(0, keepdim=True)
        X_delta = X - X_mean + 1e-4 * X.std(0)
        LX = 2 * (X_delta.T.mm(X_delta) / (X.shape[0] - 1)).cholesky()
        iLX = LX.inverse()
        self.X_mean, self.LX, self.iLX = X_mean, LX, iLX

    def set_output_scaling(self, Y):
        Y_mean = Y.mean(0, keepdim=True)
        Y_delta = Y - Y_mean + 1e-4 * Y.std(0)
        LY = 2 * (Y_delta.T.mm(Y_delta) / (Y.shape[0] - 1)).cholesky()
        iLY = LY.inverse()
        self.Y_mean, self.LY, self.iLY = Y_mean, LY, iLY

    def rescale(self, x, X_mean, iLX):
        transform = ScalingTransform(-X_mean.mm(iLX), iLX)
        return transform(x)

    def rescale_dist(self, dist, Y_mean, LY):
        transform = ScalingTransform(Y_mean, LY)
        return torch.distributions.TransformedDistribution(dist, transform)

    def rescale_params(self, dist_params, Y_mean, iLY, LY):
        if 'mu' in dist_params:
            dist_params['mu'] = self.rescale(dist_params['mu'],
                                             -Y_mean.mm(iLY), LY)
        if 'sqrtSigma' in dist_params:
            dist_params['sqrtSigma'] = self.rescale(dist_params['sqrtSigma'],
                                                    torch.zeros_like(Y_mean),
                                                    LY)
        return dist_params

    def regularization_loss(self):
        return self.base_model.regularization_loss()

    def forward(self, x, temperature=1.0, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if self.X_mean is not None and self.iLX is not None:
            # rescale inputs
            x = self.rescale(x, self.X_mean, self.iLX)

        # forward through mlp
        params = self.base_model(x, **kwargs)

        # get output distribution
        dist, dist_params = self.get_dist(params, temperature)

        if self.Y_mean is not None and self.iLY is not None:
            # rescale output distribution
            dist = self.rescale_dist(dist, self.Y_mean, self.LY)
            dist_params = self.rescale_params(dist_params, self.Y_mean,
                                              self.iLY, self.LY)

        return dist, dist_params


class GaussianDN(ConditionalDensityModel):
    '''
        Gaussian Density Network with full covariance matrix
    '''
    @staticmethod
    def n_params(D):
        # D for mean + D for each of u, v, d used for
        # constructing the sqrtSigma matrix
        return 4 * D

    def get_dist(self, params, temperature):
        D = int(params.shape[-1] / 4)

        # extract params from mlp output
        mu = params[..., 0:D]
        u, v, d = params[..., D:params.shape[-1]].view(-1, 3, D,
                                                       1).transpose(0, 1)
        sqrtSigma = u.matmul(v.transpose(
            -1, -2)).tril(-1) + torch.eye(D) * d.exp()
        sqrtSigma = temperature * sqrtSigma
        shp = list(params.shape[0:-1])
        sqrtSigma = sqrtSigma.view(shp + [D, D])

        # prepare output distribution
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=sqrtSigma)

        return dist, dict(mu=mu, sqrtSigma=sqrtSigma)


class RelaxedMixtureSameFamily(torch.distributions.MixtureSameFamily):
    '''
        Mixture distribution with reparametrized sampling for
        pathwise derivatives
    '''
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 temperature=0.1):
        super().__init__(mixture_distribution, component_distribution)
        self.temperature = temperature

    def rsample(self, sample_shape=torch.Size()):
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len
        es = self.event_shape

        # mixture samples [n, B, k]
        relaxed_mix = torch.distributions.RelaxedOneHotCategorical(
            self.temperature, logits=self.mixture_distribution.logits)
        mix_simplex = relaxed_mix.rsample(sample_shape)
        # here we take the argmax, we can alternatively sample from the
        # simplex
        mix_id = mix_simplex.argmax(-1).view(-1, 1)

        # get indicator for the max
        nc = mix_simplex.shape[-1]
        mix_sample = torch.zeros_like(mix_simplex.view(-1, nc)).scatter(
            1, mix_id, 1).view(mix_simplex.shape)

        # get mix_sample but backprop through mix_simplex
        mix_sample = ((mix_sample - mix_simplex).detach() + mix_simplex)

        # component samples [n, B, k, E]
        comp_samples = self.component_distribution.rsample(sample_shape)

        # since only one entry in mix_sample will be equal to one, this
        # recovers the samples from the selected component
        samples = (comp_samples * mix_sample[..., None]).sum(-2)

        return samples


class GaussianMDN(ConditionalDensityModel):
    '''
        Gaussian Mixture Density Network where
        each component is a multivariate normal with a 
        full covariance matrix
    '''
    def __init__(self, model, n_components):
        super().__init__(model)
        self.nc = n_components

    @staticmethod
    def n_params(D, n_components):
        return (4 * D + 1) * n_components

    def get_dist(self, params, temperature):
        nc = self.nc
        D = int((params.shape[-1] / nc - 1) / 4)

        # extract params from mlp output
        shp = list(params.shape[0:-1])
        mu = params[..., 0:D * nc].view(shp + [nc, D])
        u, v, d = params[..., D * nc:4 * D * nc].view(-1, 3, nc, D,
                                                      1).transpose(0, 1)
        sqrtSigma = u.matmul(v.transpose(
            -1, -2)).tril(-1) + torch.eye(D) * d.exp()
        sqrtSigma = temperature * sqrtSigma
        sqrtSigma = sqrtSigma.view(shp + [nc, D, D])
        logit_pi = params[..., 4 * D *
                          nc:params.shape[-1]].view(shp + [nc]) / temperature

        # prepare output distribution
        mix = torch.distributions.Categorical(logits=logit_pi)
        comp = torch.distributions.MultivariateNormal(mu, scale_tril=sqrtSigma)
        dist = RelaxedMixtureSameFamily(mix, comp, temperature)
        return dist, dict(mu=mu, sqrtSigma=sqrtSigma, logit_pi=logit_pi)


class SoftmaxDN(ConditionalDensityModel):
    def __init__(self, model):
        super().__init__(model)

    def rescale_dist(self, dist, Y_mean, LY):
        return dist

    def get_dist(self, params, temperature):
        D = int(params.shape[-1])
        dist = torch.distributions.OneHotCategorical(logits=params /
                                                     temperature)
        return dist, dict(logits=params)


class RelaxedSoftmaxDN(SoftmaxDN):
    def __init__(self, model):
        super().__init__(model)

    def get_dist(self, params, temperature):
        D = int(params.shape[-1])
        dist = torch.distributions.RelaxedOneHotCategorical(
            torch.tensor([0.1]), logits=params / temperature)
        return dist, dict(logits=params)


def density_network_mlp(inputs,
                        outputs,
                        density_model=GaussianDN,
                        hids=[200, 200],
                        dropout=0.1,
                        activation=torch.nn.ReLU):
    '''
        Utility method to build single gaussian model
    '''
    if isinstance(dropout, numbers.Number):
        dropout = [modules.CDropout(dropout * torch.ones(hid)) for hid in hids]

    net = core.mlp(inputs,
                   density_model.n_params(outputs),
                   hids,
                   dropout_layers=dropout,
                   nonlin=activation)
    model = density_model(net)
    return model


def mixture_density_network_mlp(inputs,
                                outputs,
                                nc=5,
                                density_model=GaussianMDN,
                                hids=[200, 200],
                                dropout=0.1,
                                activation=torch.nn.ReLU):
    '''
        Utility method to build a mixture of gaussians model
    '''
    if isinstance(dropout, numbers.Number):
        dropout = [modules.CDropout(dropout * torch.ones(hid)) for hid in hids]

    net = core.mlp(inputs,
                   density_model.n_params(outputs, nc),
                   hids,
                   dropout_layers=dropout,
                   nonlin=activation)
    model = density_model(net, nc)
    return model
