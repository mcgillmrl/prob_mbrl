import torch
import numpy as np

from . import core
from . import modules
from . import activations


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
        self.X_mean = torch.tensor([[0.]])
        self.LX = torch.tensor([[1.]])
        self.iLX = torch.tensor([[1.]])
        self.Y_mean = torch.tensor([[0.]])
        self.LY = torch.tensor([[1.]])
        self.iLY = torch.tensor([[1.]])

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
        X_mean = X.mean(0, keepdim=True)
        X_delta = X - X_mean + 1e-4 * X.std(0)
        LX = 2 * (X_delta.T.mm(X_delta) / (X.shape[0] - 1)).cholesky()
        iLX = LX.inverse()
        Y_mean = Y.mean(0, keepdim=True)
        Y_delta = Y - Y_mean + 1e-4 * Y.std(0)
        LY = 2 * (Y_delta.T.mm(Y_delta) / (Y.shape[0] - 1)).cholesky()
        iLY = LY.inverse()
        self.X_mean, self.LX, self.iLX = X_mean, LX, iLX
        self.Y_mean, self.LY, self.iLY = Y_mean, LY, iLY

    def rescale(self, x, X_mean, iLX):
        transform = torch.distributions.transforms.AffineTransform(
            -X_mean.mm(iLX), iLX)
        return transform(x)

    def rescale_dist(self, dist, Y_mean, LY):
        transform = torch.distributions.transforms.AffineTransform(Y_mean, LY)
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

    def forward(self,
                x,
                y_true=None,
                temperature=1.0,
                seed=None,
                n_samples=torch.Size([]),
                **kwargs):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # rescale inputs
        x = self.rescale(x, self.X_mean, self.iLX)

        # forward through mlp
        params = self.base_model(x, **kwargs)

        # get output distribution
        dist, dist_params = self.get_dist(params, temperature)

        # rescale output distribution
        dist = self.rescale_dist(dist, self.Y_mean, self.LY)
        dist_params = self.rescale_params(dist_params, self.Y_mean, self.iLY,
                                          self.LY)

        # compute outputs
        if y_true is not None:
            return dist.log_prob(y_true), dist_params
        else:
            samples = dist.rsample(n_samples)
            log_probs = dist.log_prob(samples)
            return samples, log_probs, dist_params


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

    def get_dist(self, params, temperature):
        D = int(params.shape[-1])
        dist = torch.distributions.OneHotCategorical(logits=params /
                                                     temperature)
        return dist, dict(logits=params)


class RelaxedSoftmaxDN(ConditionalDensityModel):
    def __init__(self, model):
        super().__init__(model)

    def get_dist(self, params, temperature):
        D = int(params.shape[-1])
        dist = torch.distributions.RelaxedOneHotCategorical(1.0,
                                                            logits=params /
                                                            temperature)
        return dist, dict(logits=params)


def density_network_mlp(inputs,
                        outputs,
                        density_model=GaussianDN,
                        hids=[200, 200],
                        drop_rate=0.1,
                        activation=activations.hhSinLU):
    '''
        Utility method to build single gaussian model
    '''
    net = core.mlp(inputs,
                   density_model.n_params(outputs),
                   hids,
                   dropout_layers=[
                       modules.CDropout(drop_rate * torch.ones(hid))
                       for hid in hids
                   ],
                   nonlin=activation)
    model = density_model(net)
    return model


def mixture_density_network_mlp(inputs,
                                outputs,
                                nc=5,
                                density_model=GaussianMDN,
                                hids=[200, 200],
                                drop_rate=0.1,
                                activation=activations.hhSinLU):
    '''
        Utility method to build single gaussian model
    '''
    net = core.mlp(inputs,
                   density_model.n_params(outputs, nc),
                   hids,
                   dropout_layers=[
                       modules.CDropout(drop_rate * torch.ones(hid))
                       for hid in hids
                   ],
                   nonlin=activation)
    model = density_model(net, nc)
    return model
