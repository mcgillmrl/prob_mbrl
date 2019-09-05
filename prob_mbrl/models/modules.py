import torch

from torch import nn
from torch.nn import Parameter
from math import pi, e

jit_scripts = {}


class StochasticModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(StochasticModule, self).__init__(*args, **kwargs)


class BDropout(StochasticModule):
    """
        Extends the base Dropout layer by adding a regularizer as derived by
        Gal and Ghahrahmani "Dropout as a Bayesian Approximation" (2015)
    """

    def __init__(self, rate=0.5, name=None, regularizer_scale=1.0, **kwargs):
        super(BDropout, self).__init__(**kwargs)
        self.name = name
        self.register_buffer('regularizer_scale',
                             torch.tensor(0.5 * regularizer_scale))
        self.register_buffer('rate', torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer('noise', torch.bernoulli(self.p))

    def weights_regularizer(self, weights):
        self.p = 1 - self.rate
        return self.regularizer_scale * ((weights**2).sum(0) / self.p).sum()

    def biases_regularizer(self, biases):
        return self.regularizer_scale * ((biases**2).sum(0) / self.p**2).sum()

    def resample(self, seed=None):
        self.update_noise(self.noise, seed)

    def update_noise(self, x, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=True, mask_dims=2, seed=None, **kwargs):
        sample_shape = x.shape[-mask_dims:]
        if (sample_shape[1:] != self.noise.shape[1:]
                or sample_shape[0] > self.noise.shape[0]):
            # resample if we can't re-use old numbers
            # this happens when the incoming batch size is bigger than
            # the noise batch size, or when the rest of the shape differs
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample, seed)
        elif resample:
            if seed is not None:
                torch.manual_seed(seed)
            return (x * torch.bernoulli(self.p.expand(x.shape))) / self.p

        # we never need the noise gradients
        return (x * self.noise[..., :x.shape[-mask_dims], :].detach()) / self.p

    def extra_repr(self):
        return 'rate={}, regularizer_scale={}'.format(self.rate,
                                                      self.regularizer_scale)


class CDropout(BDropout):
    def __init__(self,
                 rate=0.5,
                 name=None,
                 regularizer_scale=1.0,
                 dropout_regularizer=1.0,
                 temperature=0.1,
                 **kwargs):
        super(CDropout, self).__init__(rate, name, regularizer_scale, **kwargs)
        self.register_buffer('temp', torch.tensor(temperature))
        self.register_buffer('dropout_regularizer',
                             torch.tensor(dropout_regularizer))
        self.logit_p = Parameter(-torch.log(1.0 / self.p - 1.0))
        self.register_buffer('concrete_noise', torch.bernoulli(self.p))

    def weights_regularizer(self, weights):
        # logit_p = torch.nn.functional.softplus(self.logit_p, 50)
        p = self.p
        reg = self.regularizer_scale * ((weights**2).sum(0) / p)
        reg += self.dropout_regularizer * (p * p.log() + (1 - p) *
                                           (1 - p).log())
        return reg.sum()

    def update_noise(self, x, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.noise.data = torch.rand_like(x)
        if not self.training:
            self.update_concrete_noise(self.noise)

    def update_concrete_noise(self, noise):
        """Updates the concrete dropout masks.

        Args:
            noise (Tensor): Input.
        """
        noise_p = noise + 1e-7
        noise_m = noise - 1e-7
        #logit_p = torch.nn.functional.softplus(self.logit_p, 50)
        concrete_p = self.logit_p + (noise_p / (1 - noise_m)).log()
        probs = (concrete_p / self.temp + 1e-9).sigmoid()
        noise = torch.bernoulli(probs)
        # forward pass uses bernoulli sampled noise, but backwards
        # through concrete distribution
        self.concrete_noise = (noise - probs).detach() + probs
        self.p = self.logit_p.sigmoid()

    def forward(self, x, resample=False, mask_dims=2, seed=None, **kwargs):
        """Computes the concrete dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        resampled = False
        if resample:
            if seed is not None:
                torch.manual_seed(seed)
            noise = torch.rand_like(x)
            resampled = True
        elif (sample_shape[1:] != self.noise.shape[1:]
              or sample_shape[1:] != self.concrete_noise.shape[1:]
              or sample_shape[0] > self.concrete_noise.shape[0]):
            # resample if we can't re-use old numbers
            # this happens when the incoming batch size is bigger than
            # the noise batch size, or when the rest of the shape differs
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample, seed)
            noise = self.noise
            resampled = True

        if self.training:
            self.update_concrete_noise(noise)
            concrete_noise = self.concrete_noise
            p = self.p
        else:
            if resampled:
                self.update_concrete_noise(noise)
            # We never need these gradients in evaluation mode.
            concrete_noise = self.concrete_noise.detach()
            p = self.p.detach()

        return (x * concrete_noise[..., :x.shape[-mask_dims], :]) / p

    def extra_repr(self):
        #logit_p = torch.nn.functional.softplus(self.logit_p, 50)
        return 'rate={}, temperature={}, regularizer_scale={}'.format(
            1 - self.logit_p.sigmoid(), self.temp, self.regularizer_scale)


class TLNDropout(BDropout):
    '''
        'Implements truncated log-normal dropout (NIPS 2017)
    '''

    def __init__(self, in_dim, 
                rate=0.5,
                 name=None,
                 regularizer_scale=1.0, 
         interval=[-10, 0], **kwargs,):
        super(TLNDropout, self).__init__(rate, name, regularizer_scale, **kwargs)

        self.register_buffer('a', torch.tensor(interval[0]).float())
        self.register_buffer('b', torch.tensor(interval[1]).float())

        #scaling terms for the bounded sigma range
        uniform_std = torch.sqrt(torch.pow(self.b - self.a, 2) / 12.0 )
        self.register_buffer('s_max', uniform_std)
        self.register_buffer('s_min', torch.tensor(1e-2))

        a = self.a
        b = self.b
        mu0 = max(a + 1e-2*(b-a), 0) + min(b - 1e-2*(b-a), 0)
        logit_u0 = -torch.log((b-a)/(mu0 - a) - 1)

        #to be exaaactly like them, I think this has to be one number..
        self.logit_mu = Parameter(torch.ones([1, in_dim]) * logit_u0)
        #not 100% but pretty sure they sample all of them...
        self.logit_sig = Parameter(torch.ones([1, in_dim]).uniform_(-3.0, -1.0))
        
        self.register_buffer('unit_mu', torch.zeros_like(self.logit_mu)) 
        self.register_buffer('unit_sig', torch.ones_like(self.logit_sig))

    def pdf(self, x):
        #littl phi in paper
        #probability density function of unit normal
        return torch.exp(-0.5 * torch.pow(x, 2.)) / torch.sqrt(torch.tensor(2. * pi))
    
    def cdf(self, x):
        #Big phi in paper
        #cumulative density function of unit normal
        return 0.5 * (torch.erfc(-x / torch.sqrt(torch.tensor(2.))))
    
    def inv_cdf(self, y):
        #pytorch doesn't have a convenient erfcinv function like theano
        unit_normal = torch.distributions.Normal(self.unit_mu, self.unit_sig)
        return unit_normal.icdf(y)
        
    def weights_regularizer(self, weights):
        '''
        In this case the weights regularizer is actually independent of the
        weights (only depends on the alpha parameter)
        '''
        _, sig, beta, alpha, Z = self.get_parameters()
        
        #calculations for kl term below
        
        term1 = torch.log( self.b - self.a) - torch.log(torch.sqrt(torch.tensor(2. * pi * e)) * sig )
        
        #TODO: the / sig in dense_diff is from kusanagi code base...not in paper from what I can tell 
        dense_diff = (alpha * self.pdf(alpha)  - beta * self.pdf(beta)) / sig
        
        #term 2 and 3 calculations
        term2 = torch.log(Z)
        term3 = dense_diff / (2 * Z)
        
        kl = (term1 - term2 - term3).sum() # is mean in original paper's code base,

        return kl * self.regularizer_scale

    def update_noise(self, x, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.noise.data = torch.rand_like(x)
        
        if not self.training:
            self.update_truncated_lognormal_noise(self.noise)
            
    def get_parameters(self): 
        sig = (self.s_max -  self.s_min) * torch.sigmoid(self.logit_sig) + self.s_min
        mu =  (self.b - self.a) * torch.sigmoid(self.logit_mu) + self.a
        
        #values used in all terms
        beta = (self.b - mu) / sig
        alpha = (self.a - mu) / sig
        Z = self.cdf(beta) - self.cdf(alpha)
        
        return mu, sig, beta, alpha, Z

    def update_truncated_lognormal_noise(self, noise):
        #input: noise: the actual dimensions of noise to be applied
        mu, sig, _, alpha, Z  = self.get_parameters()

        gamma = self.cdf(alpha) + Z * noise
        
        self.tln_noise = torch.exp(mu + sig * self.inv_cdf(torch.clamp(gamma, min=1e-5, max=1-1e-5)))
        return self.tln_noise 
        
    def forward(self, x, resample=False, mask_dims=2, seed=None, **kwargs):
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        resampled = False
        if resample:
            if seed is not None:
                torch.manual_seed(seed)
            noise = torch.rand_like(x)
            resampled = True
        elif (sample_shape[1:] != self.noise.shape[1:]
              or sample_shape[1:] != self.tln_noise.shape[1:]
              or sample_shape[0] > self.tln_noise.shape[0]):
            # resample if we can't re-use old numbers
            # this happens when the incoming batch size is bigger than
            # the noise batch size, or when the rest of the shape differs
            sample = x.view(-1, *sample_shape)[0]
            self.update_noise(sample, seed)
            noise = self.noise
            resampled = True

        if self.training:
            self.update_truncated_lognormal_noise(noise)
            tln_noise = self.tln_noise
            p = self.p
        else:
            if resampled:
                self.update_truncated_lognormal_noise(noise)
            # We never need these gradients in evaluation mode.
            tln_noise = self.tln_noise.detach()
            p = self.p.detach()

        return x * tln_noise[..., :x.shape[-mask_dims], :]  



class BSequential(nn.modules.Sequential):
    " An extension to sequential that allows for controlling resampling"

    def __init__(self, *args):
        super(BSequential, self).__init__(*args)

    def resample(self, seed=None):
        i = 0
        for module in self._modules.values():
            if isinstance(module, BDropout):
                if seed is not None:
                    module.resample(seed + i)
                i += 1

    def forward(self, input, resample=True, repeat_mask=False, **kwargs):
        modules = list(self._modules.values())
        for i, module in enumerate(modules):
            if isinstance(module, BDropout) or isinstance(
                    module, torch.nn.Dropout):
                if i < len(modules):
                    next_module = modules[i + 1]
                    if isinstance(next_module, SpectralNorm):
                        # rescale lipschitz constant by dropout probability
                        pass
            if isinstance(module, StochasticModule):
                input = module(input,
                               resample=resample,
                               repeat_mask=repeat_mask,
                               **kwargs)
            else:
                input = module(input)
        return input

    def regularization_loss(self):
        modules = list(self._modules.values())
        reg_loss = 0
        for i, module in enumerate(modules):
            if hasattr(module, 'weights_regularizer'):
                # find first subsequent module, from current,output_samples
                # with a weight attribute
                for next_module in modules[i:]:
                    if isinstance(next_module, SpectralNorm):
                        next_module = next_module.module
                    if isinstance(next_module, nn.Linear)\
                            or isinstance(next_module,
                                          nn.modules.conv._ConvNd):
                        reg_loss += module.weights_regularizer(
                            next_module.weight)
                        if hasattr(next_module, 'bias')\
                                and next_module.bias is not None\
                                and hasattr(module, 'biases_regularizer'):
                            reg_loss += module.biases_regularizer(
                                next_module.bias)
                        break
            elif hasattr(module, 'regularization_loss'):
                reg_loss += module.regularization_loss()
        return reg_loss


class SpectralNorm(torch.nn.Module):
    """
        Applies spectral normalization to the weights matrix, i.e.
        W_sn = W/sigma(W), where sigma(W) is the largest eigenvalue
        of W
    """

    def __init__(self,
                 module,
                 power_iterations=1,
                 max_K=10,
                 param_name='weight',
                 train_scale=False):
        assert hasattr(module, param_name)
        super(SpectralNorm, self).__init__()
        self.module = module
        self.param_name = param_name
        self.n_iter = power_iterations
        self.max_K = max_K
        self.init_params()
        self.scale = torch.nn.Parameter(torch.zeros(1),
                                        requires_grad=train_scale)

    def extra_repr(self):
        if hasattr(self.module, self.param_name):
            w = getattr(self.module, self.param_name)
        else:
            w = getattr(self.module, self.param_name + '_bar')
        return "scale={}, norm={}".format(
            (self.max_K * self.scale.sigmoid()).data,
            torch.svd(w)[1][0])

    def init_params(self):
        w = self.module._parameters[self.param_name]
        w_sn = torch.nn.Parameter(w.data)
        M = w.shape[0]
        N = w.view(M, -1).shape[0]
        u = torch.randn(M).to(w.device, w.dtype)
        u.data = u / (u.norm() + 1e-12)
        v = torch.randn(N).to(w.device, w.dtype)
        v.data = v / (v.norm() + 1e-12)

        self.module.register_parameter(self.param_name + "_bar", w_sn)
        self.module.register_buffer(self.param_name + "_u", u)
        self.module.register_buffer(self.param_name + "_v", v)

        del self.module._parameters[self.param_name]

    def power_iteration(self, n_iters=1):
        u = getattr(self.module, self.param_name + '_u')
        v = getattr(self.module, self.param_name + '_v')
        w = getattr(self.module, self.param_name + '_bar')
        M = w.shape[0]
        w_square = w.view(M, -1)

        for i in range(n_iters):
            v_ = torch.mv(w_square.data.transpose(0, 1), u.data)
            v.data = v_ / (v_.norm() + 1e-12)
            u_ = torch.mv(w_square.data, v.data)
            u.data = u_ / (u_.norm() + 1e-12)

        sigma_w = u.dot(w.view(M, -1).mv(v))
        setattr(self.module, self.param_name,
                self.max_K * self.scale.sigmoid() * (w / sigma_w.expand_as(w)))

    def forward(self, *args, **kwargs):
        if self.training:
            self.power_iteration(self.n_iter)
        return self.module(*args, **kwargs)
