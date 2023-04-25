######
# Various kernels as used in Simpson et al. (2021)
#####

import numpy as np
import torch as th
import gpytorch as gpy
import torch.distributions as thd

# TODO: For now all lengthscales are shared over the dimensions -> Generalize
# TODO: Implement sampling of kernels


def scale_kernel(kernel):
    kernel = gpy.kernels.ScaleKernel(kernel)
    kernel.outputscale = th.exp(th.randn(1, 1))
    return kernel


def generate_rbf():
    kernel = gpy.kernels.RBFKernel()
    kernel.lengthscale = th.exp(th.randn(1, 1))
    return kernel


def generate_periodic():
    kernel = gpy.kernels.PeriodicKernel()
    kernel.lengthscale = th.exp(th.randn(1, 1))
    kernel.period_length = th.exp(th.randn(1, 1))
    return kernel


class WhiteNoise(gpy.kernels.Kernel):
    # This can probably be done more elegantly with newer pytorch versions
    # Adapted from https://github.com/cornellius-gp/gpytorch/blob/2fab88ea90f1785130313bb257a090fe07ef3f61/gpytorch/kernels/white_noise_kernel.py
    # TODO: Extend to multiple dimensions
    noise = th.exp(th.randn(1))

    def forward(self, x1, x2, **params):
        if (
            x1.shape[-2] == x2.shape[-2]
            and x1.shape[-2] == self.noise.shape[-1]
            and th.equal(x1, x2)
        ):
            return th.diag(self.noise)
        else:
            return th.zeros((x1.shape[-3], x1.shape[-2], x1.shape[-1]))


def generate_white():
    return WhiteNoise()


def generate_matern(nu):
    assert nu in [1 / 2, 3 / 2, 5 / 2]
    kernel = gpy.kernels.MaternKernel(nu=nu)
    kernel.lengthscale = th.exp(th.randn(1, 1))
    return kernel


def generate_cosine():
    kernel = gpy.kernels.CosineKernel()
    # TODO: Simpson claims not to enforce a positivity constraint on the lengthscale
    #   I don't get why that should work... and thus just take the absolute for now
    kernel.period_length = thd.Cauchy(0, 5).sample((1, 1)).abs()
    return kernel


class ModifiedLinear(gpy.kernels.Kernel):
    # Simpson et al. use K(x,z) = s^2 (x - c)(z - c) while gpy provides K(x,z) = s^2x * z
    is_stationary = False
    linear = gpy.kernels.LinearKernel()
    shift = thd.Cauchy(0, 5).sample((1, 1))

    def forward(self, x1, x2, **params):
        x1_shift = x1 - self.shift
        x2_shift = x2 - self.shift
        return self.linear(x1_shift, x2_shift)


def generate_linear():
    return ModifiedLinear()


def sample_basic_kernel(name=None, scale=False):
    if name == "EQ":
        kernel = generate_rbf()
    elif name == "periodic":
        kernel = generate_periodic()
    elif name == "white":
        kernel = generate_white()
    elif name == "matern12":
        kernel = generate_matern(1 / 2)
    elif name == "matern32":
        kernel = generate_matern(3 / 2)
    elif name == "matern52":
        kernel = generate_matern(5 / 2)
    elif name == "cosine":
        kernel = generate_cosine()
    elif name == "linear":
        kernel = generate_linear()
    else:
        raise NotImplementedError(f"{name} does not have a kernel implementation")

    if scale and name != "white":
        kernel = scale_kernel(kernel)

    return kernel


def get_names():
    "Not the nices formulation, but it works for now"
    foo = [
        "None",
        "EQ",
        "periodic",
        "white",
        "matern12",
        "matern32",
        "matern52",
        "cosine",
        "linear",
    ]

    bar = np.unique([sorted((f, b)) for f in foo for b in foo], axis=0)
    names = []
    for pair in bar:
        # Get the single kernels
        if "None" in pair:
            if pair[0] == pair[1]:
                continue
            names.append(list(pair["None" != pair]))
        # Remove noise and EQ products as already available in the single
        elif "white" in pair:
            continue
        elif sum("EQ" == pair) == 2:
            continue
        else:
            names.append(list(pair))
    # Simpson et al. get 34, I get 36. TODO: Figure out why
    return names


def generate_kernel(pair):
    kernel = scale_kernel(np.prod([sample_basic_kernel(name) for name in pair]))
    return kernel


def sample_kernel():
    names = get_names()
    pair = names[np.random.choice(len(names))]
    return generate_kernel(pair)
