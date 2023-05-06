######
# A Stheno version of `kernelgrammar_gpytorch.py`

import numpy as np
import torch as th
import stheno.torch as st


def scale_kernel(kernel):
    outputscale = th.exp(th.randn(1))
    outputscale = np.exp(np.random.randn())
    return outputscale * kernel


def sample_length():
    # return th.exp(th.randn(1))
    return np.exp(np.random.randn())


def generate_rbf():
    return st.EQ().stretch(sample_length())


def generate_periodic():
    raise NotImplementedError("See doc for a function on how to make kernels period")
    kernel = gpy.kernels.PeriodicKernel()
    kernel.lengthscale = th.exp(th.randn(1, 1))
    kernel.period_length = th.exp(th.randn(1, 1))
    return kernel


def generate_white():
    raise NotImplementedError("Implement manually")
    return WhiteNoise()


def generate_matern(nu):
    assert nu in [1 / 2, 3 / 2, 5 / 2]
    if nu == 1 / 2:
        kernel = st.Matern12()
    elif nu == 3 / 2:
        kernel = st.Matern32()
    elif nu == 5 / 2:
        kernel = st.Matern52()
    return kernel.stretch(sample_length())


def generate_cosine():
    raise NotImplementedError
    kernel = gpy.kernels.CosineKernel()
    # TODO: Simpson claims not to enforce a positivity constraint on the lengthscale
    #   I don't get why that should work... and thus just take the absolute for now
    kernel.period_length = thd.Cauchy(0, 5).sample((1, 1)).abs()
    return kernel


# class ModifiedLinear(gpy.kernels.Kernel):
#     # Simpson et al. use K(x,z) = s^2 (x - c)(z - c) while gpy provides K(x,z) = s^2x * z
#     is_stationary = False
#     linear = gpy.kernels.LinearKernel()
#     shift = nn.Parameter(thd.Cauchy(0, 5).sample((1, 1)))
#     shift.requires_grad = False
#
#     def forward(self, x1, x2, **params):
#         # TODO: Do this properly
#         self.shift = self.shift.to(x1.device)
#         self.linear = self.linear.to(x1.device)
#
#         x1_shift = x1 - self.shift
#         x2_shift = x2 - self.shift
#         return self.linear(x1_shift, x2_shift)


# TODO: Not the full linear kernel from Simpson et al.
def generate_linear():
    return st.Linear().stretch(sample_length())


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


def get_names(single=False, product=False):
    "Not the nices formulation, but it works for now"
    basic = [
        "None",
        "EQ",
        # "periodic",
        # "white",  # TODO: Currently gives some dimensionality error in batch mode
        "matern12",
        "matern32",
        "matern52",
        # "cosine",
        # "linear", # TODO: Some device issues with stheano
    ]
    if single:
        # Don't return the None kernel
        return basic[1:]

    raw = np.unique([sorted((f, b)) for f in basic for b in basic], axis=0)
    names = []
    for pair in raw:
        # Get the single kernels
        if "None" in pair:
            if pair[0] == pair[1]:
                continue
            names.append(list(pair["None" != pair]))
        # Remove noise and EQ products as already available in the single
        elif "white" in pair:
            continue
        elif product and (sum("EQ" == pair) == 2):
            continue
        else:
            names.append(list(pair))
    # Simpson et al. get 34, I get 36. TODO: Figure out why

    return names


def generate_kernel(names, single, product):
    if single:
        kernel = scale_kernel(sample_basic_kernel(names))
    else:
        if product:
            kernel = scale_kernel(
                np.prod([sample_basic_kernel(name) for name in names])
            )
        else:
            kernel = np.sum([scale_kernel(sample_basic_kernel(name)) for name in names])
    return kernel


def sample_kernel(single=False):
    product = np.random.randn() > 0
    names = get_names(single, product)
    names = names[np.random.choice(len(names))]
    return generate_kernel(names, single, product)
