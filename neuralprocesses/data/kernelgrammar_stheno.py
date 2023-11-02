######
# A Stheno version of `kernelgrammar_gpytorch.py`

import numpy as np
import torch as th
import stheno.torch as st


def scale_kernel(kernel):
    outputscale = np.exp(np.random.randn())
    return outputscale * kernel


def sample_length():
    return np.exp(np.random.randn())


def generate_rbf():
    return st.EQ().stretch(sample_length())


def generate_matern(nu):
    assert nu in [1 / 2, 3 / 2, 5 / 2]
    if nu == 1 / 2:
        kernel = st.Matern12()
    elif nu == 3 / 2:
        kernel = st.Matern32()
    elif nu == 5 / 2:
        kernel = st.Matern52()
    return kernel.stretch(sample_length())


def sample_basic_kernel(name=None, scale=False):
    if name == "EQ":
        kernel = generate_rbf()
    elif name == "matern12":
        kernel = generate_matern(1 / 2)
    elif name == "matern32":
        kernel = generate_matern(3 / 2)
    elif name == "matern52":
        kernel = generate_matern(5 / 2)
    else:
        raise NotImplementedError(f"{name} does not have a kernel implementation")

    if scale and name != "white":
        kernel = scale_kernel(kernel)

    return kernel


def get_names(single=False):
    "Not the nicest formulation, but it works for now"
    basic = [
        "None",
        "EQ",
        "matern12",
        "matern32",
        "matern52",
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
        elif sum("EQ" == pair) == 2:
            continue
        else:
            names.append(list(pair))

    return names


def generate_kernel(names, single):
    if single:
        return scale_kernel(sample_basic_kernel(names))
    else:
        kernel = scale_kernel(np.prod([sample_basic_kernel(name) for name in names]))
    return kernel


def sample_kernel(single=False):
    if single:
        names = get_names(True)
        names = names[np.random.choice(len(names))]
        return generate_kernel(names, True)

    sum = np.random.randn() > 0
    if sum:
        product = np.random.randn() > 0
        names = get_names(product)
        names = names[np.random.choice(len(names))]
        kernel1 = generate_kernel(names, product)
        product = np.random.randn() > 0
        names = get_names(product)
        names = names[np.random.choice(len(names))]
        kernel2 = generate_kernel(names, product)

        return kernel1 + kernel2
    else:
        product = np.random.randn() > 0
        names = get_names(product)
        names = names[np.random.choice(len(names))]
        return generate_kernel(names, product)
