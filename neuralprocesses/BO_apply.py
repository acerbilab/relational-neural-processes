import numpy as np
import torch as th
import botorch as bth
import seaborn as sns
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

# from scipy.stats.qmc import Halton
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf


# See https://www.sfu.ca/~ssurjano/optimization.html for target details

target_name = "hartmann3d"
n_steps = 100


def get_target(target_name):
    data_set_dims = {
        "hartmann3d": (3, (0, 1)),
        "rastrigin": (4, (-1, 1)),
        "ackley": (5, (-32.768, 32.768)),
        "hartmann6d": (6, (0, 1)),
    }

    if target_name == "hartmann3d":
        target = bth.test_functions.Hartmann(3)
        tar_min = -3.86278
    elif target_name == "rastrigin":
        target = bth.test_functions.Rastrigin(4)
        tar_min = 0.0
    elif target_name == "ackley":
        target = bth.test_functions.Ackley(5)
        tar_min = 0.0
    elif target_name == "hartmann6d":
        target = bth.test_functions.Hartmann(3)
        tar_min = -3.32237
    else:
        raise NotImplementedError()

    return lambda x: -target(x), data_set_dims[target_name], tar_min


def baseline_get_candidate(train_X, train_Y, bound, curr_min):
    dim_x = train_X.shape[1]
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acq = ExpectedImprovement(gp, best_f=curr_min)

    bounds = th.stack([bound[0] * th.zeros(dim_x), bound[1] * th.ones(dim_x)])
    candidate, acq_value = optimize_acqf(
        acq,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    return candidate  # tensor([0.4887, 0.5063])


def baseline_train(init_x, init_y, n_steps, bound, target):
    optima = th.zeros(n_steps)

    train_X = th.clone(init_x).type(th.double)
    train_Y = th.clone(init_y).type(th.double)
    optima[0] = target(train_Y)
    for step in tqdm(range(1, n_steps)):
        candidate = baseline_get_candidate(train_X, train_Y, bound, optima[step])
        train_X = th.cat((train_X, candidate))
        train_Y = th.cat((train_Y, target(candidate)[None]))
        if train_Y[-1, 0] > optima[step - 1]:
            optima[step] = train_Y[-1, 0]
        else:
            optima[step] = optima[step - 1]
    return optima


target, (dim, (min_x, max_x)), tar_min = get_target("hartmann3d")


# Assuming mon_x <= 0 < max_x
train_X = th.rand((1, dim)) * (np.abs(min_x) + max_x) + min_x
optim_gp = baseline_train(
    train_X, target(train_X)[:, None], n_steps, (min_x, max_x), target
)

plt.axhline(tar_min, color="black", ls="dashed", alpha=0.8)
plt.plot(-optim_gp)
plt.show()


if False:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.utils import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_X = torch.rand(10, 2)
    Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
    Y = Y + 0.1 * torch.randn_like(Y)  # add some noise

    for i in tqdm(range(50)):
        train_Y = standardize(Y)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        from botorch.acquisition import UpperConfidenceBound

        UCB = UpperConfidenceBound(gp, beta=0.1)

        from botorch.optim import optimize_acqf

        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        candidate  # tensor([0.4887, 0.5063])

        train_X = th.cat((train_X, candidate))
        Y = th.cat((Y, 1 - th.norm(candidate - 0.5, -1, keepdim=True)))
