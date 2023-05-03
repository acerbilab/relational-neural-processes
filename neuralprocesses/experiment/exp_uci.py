import numpy as np
import torch as th
import gpytorch as gpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import neuralprocesses as nps


# TODO: Fix imports later
from data.uci_loader import load_dataset


train_x, test_x, train_y, test_y = load_dataset(
    "boston", raw=True, path="../../../data"
)


# GPY Baseline with a simple RBF kernel
def train_gp(train_x, train_y, test_x, test_y):
    # Follows the Gpytorch tutorial
    class ExactGPModel(gpy.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpy.means.ConstantMean()
            self.covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    train_y = train_y.flatten()

    # gpy.models.ExactGP()
    likelihood = gpy.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optim = th.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    likelihood.train()
    model.train()

    n_steps = 100
    log_loss = th.zeros(n_steps)
    for i in range(n_steps):
        optim.zero_grad()
        out = model(train_x)
        loss = -mll(out, train_y).mean()
        loss.backward()
        optim.step()
        log_loss[i] = loss.item()

    with th.no_grad():
        likelihood.eval()
        model.eval()
        test_mll = mll(model(test_x), test_y).mean()

    print(f"Test MLL: {test_mll:.4f}")


# Hyper
# batch_size = 16


def generate_batch(train_x, train_y, batch_size=16):
    n_train, dim_x = train_x.shape
    batch = dict()
    N = np.random.randint(10, 50)
    indices_context = [
        np.random.choice(n_train, N, replace=False) for _ in range(batch_size)
    ]

    raw_x = th.stack([train_x[ind].T for ind in indices_context])
    raw_y = th.stack([train_y[ind].T for ind in indices_context])

    # Not sure why he has this nested structure
    batch["contexts"] = [
        (raw_x, raw_y),
    ]

    # TODO: Currently the two can still overlap
    indices_target = [
        np.random.choice(n_train, 50, replace=False) for _ in range(batch_size)
    ]
    raw_x = th.stack([train_x[ind].T for ind in indices_target])
    raw_y = th.stack([train_y[ind].T for ind in indices_target])

    batch["xt"] = raw_x
    batch["yt"] = raw_y

    return batch


def generate_test(train_x, train_y, test_x, test_y):
    pass
