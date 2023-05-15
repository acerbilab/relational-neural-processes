import click
import cma
import lab as B
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.distributions as thd
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

from .config import PARAM, config
from .utils import get_target, get_model


def baseline_get_candidate(train_X, train_Y, bound, curr_min):
    dim_x = train_X.shape[1]
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acq = ExpectedImprovement(gp, best_f=curr_min)

    bounds = th.stack([bound[0] * th.zeros(dim_x), bound[1] * th.ones(dim_x)])
    # TODO: get the same for our Setup
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
    optima[0] = target(train_Y).max()
    for step in tqdm(range(1, n_steps)):
        candidate = baseline_get_candidate(train_X, train_Y, bound, optima[step])
        train_X = th.cat((train_X, candidate))
        train_Y = th.cat((train_Y, target(candidate)[None]))
        if train_Y[-1, 0] > optima[step - 1]:
            optima[step] = train_Y[-1, 0]
        else:
            optima[step] = optima[step - 1]
    return optima


def rnp_optim(model, init_x, init_y, n_steps, target):
    optima = th.zeros(n_steps)

    train_X = th.clone(init_x)  # .type(th.double)
    train_Y = th.clone(init_y)  # .type(th.double)
    optima[0] = target(train_Y).max()
    for step in tqdm(range(1, n_steps)):
        tqdm.write(f"## {step}/{n_steps}")
        candidate = rnp_get_candidate(model, train_X, train_Y)

        train_X = th.cat((train_X, candidate))
        train_Y = th.cat((train_Y, target(candidate)[None]))
        if train_Y[-1, 0] > optima[step - 1]:
            optima[step] = train_Y[-1, 0]
        else:
            optima[step] = optima[step - 1]
    return optima


def compute_EI(mu, var, f_best):
    p = thd.Normal(0, 1)
    Z = (mu - f_best) / var.sqrt()

    return var.sqrt() * (Z * p.cdf(Z) + p.log_prob(Z).exp())


@th.no_grad()
def EI_objective(x, model, train_X, train_Y, f_best):
    x = th.from_numpy(x).float()
    if not all(th.logical_and(0 < x, x < 1)):
        return 0
    pred = model(train_X.T[None], train_Y.T[None], x[None, :, None])
    return compute_EI(pred.mean, pred.var, f_best).item()


def rnp_get_candidate(model, train_X, train_Y):
    opt_func = lambda x: -EI_objective(x, model, train_X, train_Y, train_Y.max())

    # Get candidates and yes, this does not need to be a loop
    # TODO: change that
    Xcan = th.rand((100, 3)).numpy()
    res = [opt_func(x) for x in Xcan]

    Xstart = Xcan[np.argmin(res)]

    # xopt, es = cma.fmin2(opt_func, train_X[train_Y.argmax()], 0.5)
    xopt, es = cma.fmin2(opt_func, Xstart, 0.5)
    return th.from_numpy(xopt[None]).float()


def rnp_get_candidate_rand(model, train_X, train_Y):
    x_range = (th.rand((50, train_X.shape[1])).T)[None]
    with th.no_grad():
        pred = model(train_X, train_Y, x_range)
        EI = compute_EI(pred.mean.squeeze(), pred.var.squeeze(), train_Y.max())
        candidate = x_range[:, :, th.argmax(EI)]
    return candidate


def visualize(tar_min, methods, name):
    plt.axhline(tar_min, color="black", ls="dashed", alpha=0.8)
    for i, method in enumerate(methods):
        for mm in method[0]:
            plt.plot(-mm, color=sns.color_palette()[i], alpha=0.3)
        plt.plot(
            -th.stack(method[0]).mean(0),
            color=sns.color_palette()[i],
            label=method[1],
        )

        plt.legend()
        sns.despine()
        plt.title(f"{name}")
        plt.savefig(f"{name}.png")
        plt.close()


@click.command()
@click.option("--vis", is_flag=True)
@click.option("--n_rep", default=1)
@click.option("--verbose", is_flag=True)
def main(vis, n_rep):
    target, (dim_x, (min_x, max_x)), tar_min = get_target("hartmann3d")

    B.epsilon = config["epsilon"]

    model = get_model(PARAM.model_name, "cpu")

    model.load_state_dict(
        th.load(f"{PARAM.load_name}.pt", map_location=th.device("cpu"))
    )

    tot_gp = []
    tot_rnp = []
    for i in range(n_rep):
        # Assuming mon_x <= 0 < max_x
        train_X = th.rand((5, dim_x)) * (np.abs(min_x) + max_x) + min_x
        optim_gp = baseline_train(
            train_X, target(train_X)[:, None], PARAM.n_steps, (min_x, max_x), target
        )
        tot_gp.append(optim_gp)
        optim_rnp = rnp_optim(
            model,
            train_X,
            target(train_X)[:, None],
            PARAM.n_steps,
            (min_x, max_x),
            target,
        )
        tot_rnp.append(optim_rnp)
        if verbose:
            print(optim_gp)
            print(optim_rnp)

    if vis:
        visualize(tar_min, [(tot_gp, "GP"), (tot_rnp, "RNP")])


if __name__ == "__main__":
    main()
