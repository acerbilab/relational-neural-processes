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
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

from config import PARAM, config, args
from utils import get_target, get_model


def baseline_get_candidate(train_X, train_Y, bound, curr_min):
    dim_x = train_X.shape[1]
    gp = SingleTaskGP(train_X, standardize(train_Y))
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


def baseline_get_candidate_cma(train_X, train_Y, bound=None):
    assert bound is None, "Don't allow non-cube bounds for now"
    # maximize the negative target
    Y = -standardize(train_Y)
    gp = SingleTaskGP(train_X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    # EI = ExpectedImprovement(gp, best_f=Y.min())
    EI = ExpectedImprovement(gp, best_f=Y.max())

    x0 = np.random.rand(train_X.shape[1])
    x0tmp = np.random.rand(100, train_X.shape[1])
    ymp = EI(th.tensor(x0tmp, dtype=train_X.dtype).unsqueeze(-2))
    x0 = x0tmp[ymp.argmax()]

    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=0.2,
        inopts={"bounds": [0, 1], "popsize": 50},
    )

    X = train_X

    while not es.stop():
        xs = es.ask()
        X = th.tensor(xs, device=X.device, dtype=X.dtype)

        # cma again expects us to minimize
        with th.no_grad():
            Y = -EI(X.unsqueeze(-2))

        y = Y.view(-1).double().numpy()
        es.tell(xs, y)

    return th.from_numpy(es.best.x).to(X).unsqueeze(0)


def rnp_get_candidate_cma_tmp(model, train_X, train_Y):
    opt_func = lambda x: EI_objective(x, model, train_X, train_Y, -train_Y.min())

    # x0 = np.random.rand(train_X.shape[1])

    x0tmp = np.random.rand(100, train_X.shape[1])
    ytmp = -opt_func(th.tensor(x0tmp, device=train_X.device, dtype=train_X.dtype))
    x0 = x0tmp[ytmp.argmin()]

    # print("foo")
    # print(ytmp.min())
    # print(train_Y.min())
    # print("bar")

    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=0.2,
        inopts={"bounds": [0, 1], "popsize": 50},
    )

    X = train_X

    while not es.stop():
        xs = es.ask()
        X = th.tensor(xs, device=X.device, dtype=X.dtype)

        # cma again expects us to minimize
        with th.no_grad():
            Y = -opt_func(X)

        y = Y.view(-1).double().numpy()
        es.tell(xs, y)

    return th.from_numpy(es.best.x).to(X).unsqueeze(0)


def rnp_get_candidate_cma(model, train_X, train_Y):
    # Y = -standardize(train_Y)
    # print(train_Y.shape)
    # print(Y.shape)
    train_Y = -standardize(train_Y)
    # print(train_Y.shape)
    # print(Y.shape)
    opt_func = lambda x: EI_objective(x, model, train_X, train_Y, train_Y.max())

    # x0 = np.random.rand(2, train_X.shape[1])
    x0 = np.random.rand(train_X.shape[1])
    # print(x0.shape)

    x0tmp = np.random.rand(100, train_X.shape[1])
    ytmp = opt_func(th.tensor(x0tmp, device=train_X.device, dtype=train_X.dtype))
    x0 = x0tmp[ytmp.argmax()]

    # print("foo")
    # print(ytmp.min())
    # print(train_Y.min())
    # print("bar")

    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=0.2,
        inopts={"bounds": [0, 1], "popsize": 50},
    )

    X = train_X

    # print(opt_func(X).max())
    while not es.stop():
        xs = es.ask()
        X = th.tensor(xs, device=X.device, dtype=X.dtype)

        # cma again expects us to minimize
        with th.no_grad():
            Y = -opt_func(X)

        y = Y.view(-1).double().numpy()
        es.tell(xs, y)

    return th.from_numpy(es.best.x).to(X).unsqueeze(0)


@th.no_grad()
def EI_objective(x, model, train_X, train_Y, f_best):
    #  x = th.from_numpy(x).float()
    # print("foo")
    # print(x.shape)
    # print(train_X.T[None].shape)
    # print(train_Y.T[None].shape)
    dimx = x.shape[1]

    pred = model(train_X.T[None], train_Y.T[None], x.T[None])

    res = compute_EI(pred.mean, pred.var, f_best).data.flatten()
    bound = th.logical_and(0 < x, x < 1).sum(-1)
    res[bound < dimx] = 0.0
    # print(res)
    return res


def rnp_get_candidate_cma_old(model, train_X, train_Y):
    opt_func = lambda x: EI_objective(x, model, train_X, train_Y, train_Y.max())

    # Get candidates and yes, this does not need to be a loop
    # TODO: change that
    Xcan = th.rand((100, 3)).numpy()
    res = [opt_func(x) for x in Xcan]

    Xstart = Xcan[np.argmin(res)]

    # xopt, es = cma.fmin2(opt_func, train_X[train_Y.argmax()], 0.5)
    xopt, es = cma.fmin2(opt_func, Xstart, 0.5)
    return th.from_numpy(xopt[None]).float()


def baseline_optim(init_x, init_y, n_steps, bound, target):
    optima = th.zeros(n_steps)

    train_X = th.clone(init_x).type(th.double)
    train_Y = th.clone(init_y).type(th.double)
    # Want to minimize the target
    # optima[0] = target(train_Y).min()
    optima[0] = train_Y.min()
    for step in tqdm(range(1, n_steps)):
        # candidate = baseline_get_candidate(train_X, train_Y, bound, optima[step])
        candidate = baseline_get_candidate_cma(train_X, train_Y)
        # print(target(candidate))
        train_X = th.cat((train_X, candidate))
        train_Y = th.cat((train_Y, target(candidate)[None]))
        if train_Y[-1, 0] < optima[step - 1]:
            optima[step] = train_Y[-1, 0]
        else:
            optima[step] = optima[step - 1]
    return optima


def rnp_optim(model, init_x, init_y, n_steps, target):
    optima = th.zeros(n_steps)

    train_X = th.clone(init_x)  # .type(th.double)
    train_Y = th.clone(init_y)  # .type(th.double)
    # optima[0] = target(train_Y).min()
    optima[0] = train_Y.min()
    # assert False
    for step in tqdm(range(1, n_steps)):
        tqdm.write(f"## {step}/{n_steps}")
        # print(f"Y: {train_Y.min():.4f}")
        # print(f"O: {optima.min():.4f}")
        # candidate = rnp_get_candidate_cma(model, train_X, train_Y)
        candidate = rnp_get_candidate_cma(model, train_X, train_Y)
        # candidate = rnp_get_candidate_rand(model, train_X, train_Y)

        train_X = th.cat((train_X, candidate))
        train_Y = th.cat((train_Y, target(candidate)[None]))
        # print(f"T: {target(candidate).item():.4f}")
        if train_Y[-1, 0] < optima[step - 1]:
            optima[step] = train_Y[-1, 0]
        else:
            optima[step] = optima[step - 1]
    # print("#####")
    # print(train_Y.flatten())
    # print(optima)
    # assert False
    return optima


def compute_EI(mu, var, f_best):
    p = thd.Normal(0, 1)
    Z = (mu - f_best) / var.sqrt()

    return var.sqrt() * (Z * p.cdf(Z) + p.log_prob(Z).exp())


def rnp_get_candidate_rand(model, train_X, train_Y):
    x_range = th.rand((50, train_X.shape[1])).T[None]
    with th.no_grad():
        # print(train_X.T[None].shape)
        # print(train_Y.T[None].shape)
        # print(x_range[None].shape)
        pred = model(train_X.T[None], train_Y.T[None], x_range)
        EI = compute_EI(pred.mean.squeeze(), pred.var.squeeze(), train_Y.max())
        candidate = x_range[:, :, th.argmax(EI)]
    return candidate


def visualize(tar_min, methods, name="tmp"):
    plt.axhline(tar_min, color="black", ls="dashed", alpha=0.8)
    for i, method in enumerate(methods):
        for mm in method[0]:
            plt.plot(mm, color=sns.color_palette()[i], alpha=0.3)
        plt.plot(
            th.stack(method[0]).mean(0),
            color=sns.color_palette()[i],
            label=method[1],
        )

        plt.legend()
        sns.despine()
        plt.title(f"{name}")
    plt.show()
    # plt.savefig(f"{name}.png")
    # plt.close()


@click.command()
@click.option("--vis", is_flag=True)
@click.option("--n_rep", default=1)
@click.option("--n_steps", default=50)
@click.option("--model_name", default="rcnp")
@click.option("--verbose", is_flag=True)
def main(vis, n_rep, n_steps, model_name, verbose):
    target_name = "hartmann3d"
    target, (dim_x, (min_x, max_x)), tar_min = get_target(target_name)

    B.epsilon = config["epsilon"]

    model = get_model(model_name, args, "cpu")
    param = PARAM
    # param.load_name = f"{param.load_name}single_kernel/{target_name}_{model_name}3"
    param.load_name = f"{param.load_name}{target_name}_{model_name}_matern300_2"

    model.load_state_dict(
        th.load(f"{PARAM.load_name}.pt", map_location=th.device("cpu"))
    )

    tot_gp = []
    tot_rnp = []
    for i in range(n_rep):
        # Assuming min_x <= 0 < max_x
        train_X = th.rand((5, dim_x))  # * (np.abs(min_x) + max_x) + min_x

        optim_gp = baseline_optim(
            train_X, target(train_X)[:, None], n_steps, (min_x, max_x), target
        )
        # print(optim_gp)
        tot_gp.append(optim_gp)
        optim_rnp = rnp_optim(
            model,
            train_X,
            target(train_X)[:, None],
            n_steps,
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
