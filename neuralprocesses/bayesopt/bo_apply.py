from functools import partial

import botorch as bth
import cma
import experiment as exp
import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import numpy as np
import torch as th
import torch.distributions as thd
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm


# See https://www.sfu.ca/~ssurjano/optimization.html for target details


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


target, (dim, (min_x, max_x)), tar_min = get_target("hartmann3d")


# # Assuming mon_x <= 0 < max_x
# train_X = th.rand((1, dim)) * (np.abs(min_x) + max_x) + min_x
# optim_gp = baseline_train(
#     train_X, target(train_X)[:, None], n_steps, (min_x, max_x), target
# )
#
# plt.axhline(tar_min, color="black", ls="dashed", alpha=0.8)
# plt.plot(-optim_gp)
# plt.show()


n_steps = 20
device = "cpu"
model_name = "rnp"
# # model_name = "gnp"
# model_name = "agnp"
target_name = "hartmann3d"
# target_name = "rastrigin"
# target_name = "ackley"
# target_name = "hartmann6d"
save_name = None
if save_name is None:
    local_path = "../../../results/"
    local_path = ""
    save_name = f"{local_path}BO/{target_name}_{model_name}"
    # save_name = f"{local_path}BO/{target_name}_{model_name}_product"
# save_name = "BO/hartmann6d_rnp"


data_set_dims = {
    "hartmann3d": (3, (0, 1)),
    "rastrigin": (4, (-1, 1)),
    "ackley": (5, (-32.768, 32.768)),
    "hartmann6d": (6, (0, 1)),
}

dim_x, context = data_set_dims[target_name]


# TODO: For now these are taken from Bruinsma's experiments essentially without adaptation
config = {
    "default": {
        "epochs": None,
        "rate": None,
        "also_ar": False,
    },
    "epsilon": 1e-8,
    "epsilon_start": 1e-2,
    "cholesky_retry_factor": 1e6,
    "fix_noise": None,
    "fix_noise_epochs": 3,
    "width": 256,
    "dim_embedding": 256,
    "relational_width": 64,
    "dim_relational_embeddings": 128,
    "enc_same": False,
    "num_heads": 8,
    "num_layers": 6,
    "num_relational_layers": 3,
    "unet_channels": (64,) * 6,
    "unet_strides": (1,) + (2,) * 5,
    "conv_channels": 64,
    "encoder_scales": None,
    "x_range_context": context,
    "fullconvgnp_kernel_factor": 2,
    "mean_diff": 0,
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "dim_x": dim_x,
    "dim_y": 1,
}

args = {
    "dim_x": dim_x,
    "dim_y": 1,
    "data": "eq",
    "batch_size": 32,
    "epochs": 200,
    "rate": 3e-4,
    "objective": "loglik",
    "num_samples": 20,
    "unnormalised": False,
    "evaluate_num_samples": 512,
    "evaluate_batch_size": 8,
    "train_fast": False,
    "evaluate_fast": True,
}

B.epsilon = config["epsilon"]


class mydict(dict):
    def __getattribute__(self, key):
        if key in self:
            return self[key]
        else:
            return super().__getattribute__(key)


args = mydict(args)

gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
    args,
    config,
    num_tasks_train=2**6 if args.train_fast else 2**14,
    # num_tasks_train=2**10 if args.train_fast else 2**14,
    num_tasks_cv=2**6 if args.train_fast else 2**12,
    num_tasks_eval=2**6 if args.evaluate_fast else 2**12,
    device=device,
)

objective = partial(
    nps.loglik,
    num_samples=args.num_samples,
    normalise=not args.unnormalised,
)
objective_cv = partial(
    nps.loglik,
    num_samples=args.num_samples,
    normalise=not args.unnormalised,
)


def get_model(model_name, device):
    if model_name == "rnp":
        model = nps.construct_rnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            relational_width=config["relational_width"],
            num_relational_enc_layers=config["num_relational_layers"],
            likelihood="lowrank",
            transform=config["transform"],
        )
    elif model_name == "gnp":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    elif model_name == "agnp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    else:
        raise NotImplementedError(f"{model_name} is not implemented")

    model = model.to(device)
    return model


model = get_model(model_name, "cpu")

model.load_state_dict(th.load(f"{save_name}.pt", map_location=th.device("cpu")))


def rnp_optim(model, init_x, init_y, n_steps, bound, target):
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
    return optima  # , train_X, train_Y


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
    Xcan = th.rand((100, 3)).numpy()
    res = [opt_func(x) for x in Xcan]

    Xstart = Xcan[np.argmin(res)]

    # xopt, es = cma.fmin2(opt_func, train_X[train_Y.argmax()], 0.5)
    xopt, es = cma.fmin2(opt_func, Xstart, 0.5)
    return th.from_numpy(xopt[None]).float()


# def rnp_get_candidate(model, train_X, train_Y, bound, f_best):
#     x_range = (th.rand((50, dim)).T)[None]
#     with th.no_grad():
#         pred = model(train_X, train_Y, x_range)
#         EI = compute_EI(pred.mean.squeeze(), pred.var.squeeze(), f_best)
#         # print(EI)
#         candidate = x_range[:, :, th.argmax(EI)]
#     return candidate


tot_baseline = []
tot_rnp = []
for i in range(1):
    # Assuming mon_x <= 0 < max_x
    train_X = th.rand((5, dim)) * (np.abs(min_x) + max_x) + min_x
    optim_gp = baseline_train(
        train_X, target(train_X)[:, None], n_steps, (min_x, max_x), target
    )
    tot_baseline.append(optim_gp)
    optim_rnp = rnp_optim(
        model, train_X, target(train_X)[:, None], n_steps, (min_x, max_x), target
    )
    print(optim_gp)
    print(optim_rnp)
    tot_rnp.append(optim_rnp)

import seaborn as sns

plt.axhline(tar_min, color="black", ls="dashed", alpha=0.8)
for gp in tot_baseline:
    plt.plot(-gp, color=sns.color_palette()[0], alpha=0.3)
plt.plot(-th.stack(tot_baseline).mean(0), color=sns.color_palette()[0], label="gp")
# plt.plot(-optim_gp, label="gp")
for rnp in tot_rnp:
    plt.plot(-rnp, color=sns.color_palette()[1], alpha=0.3)
plt.plot(-th.stack(tot_rnp).mean(0), color=sns.color_palette()[1], label="rnp")

plt.legend()
sns.despine()
plt.title(f"{target_name}-prod")
plt.savefig("tmp.png")
plt.close()
# plt.show()
