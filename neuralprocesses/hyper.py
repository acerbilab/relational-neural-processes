import gc
import os
from functools import partial

import experiment as exp
import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import torch
import torch as th
from experiment import plot


import seaborn as sns


def distplot(data, label=None, set_theme=True):
    if set_theme:
        sns.set_theme()

    sns.histplot(data, kde=True, stat="density", label=label)
    sns.rugplot(data)


# from tqdm.notebook import tqdm
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device("cuda") if USE_CUDA else torch.device("cpu")

state = B.create_random_state(torch.float32, seed=0)
B.set_global_device(device)

# TODO: For now these are taken from Bruinsma's experiments without adaptation
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
    "fullconvgnp_kernel_factor": 2,
    "mean_diff": 0,
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "dim_x": 1,
}

args = {
    "dim_x": 1,
    "dim_y": 1,
    "data": "eq",
    "batch_size": 16,
    "epochs": 20,
    "rate": 3e-4,
    "objective": "loglik",
    "num_samples": 20,
    "unnormalised": False,
    "evaluate_num_samples": 512,
    "evaluate_batch_size": 8,
    "train_fast": True,
    "evaluate_fast": True,
}


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
objectives_eval = [
    (
        "Loglik",
        partial(
            nps.loglik,
            num_samples=args.evaluate_num_samples,
            batch_size=args.evaluate_batch_size,
            normalise=not args.unnormalised,
        ),
    )
]


def train(state, model, opt, objective, gen, *, fix_noise):
    """Train for an epoch."""
    vals = []
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            fix_noise=fix_noise,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    vals = B.concat(*vals)
    # out.kv("Loglik (T)", exp.with_err(vals, and_lower=True))
    return state, vals.mean(), B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        vals = B.concat(*vals)
        # out.kv("Loglik (VV)", exp.with_err(vals, and_lower=True))
        # if kls:
        #     out.kv("KL (full)", exp.with_err(B.concat(*kls), and_upper=True))
        # if kls_diag:
        #     out.kv("KL (diag)", exp.with_err(B.concat(*kls_diag), and_upper=True))

        # objective doesn't return pred_y, we can't plot the data

        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


B.epsilon = config["epsilon"]

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
model = model.to(device)

# Setup training loop
opt = torch.optim.Adam(model.parameters(), args.rate)

# Set regularisation high for the first epochs
original_epsilon = B.epsilon
B.epsilon = config["epsilon_start"]
log_loss = th.zeros(args.epochs)
for i in tqdm(range(0, args.epochs)):
    # Set regularisation to normal after the first epoch
    if i > 0:
        B.epsilon = original_epsilon

    # Perform an epoch.
    if config["fix_noise"] and i < config["fix_noise_epochs"]:
        fix_noise = 1e-4
    else:
        fix_noise = None
    state, vals, _ = train(
        state,
        model,
        opt,
        objective,
        gen_train,
        fix_noise=fix_noise,
    )
    log_loss[i] = vals.item()

    # The epoch is done. Now evaluate.
    state, val = eval(state, model, objective_cv, gen_cv())
    torch.cuda.empty_cache()
    gc.collect()

print("Trained successfully")
plt.plot(log_loss)
plt.savefig("log_loss.png")
plt.close()
plot.visualise(model, gen_train, path="hyper.png", config=config)

# Hyperparameter learning
import gpytorch as gpt
import GPy as gpy


def estimate_lengths(X, y_samples, filename="hist_length.png"):
    X = X.numpy()
    y_samples = y_samples.numpy()
    lengthscales = []

    for y in y_samples:
        kern = gpy.kern.Matern32(1)
        gp = gpy.models.GPRegression(X, y[:, None], kern)
        gp.kern.variance.fix()
        gp.likelihood.fix()
        gp.optimize()
        lengthscales.append(gp.kern.lengthscale[0])

    distplot(lengthscales)
    plt.xlim(0, 3)
    plt.savefig(filename)
    plt.close()

    return lengthscales


n_test = 200
x_range = th.linspace(-1, 1, n_test)
batch = nps.batch_index(gen_train.generate_batch(), slice(0, 1, None))

with th.no_grad():
    mean, var, samples, _ = nps.predict(
        model,
        batch["contexts"],
        nps.AggregateInput(
            *((x_range[None, None, :], i) for i in range(config["dim_y"]))
        ),
    )

samples = samples[0].squeeze()
for s in samples:
    plt.plot(s, color="darkblue", alpha=0.1)
plt.savefig("samples.png")
plt.close()

lengths = estimate_lengths(x_range[:, None], samples)
