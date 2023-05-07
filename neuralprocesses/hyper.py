import gc
import os
from functools import partial

import experiment as exp
import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import numpy as np
import seaborn as sns
import torch
import torch as th
import torch.distributions as thd
from experiment import plot
from stheno.torch import GP, Matern32

# from tqdm.notebook import tqdm
from tqdm import tqdm
from varz import Vars, minimise_adam
from wbml.plot import tweak

# model_name = "rnp"
# model_name = "gnp"
model_name = "agnp"
print(f"Training: {model_name}")

if not os.path.exists("HYPER"):
    print("Created Directory")
    os.mkdir("HYPER")


save_name = f"HYPER/hyper_{model_name}"
grace_period = 10


def distplot(data, label=None, set_theme=True):
    if set_theme:
        sns.set_theme()

    sns.histplot(data, kde=True, stat="density", label=label)
    sns.rugplot(data)


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
    "epochs": 100,
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

        return state, vals.mean(), B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


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


def estimate_lengths(X, y_samples, filename="hist_length.png"):
    X = X.to(device)
    y_samples = y_samples.to(device)
    lengthscales = []

    for y in y_samples:
        _, length = lengthscale_opt(X, y)
        lengthscales.append(length)

    print(len(lengthscales))
    lengthscales = th.stack(lengthscales).cpu()
    distplot(lengthscales)
    plt.xlim(0, 3)
    plt.savefig(filename)
    plt.close()

    return lengthscales


def true_function():
    kernel = Matern32().stretch(2 / 3)
    X = th.linspace(-1, 1, 200).to(device)
    K = kernel(X).mat
    y = thd.MultivariateNormal(th.zeros(200).to(device), K).sample()
    return X, y, kernel


def lengthscale_opt(x_obs, y_obs):
    # Increase regularisation because PyTorch defaults to 32-bit floats.
    B.epsilon = 1e-6

    # # Define points to predict at.
    # x = th.linspace(0, 2, 100)
    # x_obs = th.linspace(0, 2, 50)

    # # Sample a true, underlying function and observations with observation noise `0.05`.
    # f_true = torch.sin(5 * x)
    # y_obs = torch.sin(5 * x_obs) + 0.05**0.5 * torch.randn(50)

    def model(vs):
        """Construct a model with learnable parameters."""
        p = vs.struct  # Varz handles positivity (and other) constraints.
        kernel = Matern32().stretch(p.scale.positive())
        # kernel = stheno.EQ().stretch(p.scale.positive())
        return GP(kernel), p.noise.positive()

    vs = Vars(torch.float32)
    f, noise = model(vs)
    # vs = vs.cpu()
    # f = f.cpu()
    # noise = noise.cpu()
    #
    # print(x_obs)
    # print(y_obs)

    # Condition on observations and make predictions before optimisation.
    f_post = f | (f(x_obs, noise), y_obs)
    prior_before = f, noise
    pred_before = f_post(x_obs, noise).marginal_credible_bounds()

    def objective(vs):
        f, noise = model(vs)
        evidence = f(x_obs, noise + 1e-8).logpdf(y_obs)
        return -evidence

    # print(vs)
    # print(objective)
    # Learn hyperparameters.
    # minimise_l_bfgs_b(objective, vs)
    minimise_adam(objective, vs)

    f, noise = model(vs)

    # Condition on observations and make predictions after optimisation.
    f_post = f | (f(x_obs, noise), y_obs)
    prior_after = f, noise
    pred_after = f_post(x_obs, noise).marginal_credible_bounds()

    def plot_prediction(prior, pred):
        f, noise = prior
        mean, lower, upper = pred
        plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
        # plt.plot(x, f_true, label="True", style="test")
        plt.plot(x_obs, mean, label="Prediction", style="pred")
        plt.fill_between(x_obs, lower, upper, style="pred")
        plt.ylim(-2, 2)
        # plt.text(
        #     0.02,
        #     0.02,
        #     f"scale = {f.kernel.stretches[]:.2f}, " f"noise = {noise:.2f}",
        # transform=plt.gca().transAxes,
        # )
        tweak()

    # if True:
    #     # Plot result.
    #     plt.figure(figsize=(10, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.title("Before optimisation")
    #     plot_prediction(prior_before, pred_before)
    #     plt.subplot(1, 2, 2)
    #     plt.title("After optimisation")
    #     plot_prediction(prior_after, pred_after)
    #     plt.savefig("readme_example12_optimisation_varz.png")
    #     # plt.close()
    #     plt.show()

    return f, f.kernel.stretches[0]


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
model = get_model(model_name, device)

# Setup training loop
opt = torch.optim.Adam(model.parameters(), args.rate)

# Set regularisation high for the first epochs
original_epsilon = B.epsilon
B.epsilon = config["epsilon_start"]
log_loss = th.zeros(args.epochs)
log_eval = th.zeros(args.epochs)
min_val = -th.inf
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
    state, val, _ = eval(state, model, objective_cv, gen_cv())
    log_eval[i] = val.item()
    if log_eval[i] > min_val:
        min_val = log_eval[i]
        th.save(model.state_dict(), f"{save_name}_{model_name}.pt")
    if i > grace_period and all(val.item() < log_eval[(i - grace_period) : i]):
        model.load_state_dict(th.load(f"{save_name}_{model_name}.pt"))
        break
th.save(model.state_dict(), f"{save_name}_{model_name}.pt")
torch.cuda.empty_cache()
gc.collect()

print("Trained successfully")
plt.plot(log_loss)
plt.plot(log_eval)
plt.savefig(f"{save_name}_log_loss.png")
plt.close()
plot.visualise(model, gen_train, path=f"{save_name}_hyper.png", config=config)

Xtrue, ytrue, kernel = true_function()
Xtrue = Xtrue.to(device)
ytrue = ytrue.to(device)

n_test = 200
x_range = th.linspace(-2, 2, n_test).to(device)
batch = nps.batch_index(gen_train.generate_batch(), slice(0, 1, None))

N_context = 40

ids = np.random.choice(200, N_context, replace=False)
contexts = [(Xtrue[ids][None, None], ytrue[ids][None, None])]

with th.no_grad():
    mean, var, samples, _ = nps.predict(
        model,
        # batch["contexts"],
        contexts,
        nps.AggregateInput(
            *((x_range[None, None, :], i) for i in range(config["dim_y"]))
        ),
    )

samples = samples[0].squeeze().data

for s in samples:
    plt.plot(x_range, s, color="darkblue", alpha=0.1)
plt.plot(x_range, mean[0].squeeze())
A, B = contexts[0][0].flatten(), contexts[0][1].flatten()
plt.scatter(A, B)
plt.plot()
plt.savefig(f"{save_name}_samples.png")
plt.close()

lengths = estimate_lengths(x_range, samples, filename=f"{save_name}_hist.png")
