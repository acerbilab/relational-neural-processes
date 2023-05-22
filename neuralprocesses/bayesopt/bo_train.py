import os

import click
import lab as B
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

# TODO: Restructure this properly
from config import get_args, get_config
from utils import get_model, train_epoch, eval, get_generators, get_objectives

if not os.path.exists("../BO"):
    print("Create save dir")
    os.mkdir("../BO")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train(
    model,
    state,
    objective,
    objective_cv,
    gen_train,
    gen_cv,
    args,
    config,
    device,
    save_name,
    verbose=False,
):
    # Setup training loop
    opt = th.optim.Adam(model.parameters(), args.rate)

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
        state, vals, _ = train_epoch(
            state,
            model,
            opt,
            objective,
            gen_train,
            device=device,
            fix_noise=fix_noise,
        )
        log_loss[i] = vals.item()

        # The epoch is done. Now evaluate.
        state, val, _ = eval(state, model, objective_cv, gen_cv(), device)
        log_eval[i] = val.item()
        if log_eval[i] > min_val:
            min_val = log_eval[i]
            th.save(model.state_dict(), f"{save_name}.pt")
        if i > args.grace_period and all(
            val.item() < log_eval[(i - args.grace_period) : i]
        ):
            model.load_state_dict(th.load(f"{save_name}.pt"))
            break
        th.cuda.empty_cache()
        gc.collect()
        if (i % 10 == 0) and verbose:
            plt.plot(log_loss)
            plt.plot(log_eval)
            plt.title(f"trained for {i}/{args.epochs}")
            plt.savefig(f"{save_name}_log_loss.png")
            plt.close()
    th.save(model.state_dict(), f"{save_name}.pt")
    if verbose:
        print("Trained successfully")
        plt.plot(log_loss)
        plt.plot(log_eval)
        plt.title(f"trained for {i}/{args.epochs}")
        plt.savefig(f"{save_name}_log_loss.png")
        plt.close()
    return log_loss, log_eval


@click.command()
@click.option("--run", default=1)
@click.option("--target", default="hartmann3d")
@click.option("--model", default="rcnp")
@click.option("--exp", default="bo_fixed")
@click.option("--save_postfix", default="")
def main(run, target, model, exp, save_postfix):
    dim_x, bound = {
        "hartmann3d": (3, (0, 1)),
        "rastrigin": (4, (-1, 1)),
        "ackley": (5, (-32.768, 32.768)),
        "hartmann6d": (6, (0, 1)),
    }[target]

    args = get_args(dim_x, exp)
    config = get_config(dim_x, bound)

    # args["epochs"] = 1
    # args["train_fast"] = True
    args["seed"] = int(run)
    # assert False

    save_name = f"BO/{exp}_{target}_{model}_{run}"
    if save_postfix != "":
        save_name += f"_{save_postfix}"
    print(f"Training: {save_name}")

    USE_CUDA = th.cuda.is_available()
    device = th.device("cuda") if USE_CUDA else th.device("cpu")

    num_seeds = 3
    repnum = args.seed
    seeds = np.random.SeedSequence(
        entropy=11463518354837724398231700962801993226
    ).generate_state(num_seeds * repnum)[-num_seeds:]
    state = B.create_random_state(th.float32, seed=int(seeds[0]))
    B.set_random_seed(int(seeds[0]))

    B.set_global_device(device)
    B.epsilon = config["epsilon"]

    model = get_model(model, config, args, device)

    objective, objective_cv = get_objectives(args=args)

    gen_train, gen_cv, gens_eval = get_generators(args=args, config=config, seeds=seeds)

    train(
        model,
        state,
        objective,
        objective_cv,
        gen_train,
        gen_cv,
        args,
        config,
        device,
        save_name=save_name,
        verbose=True,
    )


if __name__ == "__main__":
    main()
