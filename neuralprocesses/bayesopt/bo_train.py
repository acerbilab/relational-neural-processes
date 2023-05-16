import gc
import os
import lab as B

import click
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm

# TODO: Restructure this properly instead of the current mess
from utils import get_model, train_epoch, eval
from config import get_generators, get_objectives, PARAM, args, config

if not os.path.exists("../BO"):
    print("Create save dir")
    os.mkdir("../BO")


def train(
    model,
    state,
    objective,
    objective_cv,
    gen_train,
    gen_cv,
    param,
    device,
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
            th.save(model.state_dict(), f"{param.save_name}.pt")
        if i > param.grace_period and all(
            val.item() < log_eval[(i - param.grace_period) : i]
        ):
            model.load_state_dict(th.load(f"{param.save_name}.pt"))
            break
        th.cuda.empty_cache()
        gc.collect()
    th.save(model.state_dict(), f"{param.save_name}.pt")
    if verbose:
        print("Trained successfully")
        plt.plot(log_loss)
        plt.plot(log_eval)
        plt.savefig(f"{param.save_name}_log_loss.png")
        plt.close()
    return log_loss, log_eval


@click.command()
@click.option("--run", default=1)
@click.option("--target", default="hartmann3d")
@click.option("--model", default="rcnp")
@click.option("--save_postfix", default="")
def main(run, target, model, save_postfix):
    param = PARAM
    param.save_name = f"{target}_{param.save_name}_{model}_{save_postfix}"
    print(f"Training: {param.save_name}")
    param.target_name = target
    param.model_name = model
    # TODO: Do I need this?
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    USE_CUDA = th.cuda.is_available()
    device = th.device("cuda") if USE_CUDA else th.device("cpu")

    args["seed"] = run

    B.set_global_device(device)
    state = B.create_random_state(th.float32, seed=0)
    B.epsilon = config["epsilon"]

    model = get_model(param.model_name, args, device)

    objective, objective_cv = get_objectives()

    gen_train, gen_cv, gens_eval = get_generators(args.data)

    train(
        model,
        state,
        objective,
        objective_cv,
        gen_train,
        gen_cv,
        param,
        device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
