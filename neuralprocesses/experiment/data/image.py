import os
import torch
from functools import partial

import neuralprocesses.torch as nps
from .util import register_data
from neuralprocesses.dist import UniformDiscrete
from neuralprocesses.comparison_function import difference

__all__ = []


def toroid_difference(relational_encoding_type, xc, yc, xt, L=None, **kwargs):
    "Calculate difference between pixel locations on a torus."

    if relational_encoding_type != 'simple':
        raise NotImplementedError

    if len(kwargs.get('non_equivariant_dim', [])) > 0:
        raise NotImplementedError

    diff_x = difference(relational_encoding_type, xc, yc, xt, **kwargs)
    dim = xt.shape[-1]
    diff_x[:, :, :dim] = (((L-1) * diff_x[:, :, :dim]) % L) / (L-1)
    return diff_x


def setup(
    args,
    config,
    *,
    num_tasks_train,
    num_tasks_cv,
    num_tasks_eval,
    device,
    seeds=None
):
    config["default"]["epochs"] = 200
    config["dim_x"] = 2
    config["dim_y"] = 1 if config["image_dataset"].startswith("mnist") else 3
    config["dim_context"] = (config["dim_y"],)

    small_value = 0.00001
    config["transform"] = (-small_value, 1 + small_value)  # bounded output

    # convcnp architecture modelled on the synthetic regression experiments:
    config["conv_receptive_field"] = 1
    config["margin"] = 0.1
    config["points_per_unit"] = 128
    config["unet_strides"] = (1,) + (2,) * 4
    config["unet_channels"] = (64,) * 5

    # mnist experiments benefit from training with fixed noise
    config["fix_noise"] = config["image_dataset"].startswith("mnist")
    config["fix_noise_epochs"] = args.fix_noise_epochs or 50

    # where images are stored
    rootdir = os.path.join(*args.datadir)
    os.makedirs(rootdir, exist_ok=True)

    # data generators
    this_seeds = seeds or [10, 20]

    gen_train = nps.ImageGenerator(
        torch.float32,
        rootdir,
        config["image_dataset"],
        seed=this_seeds[0],
        num_tasks=num_tasks_train,
        load_data=True,
        subset="train",
        device=device,
    )
    gen_cv = lambda: nps.ImageGenerator(
        torch.float32,
        rootdir,
        config["image_dataset"],
        seed=this_seeds[1],
        num_tasks=num_tasks_cv,
        load_data=False,
        subset="valid",
        device=device,
    )

    # it is possible to run the image experiment with a custom comparison function
    if config["comparison_function"] == "custom":
        # calculate the difference between pixel locations on a torus
        custom_comparison = partial(toroid_difference, L=gen_train.image_size)
        # pairwise comparison output dimension
        comparison_dim = config["dim_x"]
        # custom comparison function
        config["comparison_function"] = (custom_comparison, comparison_dim, 2*comparison_dim)

    # test setup
    n_min = int(gen_train.n_tot/100)
    n_ctx_list = [n_min * mult for mult in [1, 10, 20, 50]]

    def gens_eval():
        return [
            (
                f"Random {n_ctx}",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    config["image_dataset"],
                    seed=40,
                    num_tasks=num_tasks_eval,
                    load_data=False,
                    subset="test",
                    num_context=UniformDiscrete(n_ctx, n_ctx),
                    device=device,
                ),
            )
            for n_ctx in n_ctx_list]

    return gen_train, gen_cv, gens_eval


register_data("image", setup)
