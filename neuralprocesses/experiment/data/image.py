import torch

import neuralprocesses.torch as nps
from .util import register_data
from neuralprocesses.dist import UniformDiscrete

__all__ = []


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
    config["dim_x"] = 2
    config["dim_y"] = 1

    config["transform"] = None  # TODO: should be sigmoid

    # convcnp architecture from the gp experiments:
    config["unet_strides"] = (2,) * 6
    config["conv_receptive_field"] = 4
    config["margin"] = 0.1
    config["points_per_unit"] = 32
    # Since the PPU is reduced, we can also take off a layer of the UNet.
    config["unet_strides"] = config["unet_strides"][:-1]
    config["unet_channels"] = config["unet_channels"][:-1]

    # data generators
    this_seeds = seeds or [10, 20]
    rootdir = "data"  # note: user needs to create this, TODO include in args
    dataset = config["image_dataset"]

    gen_train = nps.ImageGenerator(
        torch.float32,
        rootdir,
        dataset,
        seed=this_seeds[0],
        num_tasks=num_tasks_train,
        #num_context=UniformDiscrete(10, 400),  # use default: uni(ntot/2, ntot/100)
        device=device,
    )
    gen_cv = lambda: nps.ImageGenerator(
        torch.float32,
        rootdir,
        dataset,
        seed=this_seeds[1],
        num_tasks=num_tasks_cv,
        #num_context=UniformDiscrete(10, 400),  # use default: uni(ntot/2, ntot/100)
        device=device,
    )

    # test setup
    if dataset == "mnist":
        n_ctx_list = [10, 100, 200, 400]
    elif dataset == "mnist16":
        n_ctx_list = [5, 20, 50, 100]
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    def gens_eval():
        return [
            (
                f"Random {n_ctx}",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    dataset,
                    seed=30,
                    num_tasks=num_tasks_eval,
                    num_context=UniformDiscrete(n_ctx, n_ctx),
                    subset="test",
                    device=device,
                ),
            )
            for n_ctx in n_ctx_list]

    return gen_train, gen_cv, gens_eval


register_data("image", setup)
