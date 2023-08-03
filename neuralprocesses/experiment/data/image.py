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

    this_seeds = seeds or [10, 20]
    rootdir = "data"
    dataset = config["image_dataset"]

    gen_train = nps.ImageGenerator(
        torch.float32,
        rootdir,
        dataset,
        seed=this_seeds[0],
        num_tasks=num_tasks_train,
        num_context=UniformDiscrete(10, 400),
        device=device,
    )
    gen_cv = lambda: nps.ImageGenerator(
        torch.float32,
        rootdir,
        dataset,
        seed=this_seeds[1],
        num_tasks=num_tasks_cv,
        num_context=UniformDiscrete(10, 400),
        device=device,
    )

    def gens_eval():
        return [
            (
                "Random 10",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    dataset,
                    seed=30,
                    num_tasks=num_tasks_eval,
                    subset="test",
                    device=device,
                ),
            ),
            (
                "Random 100",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    dataset,
                    seed=40,
                    num_tasks=num_tasks_eval,
                    num_context=UniformDiscrete(100, 100),
                    subset="test",
                    device=device,
                ),
            ),
            (
                "Random 200",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    dataset,
                    seed=50,
                    num_tasks=num_tasks_eval,
                    num_context=UniformDiscrete(200, 200),
                    subset="test",
                    device=device,
                ),
            ),
            (
                "Random 400",
                nps.ImageGenerator(
                    torch.float32,
                    rootdir,
                    dataset,
                    seed=60,
                    num_tasks=num_tasks_eval,
                    num_context=UniformDiscrete(400, 400),
                    subset="test",
                    device=device,
                ),
            ),
        ]

    return gen_train, gen_cv, gens_eval


register_data("image", setup)
