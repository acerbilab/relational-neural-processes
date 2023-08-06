"""from functools import partial
import lab as B

import neuralprocesses.torch as nps
import torch
from ...neuralprocesses.dist.uniform import UniformDiscrete, UniformContinuous

from .util import register_data

__all__ = []


def setup(
    name, args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device, seeds
):
    config["dim_x"] = args.dim_x
    config["dim_y"] = args.dim_y

    if "x_range_context" in config.keys():
        x_range_context = config["x_range_context"]
        x_range_target = config["x_range_context"]
    else:
        x_range_context = (-1, 1)
        x_range_target = (-1, 1)

    # Architecture choices specific for the GP experiments:
    # TODO: We should use a stride of 1 in the first layer, but for compatibility
    #    reasons with the models we already trained, we keep it like this.
    config["unet_strides"] = (2,) * 6
    config["conv_receptive_field"] = 4
    config["margin"] = 0.1
    if args.dim_x == 1:
        config["points_per_unit"] = 64
    elif args.dim_x != 1:  # before: args.dim_x == 2
        # Reduce the PPU to reduce memory consumption.
        config["points_per_unit"] = 32
        # Since the PPU is reduced, we can also take off a layer of the UNet.
        config["unet_strides"] = config["unet_strides"][:-1]
        config["unet_channels"] = config["unet_channels"][:-1]
    else:
        raise RuntimeError(f"Invalid input dimensionality {args.dim_x}.")
    # Relational encoder setup
    if args.dim_x > 3:
        config["width"] = 128
        config["relational_width"] = 128
        config["dim_embedding"] = 128
        config["dim_relational_embeddings"] = 128

    # Other settings specific to the GP experiments:
    config["plot"] = {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 2), (-2, 2))},
    }
    factor = B.sqrt(args.dim_x)
    config = {
        "num_tasks": args.num_tasks,
        "batch_size": args.batch_size,
        "dist_x_context": UniformContinuous(*((x_range_context,) * args.dim_x)),
        "dist_x_target": UniformContinuous(*((x_range_target,) * args.dim_x)),
        "dim_y": args.dim_y,
        "device": args.device,
        "transform":None
    }
    gen_train = nps.GPGeneratorRotate(            
        torch.float32,
        seed=int(seeds[0]),
        noise=0.05,
        kernel=EQ().stretch(factor * 1),
        num_context=UniformDiscrete(1, 30 * args.dim_x),
        num_target=UniformDiscrete(50 * args.dim_x, 50 * args.dim_x),         
        pred_logpdf=False,
        pred_logpdf_diag=False,
        **config,
    )

    gen_cv = lambda: nps.GPGeneratorRotate(
        torch.float32,
        seed=int(seeds[0]),
        noise=0.05,
        kernel=EQ().stretch(factor * 1),
        num_context=UniformDiscrete(1, 30 * args.dim_x),
        num_target=UniformDiscrete(50 * args.dim_x, 50 * args.dim_x),         
        pred_logpdf=False,
        pred_logpdf_diag=False,
        **config,
    )

    def gens_eval():
        return [
            (
                eval_name,
                nps.GPGeneratorRotate(
                    torch.float32,
                    seed=int(seeds[0]),
                    noise=0.05,
                    kernel=EQ().stretch(factor * 1),
                    num_context=UniformDiscrete(1, 30 * args.dim_x),
                    num_target=UniformDiscrete(50 * args.dim_x, 50 * args.dim_x),         
                    pred_logpdf=False,
                    pred_logpdf_diag=False,
                    **config,
                ),
            )
            for eval_name, x_range_context, x_range_target in [
                ("interpolation in training range", (-2, 2), (-2, 2)),
                ("interpolation beyond training range", (2, 6), (2, 6)),
                ("extrapolation beyond training range", (-2, 2), (2, 6)),
            ]
        ]

    return gen_train, gen_cv, gens_eval


names = [
    "gp_rotate",
]

for name in names:
    register_data(
        name,
        partial(setup, name),
        requires_dim_x=True,
        requires_dim_y=True,
    )"""
