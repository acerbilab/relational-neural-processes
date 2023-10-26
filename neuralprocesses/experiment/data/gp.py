from functools import partial
import wbml.out as out
import neuralprocesses.torch as nps
import torch

from .util import register_data

__all__ = []


def setup(
    name, args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device, seeds
):
    config["dim_x"] = args.dim_x
    config["dim_y"] = args.dim_y

    # k is for sparse FullRCNP
    ratio = 0.1
    if name in ["eq", "matern", "weakly-periodic"]:
        config["k"] = int(30 * args.dim_x * ratio)
    else:
        if args.dim_x == 1:
            config["k"] = 30 * ratio
        else:
            config["k"] = int(75 * args.dim_x * ratio)

    if "x_range_context" in config.keys():
        x_range_context = config["x_range_context"]
        x_range_target = config["x_range_context"]
    else:
        if name in ["bo_fixed", "bo_matern", "bo_single", "bo_sumprod"]:
            x_range_context = (-1, 1)
            x_range_target = (-1, 1)
        elif name == "gp_rotate":
            x_range_context = (-4, 0)
            x_range_target = (-4, 0)
        else:
            x_range_context = (-2, 2)
            x_range_target = (-2, 2)

    out.kv("context training range", x_range_context)
    out.kv("target training range", x_range_target)

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
    config["transform"] = None

    gen_train = nps.construct_predefined_gens(
        torch.float32,
        seed=int(seeds[0]),
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=False,
        pred_logpdf_diag=False,
        x_range_context=x_range_context,
        x_range_target=x_range_target,
        device=device,
        mean_diff=config["mean_diff"],
    )[name]

    gen_cv = lambda: nps.construct_predefined_gens(
        torch.float32,
        seed=int(seeds[1]),  # Use a different seed!
        batch_size=args.batch_size,
        num_tasks=num_tasks_cv,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        # comment this out later
        x_range_context=x_range_context,
        x_range_target=x_range_target,
        device=device,
        mean_diff=config["mean_diff"],
    )[name]

    def gens_eval():
        if args.data in ["bo_fixed", "bo_matern", "bo_single", "bo_sumprod"]:
            x_range_context_int = (-1, 1)
            x_range_target_int = (-1, 1)
            # check if BO tasks need OOID
            x_range_context_ooid = (-1, 1)
            x_range_target_ooid = (-1, 1)
        elif args.data == "gp_rotate":
            x_range_context_int = (-4, 0)
            x_range_target_int = (-4, 0)
            x_range_context_ooid = (0, 4)
            x_range_target_ooid = (0, 4)

        else:
            x_range_context_int = (-2, 2)
            x_range_target_int = (-2, 2)
            x_range_context_ooid = (2, 6)
            x_range_target_ooid = (2, 6)

        return [
            (
                eval_name,
                nps.construct_predefined_gens(
                    torch.float32,
                    seed=30,  # Use yet another seed!
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    dim_x=args.dim_x,
                    dim_y=args.dim_y,
                    pred_logpdf=True,
                    pred_logpdf_diag=True,
                    device=device,
                    x_range_context=x_range_context,
                    x_range_target=x_range_target,
                    mean_diff=config["mean_diff"],
                )[args.data],
            )
            for eval_name, x_range_context, x_range_target in [
                ("interpolation in training range", x_range_context_int, x_range_target_int),
                ("interpolation beyond training range", x_range_context_ooid, x_range_target_ooid),
                # for now we don't care about EXT
                # ("extrapolation beyond training range", (-2, 2), (2, 6)),
            ]
        ]

    return gen_train, gen_cv, gens_eval


names = [
    "eq",
    "matern",
    "weakly-periodic",
    "mix-eq",
    "mix-matern",
    "mix-weakly-periodic",
    "sawtooth",
    "mixture",
    "bo_fixed",
    "bo_matern",
    "bo_single",
    "bo_sumprod",
    "gp_rotate",
]

for name in names:
    register_data(
        name,
        partial(setup, name),
        requires_dim_x=True,
        requires_dim_y=True,
    )
