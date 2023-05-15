import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []

def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 200
    config["dim_x"] = 3
    config["dim_y"] = 1

    obs_type = "diff"  # todo: should read this from args

    if obs_type in ["sane", "cancer"]:
        config["transform"] = "softplus"
    else:
        config["transform"] = None

    # Configure the convolutional models:
    config["points_per_unit"] = 4
    config["margin"] = 1
    config["conv_receptive_field"] = 100
    config["unet_strides"] = (1,) + (2,) * 6
    config["unet_channels"] = (64,) * 7
    config["num_layers"] = 4
    config["dim_embedding"] = 128
    config["dim_relational_embedding"] = 128
    config["num_basis_functions"] = 32


    gen_train = nps.CancerGenerator(
        torch.float32,
        seed=10,
        dataset="small_train",
        obs_type=obs_type,
        num_tasks=num_tasks_train,
        mode="completion",
        device=device
    )
    gen_cv = lambda: nps.CancerGenerator(
        torch.float32,
        seed=20,
        dataset="small_train",
        obs_type=obs_type,
        num_tasks=num_tasks_cv,
        mode="completion",
        device=device
    )

    def gens_eval():
        return [
            # For the real tasks, the batch size will be one. Keep the number of batches
            # the same.
            (
                "Completion (Simulated)",
                nps.CancerGenerator(
                    torch.float32,
                    seed=30,
                    dataset="small_test",
                    obs_type=obs_type,
                    num_tasks=num_tasks_eval,
                    mode="forecasting",
                    device=device
                ),
            ),
            (
                "Forecasting (Simulated)",
                nps.CancerGenerator(
                    torch.float32,
                    seed=40,
                    dataset="small_test",
                    obs_type=obs_type,
                    num_tasks=num_tasks_eval,
                    mode="forecasting",
                    device=device
                ),
            )
            
        ]

    return gen_train, gen_cv, gens_eval


register_data("cancer", setup)
