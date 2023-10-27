import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []

def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device, seeds=None):
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 100
    config["dim_x"] = 4
    config["dim_y"] = 1

    config["transform"] = None
    
    # Configure the convolutional models:
    config["points_per_unit"] = 4
    config["margin"] = 1
    config["conv_receptive_field"] = 100
    config["unet_strides"] = (1,) + (2,) * 6
    config["unet_channels"] = (64,) * 7
    config["num_layers"] = 4
    config["num_basis_functions"] = 64


    this_seeds = seeds or [10, 20]

    gen_train = nps.CancerLatentGenerator(
        torch.float32,
        seed=this_seeds[0],
        dataset="small_train",
        num_tasks=num_tasks_train,
        mode="completion",
        device=device
    )
    gen_cv = lambda: nps.CancerLatentGenerator(
        torch.float32,
        seed=this_seeds[1],
        dataset="small_train",
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
                nps.CancerLatentGenerator(
                    torch.float32,
                    seed=30,
                    dataset="small_test",
                    num_tasks=num_tasks_eval,
                    mode="completion",
                    device=device
                ),
            ),
            (
                "Forecasting (Simulated)",
                nps.CancerLatentGenerator(
                    torch.float32,
                    seed=40,
                    dataset="small_test",
                    num_tasks=num_tasks_eval,
                    mode="forecasting",
                    device=device
                ),
            )
            
        ]

    return gen_train, gen_cv, gens_eval


register_data("cancer_latent", setup)