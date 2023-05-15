from functools import partial

import experiment as exp
import lab as B
import neuralprocesses as nps
import torch as th


class PARAM:
    model_name = "rnp"  # one of rnp, gnp, agnp
    target_name = "hartmann3d"  # one of hartmann{3,6}d, rastrigin, ackley
    grace_period = 50
    save_name = f"BO/{target_name}_{model_name}"

    dim_x, bound = {
        "hartmann3d": (3, (0, 1)),
        "rastrigin": (4, (-1, 1)),
        "ackley": (5, (-32.768, 32.768)),
        "hartmann6d": (6, (0, 1)),
    }[target_name]


# TODO: For now these are taken from Bruinsma's experiments essentially without adaptation
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
    "x_range_context": PARAM.bound,
    "fullconvgnp_kernel_factor": 2,
    "mean_diff": 0,
    # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
    # doesn't make sense to set it to a value higher of the last hidden layer of
    # the CNN architecture. We therefore set it to 64.
    "num_basis_functions": 64,
    "dim_x": PARAM.bound,
}


class mydict(dict):
    def __getattribute__(self, key):
        if key in self:
            return self[key]
        else:
            return super().__getattribute__(key)


args = mydict(
    {
        "dim_x": PARAM.bound,
        "dim_y": 1,
        "data": "eq",
        "batch_size": 32,
        "epochs": 500,
        "rate": 3e-4,
        "objective": "loglik",
        "num_samples": 20,
        "unnormalised": False,
        "evaluate_num_samples": 512,
        "evaluate_batch_size": 8,
        "train_fast": False,
        "evaluate_fast": True,
    }
)


def get_generators(data):
    # gen_train, gen_cv, gens_eval
    return exp.data[data]["setup"](
        args,
        config,
        num_tasks_train=2**6 if args.train_fast else 2**14,
        # num_tasks_train=2**10 if args.train_fast else 2**14,
        num_tasks_cv=2**6 if args.train_fast else 2**12,
        num_tasks_eval=2**6 if args.evaluate_fast else 2**12,
        device="cuda" if th.cuda.is_available() else "cpu",
    )


def get_objectives():
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
    return objective, objective_cv
