def get_config(dim_x, bound):
    return {
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
        "x_range_context": bound,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": 0,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "dim_x": dim_x,
        "dim_y": 1,
        "transform": None,
    }


class mydict(dict):
    def __getattribute__(self, key):
        if key in self:
            return self[key]
        else:
            return super().__getattribute__(key)


def get_args(dim_x, data):
    return mydict(
        {
            "dim_x": dim_x,
            "dim_y": 1,
            "data": data,
            "batch_size": 32,
            "epochs": 500,
            "grace_period": 150,
            "rate": 3e-4,
            "objective": "loglik",
            "num_samples": 20,
            "unnormalised": False,
            "evaluate_num_samples": 512,
            "evaluate_batch_size": 8,
            "train_fast": False,
            "evaluate_fast": True,
            "comparison_function": "difference",
            "non_equivariant_dim": None,
            "seed": 1,
        }
    )
