import lab as B
import torch
import botorch as bth

from config import config
import neuralprocesses.torch as nps


def train_epoch(
    state,
    model,
    opt,
    objective,
    gen,
    device,
    *,
    fix_noise,
):
    """Train for an epoch."""
    vals = []
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            fix_noise=fix_noise,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    vals = B.concat(*vals)
    return state, vals.mean(), B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


def eval(state, model, objective, gen, device):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        vals = B.concat(*vals)
        # out.kv("Loglik (VV)", exp.with_err(vals, and_lower=True))
        # if kls:
        #     out.kv("KL (full)", exp.with_err(B.concat(*kls), and_upper=True))
        # if kls_diag:
        #     out.kv("KL (diag)", exp.with_err(B.concat(*kls_diag), and_upper=True))

        # objective doesn't return pred_y, we can't plot the data

        return state, vals.mean(), B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


def get_model(model_name, args, device):
    if model_name == "cnp":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            transform=config["transform"],
        )
    elif model_name == "rcnp":
        model = nps.construct_rnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            dim_relational_embedding=config["dim_relational_embeddings"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            relational_width=config["relational_width"],
            num_relational_enc_layers=config["num_relational_layers"],
            likelihood="het",
            transform=config["transform"],
            comparison_function=args.comparison_function,
        )
    elif model_name == "rgnp":
        model = nps.construct_rnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            dim_relational_embedding=config["dim_relational_embeddings"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            relational_width=config["relational_width"],
            num_relational_enc_layers=config["num_relational_layers"],
            likelihood="lowrank",
            transform=config["transform"],
            comparison_function=args.comparison_function,
        )
    elif model_name == "srcnp":
        model = nps.construct_srnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_relational_embedding=config["dim_relational_embeddings"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            relational_width=config["relational_width"],
            num_relational_enc_layers=config["num_relational_layers"],
            likelihood="het",
            transform=config["transform"],
            comparison_function=args.comparison_function,
        )
    elif model_name == "srgnp":
        model = nps.construct_srnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_relational_embedding=config["dim_relational_embeddings"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            relational_width=config["relational_width"],
            num_relational_enc_layers=config["num_relational_layers"],
            likelihood="lowrank",
            transform=config["transform"],
            comparison_function=args.comparison_function,
        )
    elif model_name == "gnp":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    elif model_name == "np":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            dim_lv=config["dim_embedding"],
            transform=config["transform"],
        )
    elif model_name == "acnp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            transform=config["transform"],
        )
    elif model_name == "agnp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    elif model_name == "anp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            enc_same=config["enc_same"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            dim_lv=config["dim_embedding"],
            transform=config["transform"],
        )
    elif model_name == "convcnp":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
        )
    elif model_name == "convgnp":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            num_basis_functions=config["num_basis_functions"],
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
        )
    elif model_name == "convnp":
        if config["dim_x"] == 2:
            # Reduce the number of channels in the conv. architectures by a factor
            # $\sqrt(2)$. This keeps the runtime in check and reduces the parameters
            # of the ConvNP to the number of parameters of the ConvCNP.
            config["unet_channels"] = tuple(
                int(c / 2**0.5) for c in config["unet_channels"]
            )
            config["dws_channels"] = int(config["dws_channels"] / 2**0.5)
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            dim_lv=16,
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
        )
    elif model_name == "fullconvgnp":
        model = nps.construct_fullconvgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            kernel_factor=config["fullconvgnp_kernel_factor"],
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
        )
    else:
        raise ValueError(f'Invalid model "{model_name}".')

    model = model.to(device)
    return model


# See https://www.sfu.ca/~ssurjano/optimization.html for target details
def get_target(target_name):
    data_set_dims = {
        "hartmann3d": (3, (0, 1)),
        "rastrigin": (4, (-1, 1)),
        "ackley": (5, (-32.768, 32.768)),
        "hartmann6d": (6, (0, 1)),
    }

    if target_name == "hartmann3d":
        target = bth.test_functions.Hartmann(3)
        tar_min = -3.86278
    elif target_name == "rastrigin":
        target = bth.test_functions.Rastrigin(4)
        tar_min = 0.0
    elif target_name == "ackley":
        target = bth.test_functions.Ackley(5)
        tar_min = 0.0
    elif target_name == "hartmann6d":
        target = bth.test_functions.Hartmann(3)
        tar_min = -3.32237
    else:
        raise NotImplementedError()

    return target, data_set_dims[target_name], tar_min
