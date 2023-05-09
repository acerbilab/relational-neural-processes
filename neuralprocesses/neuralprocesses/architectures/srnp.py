from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from .util import construct_likelihood, parse_transform
from ..util import register_model

__all__ = ["construct_srnp"]


@register_model
def construct_srnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_embedding=256,
    dim_relational_embedding=64,
    attention=False,
    attention_num_heads=8,
    num_enc_layers=3,
    num_relational_enc_layers=2,
    enc_same=False,
    num_dec_layers=6,
    width=512,
    relational_width=64,
    likelihood="lowrank",
    num_basis_functions=512,
    dim_lv=0,
    lv_likelihood="het",
    transform=None,
    dtype=None,
    nps=nps,
    comparison_function="euclidean",
):
    """A Slim Relational Neural Process.

    Args:
        num_relational_enc_layers:
        relational_width:
        dim_relational_embedding: Dimensionality of the relational embeddings. Defaults to 64
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_embedding (int, optional): Dimensionality of the embedding. Defaults to 128.
        attention (bool, optional): Use attention for the deterministic encoder.
            Defaults to `False`.
        attention_num_heads (int, optional): Number of heads. Defaults to `8`.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to 3.
        enc_same (bool, optional): Use the same encoder for all context sets. This
            only works if all context sets have the same dimensionality. Defaults to
            `False`.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank"`.
            Defaults to `"lowrank"`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to 512.
        dim_lv (int, optional): Dimensionality of the latent variable. Defaults to 0.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"` or `"dense"`. Defaults to `"het"`.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: RNP model.
    """
    # Make sure that `dim_yc` is initialised and a tuple.
    dim_yc = convert(dim_yc or dim_y, tuple)
    # Make sure that `dim_yt` is initialised.
    dim_yt = dim_yt or dim_y

    # Check if `enc_same` can be used.
    if enc_same and any(dim_yci != dim_yc[0] for dim_yci in dim_yc[1:]):
        raise ValueError(
            "Can only use the same encoder for all context sets if the context sets "
            "are of the same dimensionality, but they are not."
        )

    mlp_out_channels, selector, likelihood = construct_likelihood(
            nps,
            spec=likelihood,
            dim_y=dim_yt,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )

    def construct_relational_mlp(dim_yci):
        return nps.RelationalMLP(
            in_dim=1 if comparison_function == "euclidean" else dim_x + dim_yci,
            relational_out_dim=dim_relational_embedding,
            num_layers=num_relational_enc_layers,
            width=relational_width,
            dtype=dtype,
            comparison_function=comparison_function
        )


    relational_encoder = construct_relational_mlp(dim_yc[0])


    encoder = nps.Chain(
        nps.RepeatForAggregateInputs(
            # encode here
            nps.Chain(
                relational_encoder,  # if enc_same=False we need multiple relational encoder
                nps.RelationalEncode(nps.InputsCoder()),
            )
        ),
        nps.DeterministicLikelihood(),

    )
    decoder = nps.Chain(
        # nps.Concatenate(),
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.MLP(
                    in_dim=dim_relational_embedding,
                    out_dim=mlp_out_channels,
                    num_layers=num_dec_layers,
                    # The capacity of this MLP should increase with the number of
                    # outputs, but multiplying by `len(dim_yc)` is too aggressive.
                    width=width * min(len(dim_yc), 2),
                    dtype=dtype,
                ),
                selector,  # Select the right target output.
            )
        ),
        likelihood,
        parse_transform(nps, transform=transform),
    )
    return nps.Model(encoder, decoder)