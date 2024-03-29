from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from .util import construct_likelihood, parse_transform
from ..util import register_model

__all__ = ["construct_tnp"]


@register_model
def construct_tnp(
        dim_x=1,
        dim_y=1,
        dim_yc=None,
        dim_yt=None,
        dim_embedding=256,
        num_heads=4,
        num_tfm_layers=6,
        enc_same=False,
        num_dec_layers=6,
        width=512,
        likelihood="het",
        num_basis_functions=512,
        transform=None,
        dtype=None,
        nps=nps,
):
    """A Transformer Neural Process.

    Args:
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
        num_heads (int, optional): Number of transformer heads. Defaults to 4.
        num_tfm_layers (int, optional): Number of transformer layers. Defaults to 6.
        enc_same (bool, optional): Use the same encoder for all context sets. This
            only works if all context sets have the same dimensionality. Defaults to
            `False`.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank"`.
            Defaults to `"lowrank"`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to 512.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: GNP model.
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

    def construct_tfm(dim_yci):
        return nps.Transformer(
            # fix all parameters
            dim_x=dim_x,
            dim_y=dim_yci,
            dim_embedding=dim_embedding,
            num_layers=num_tfm_layers,
            depth=2,
            dtype=dtype,
            num_heads=num_heads,
            width=width,
        )

    transformer_encoder = construct_tfm(dim_yc[0])
    encoder = nps.Chain(
        nps.RepeatForAggregateInputs(
            nps.RelationalEncode(transformer_encoder, encode_target=True)
        ),
        nps.DeterministicLikelihood(),
    )

    decoder = nps.Chain(
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.MLP(
                    in_dim=dim_embedding * len(dim_yc),
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
