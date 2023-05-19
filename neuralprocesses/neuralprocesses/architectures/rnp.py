from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from .util import construct_likelihood, parse_transform
from ..util import register_model

__all__ = ["construct_rnp"]


@register_model
def construct_rnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_embedding=256,
    dim_relational_embedding=256,
    attention=False,
    attention_num_heads=8,
    num_enc_layers=3,
    num_relational_enc_layers=3,
    enc_same=False,
    num_dec_layers=6,
    width=256,
    relational_width=64,
    likelihood="lowrank",
    num_basis_functions=512,
    dim_lv=0,
    lv_likelihood="het",
    transform=None,
    dtype=None,
    nps=nps,
    comparison_function="difference",
    relational_encoding_type="simple",
    non_equivariant_dim=None,
):
    """A Slim Relational Neural Process.

    Args:
        non_equivariant_dim: [1] means the second dimension does not need relational encoding
        relational_encoding_type: type of relational_encoding, choose between 'simple' or 'full'
        comparison_function: comparison function used for relational encoding. Defaults to 'difference'
        num_relational_enc_layers: (int, optional): Number of relational encoding layers. Defaults to 3
        relational_width: (int, optional): Widths of all relational MLPs. Defaults to 256.
        dim_relational_embedding: Dimensionality of the relational embeddings. Defaults to 256
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
        if relational_encoding_type == "simple":
            if comparison_function == "distance":
                in_dim = 1 + dim_yci
            elif comparison_function == "partial_distance":
                if non_equivariant_dim is None:
                    raise RuntimeError("you need to specify non-equivariant dim!")
                in_dim = 1 + dim_yci + len(non_equivariant_dim)
            else:
                in_dim = dim_x + dim_yci
        else:
            if comparison_function == "distance":
                in_dim = 2 + 2 * dim_yci
            elif comparison_function == "partial_distance":
                if non_equivariant_dim is None:
                    raise RuntimeError("you need to specify non-equivariant dim!")
                in_dim = 2 + 2 * dim_yci + len(non_equivariant_dim)
            elif comparison_function == "partial_difference":
                if non_equivariant_dim is None:
                    raise RuntimeError("you need to specify non-equivariant dim!")
                in_dim = 2 * (dim_x-len(non_equivariant_dim)) + 2 * dim_yci + len(non_equivariant_dim)
            else:
                in_dim = 2 * dim_x + 2 * dim_yci

        return nps.RelationalMLP(
            in_dim=in_dim,
            relational_out_dim=dim_relational_embedding,
            num_layers=num_relational_enc_layers,
            width=relational_width,
            dtype=dtype,
            comparison_function=comparison_function,
            relational_encoding_type=relational_encoding_type,
            non_equivariant_dim=non_equivariant_dim,
        )

    if len(dim_yc) < 2 or enc_same:
        relational_encoder = construct_relational_mlp(dim_yc[0])
        encoder = nps.Chain(
            nps.RepeatForAggregateInputs(
                nps.RelationalEncode(relational_encoder, encode_input=True)
            ),
            nps.DeterministicLikelihood(),

        )
    else:
        # if enc_same=False we need multiple relational encoder
        relational_encoder = [construct_relational_mlp(dim_yci) for dim_yci in dim_yc]
        encoder = nps.Chain(
            nps.RepeatForAggregateInputs(
                nps.Chain(
                    nps.Copy(len(dim_yc)),
                    nps.Parallel(
                        *(
                            nps.RelationalEncode(relational_encoder[ii], encode_input=True, output_index=ii)
                            for ii in range(len(dim_yc))
                        )
                    ),
                    nps.ConcatRelational(),
                ),
            ),
            nps.DeterministicLikelihood(),
        )

    if comparison_function in ["partial_difference", "partial_distance"]:
        if non_equivariant_dim is None:
            raise RuntimeError("you need to specify non-equivariant dim!")
        nb_non_equivariant_dim = len(non_equivariant_dim)
        dec_dim_input = (dim_relational_embedding + nb_non_equivariant_dim) * len(dim_yc)
    else:
        dec_dim_input = dim_relational_embedding * len(dim_yc)

    decoder = nps.Chain(
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.MLP(
                    in_dim=dec_dim_input,
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
