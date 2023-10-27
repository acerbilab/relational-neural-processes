import neuralprocesses as nps  # This fixes inspection below.
from ..util import register_model

__all__ = ["construct_fullrnp"]


@register_model
def construct_fullrnp(*args, nps=nps, **kw_args):
    """Full version of Relational Neural Process.

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
        dim_relational_embedding (int, optional): Dimensionality of the relational embeddings. Defaults to 256
        attention (bool, optional): Use attention for the deterministic encoder.
            Defaults to `False`.
        attention_num_heads (int, optional): Number of heads. Defaults to `8`.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to 3.
        num_relational_enc_layers: (int, optional): Number of relational encoding layers. Defaults to 3
        enc_same (bool, optional): Use the same encoder for all context sets. This
            only works if all context sets have the same dimensionality. Defaults to
            `False`.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        relational_width: (int, optional): Widths of all relational MLPs. Defaults to 256.
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
        comparison_function (str or tuple, optional): comparison function used for relational encoding.
            Defaults to 'different'.
        relational_encoding_type (str, optional): type of relational_encoding, choose between 'simple' or 'full'
        non_equivariant_dim: indicates which dimensions are non-equivariant. Defaults to an empty list.
        sparse (bool, optional): whether to use sparse rcnp or not. Defaults to False.
        k (int, optional): for sparse rcnp, we only choose top-k nearest context points for encoding.

    Returns:
        :class:`.model.Model`: FullRGNP model.
    """
    return nps.construct_rnp(
        *args,
        nps=nps,
        relational_encoding_type="full",
        **kw_args,
    )
