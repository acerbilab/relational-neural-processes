import lab as B
from typing import Tuple, Union, Optional
from .. import _dispatch
from ..util import register_module, batch

__all__ = ["Attention", "SelfAttention", "RelationalAttention"]


@register_module
class Attention:
    """Attention module.

    Args:
        dim_x (int): Dimensionality of the inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_embedding (int): Dimensionality of the embedding.
        num_heads (int): Number of heads.
        num_enc_layers (int): Number of layers in the encoders.
        dtype (dtype, optional): Data type.

    Attributes:
        num_heads (int): Number of heads.
        dim_head (int): Dimensionality of a head.
        encoder_x (function): Encoder for the inputs.
        encoder_xy (function): Encoder for the inputs-output pairs.
        mixer (function): Mixer.
        mlp1 (function): First MLP for the final normalisation layers.
        ln1 (function): First normaliser for the final normalisation layers.
        mlp2 (function): Second MLP for the final normalisation layers.
        ln2 (function): Second normaliser for the final normalisation layers.
    """

    def __init__(
        self,
        dim_x,
        dim_y,
        dim_embedding,
        num_heads,
        num_enc_layers,
        dtype=None,
        _self=False,
    ):
        self._self = _self

        self.num_heads = num_heads
        self.dim_head = dim_embedding // num_heads

        self.encoder_x = self.nps.MLP(
            in_dim=dim_x,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )
        self.encoder_xy = self.nps.MLP(
            in_dim=dim_x + dim_y,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )

        self.mixer = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.mlp1 = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln1 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)
        self.mlp2 = self.nps.MLP(
            in_dim=dim_embedding,
            layers=(dim_embedding,),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln2 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)

    def _extract_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        return B.reshape(z, *b, b_last * self.num_heads, c // self.num_heads, n)

    def _compress_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        # Torch dimensions may not support `//`, so convert to an integer.
        return B.reshape(z, *b, b_last // self.num_heads, c * self.num_heads, n)


@_dispatch
def code(coder: Attention, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    if coder._self:
        # Perform self-attention rather than cross-attention.
        x = xz

    if B.shape(z, -1) == 0:
        # Handle the case of empty context set.
        queries = coder.encoder_x(x)
        with B.on_device(z):
            z = B.zeros(
                B.dtype(z),
                *batch(z, 3),
                # Return in compressed head format.
                B.shape(z, -3) * coder.num_heads,
                coder.dim_head,
                B.shape(x, -1),
            )
    else:
        keys = coder._extract_heads(coder.encoder_x(xz))
        values = coder._extract_heads(coder.encoder_xy(B.concat(xz, z, axis=-2)))
        # print("xy", B.concat(xz, z, axis=-2).shape)
        # print("output", coder.encoder_xy(B.concat(xz, z, axis=-2)).shape)
        # print(values.shape)
        # Don't extract heads for the queries here, because we need it is this form down
        # at the final normalisation layers.
        queries = coder.encoder_x(x)

        activations = B.matmul(keys, coder._extract_heads(queries), tr_a=True)

        activations = activations / B.sqrt(B.shape(keys, -2))  # Keep variance constant.

        z = B.matmul(values, B.softmax(activations, axis=-2))
        print(B.softmax(activations, axis=-2)[0, :, 0])
        print(z.shape)

    # Mix heads.
    z = coder.mixer(coder._compress_heads(z))

    # Apply final two residual normalisation layers.
    z = coder.ln1(z + coder.mlp1(queries))
    z = coder.ln2(z + coder.mlp2(z))
    print(z.shape)

    return x, z


@register_module
class SelfAttention(Attention):
    """Self-attention module.

    Args:
        dim_x (int): Dimensionality of the inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_embedding (int): Dimensionality of the embedding.
        num_heads (int): Number of heads.
        num_enc_layers (int): Number of layers in the encoders.
        dtype (dtype, optional): Data type.

    Attributes:
        num_heads (int): Number of heads.
        dim_head (int): Dimensionality of a head.
        encoder_x (function): Encoder for the inputs.
        encoder_xy (function): Encoder for the inputs-output pairs.
        mixer (function): Mixer.
        mlp1 (function): First MLP for the final normalisation layers.
        ln1 (function): First normaliser for the final normalisation layers.
        mlp2 (function): Second MLP for the final normalisation layers.
        ln2 (function): Second normaliser for the final normalisation layers.
    """

    def __init__(self, *args, **kw_args):
        Attention.__init__(self, *args, _self=True, **kw_args)


@register_module
class RelationalAttention:
    def __init__(self,
                 dim_x,
                 dim_y,
                 in_dim: int,
                 relational_out_dim: int,
                 dim_embedding: int,
                 num_heads: int,
                 layers: Optional[Tuple[int, ...]] = None,
                 num_relational_enc_layers: Optional[int] = None,
                 num_enc_layers: Optional[int] = None,
                 width: Optional[int] = None,
                 nonlinearity=None,
                 dtype=None,
                 comparison_function="distance",
                 relational_encoding_type="simple",
                 non_equivariant_dim=None,
                 ):
        # relational encoder part
        self.relational_out_dim = relational_out_dim
        layers_given = layers is not None
        num_layers_given = num_relational_enc_layers is not None and width is not None
        if not (layers_given or num_layers_given):
            raise ValueError(
                f"Must specify either `layers` or `num_layers` and `width`."
            )
        # Make sure that `layers` is a tuple of various widths.
        if not layers_given and num_layers_given:
            layers = (width,) * num_relational_enc_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = self.nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = self.nn.Linear(in_dim, relational_out_dim, dtype=dtype)
        else:
            net = [self.nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(self.nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(self.nn.Linear(layers[-1], relational_out_dim, dtype=dtype))
            self.net = self.nn.Sequential(*net)

        self.comparison_function = comparison_function
        self.relational_encoding_type = relational_encoding_type
        self.non_equivariant_dim = non_equivariant_dim

        # attention part
        self.num_heads = num_heads
        self.dim_head = dim_embedding // num_heads

        self.encoder_x = self.nps.MLP(
            in_dim=dim_x,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )
        self.encoder_xy = self.nps.MLP(
            in_dim=relational_out_dim,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )

        self.mixer = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.mlp1 = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln1 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)
        self.mlp2 = self.nps.MLP(
            in_dim=dim_embedding,
            layers=(dim_embedding,),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln2 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)

    def _extract_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        return B.reshape(z, *b, b_last * self.num_heads, c // self.num_heads, n)

    def _compress_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        # Torch dimensions may not support `//`, so convert to an integer.
        return B.reshape(z, *b, b_last // self.num_heads, c * self.num_heads, n)

    def __call__(self, xc, yc, xt):
        xc = B.transpose(xc)
        yc = B.transpose(yc)
        xt = B.transpose(xt)

        batch_size, set_size, feature_dim = xc.shape
        _, target_set_size, _ = xt.shape
        _, _, out_dim = yc.shape

        if self.relational_encoding_type == "simple":
            if self.comparison_function == "distance":
                # (batch_size, target_set_size, set_size, 1))
                dist_x = B.sqrt(
                    B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(
                        -1
                    )
                )
                # (batch_size, target_set_size, set_size, 2))
                dist_x = B.concat(
                    dist_x, yc.unsqueeze(1).repeat(1, target_set_size, 1, 1), axis=-1
                )
                # (batch_size, target_set_size * set_size, 2))
                dist_x = dist_x.reshape(batch_size, -1, 1 + out_dim)
                batch_size, diff_size, filter_size = dist_x.shape
                # (batch_size * target_set_size * set_size, 2))
                x = dist_x.view(batch_size * diff_size, filter_size)
            elif self.comparison_function == "difference":
                xc_pairs = B.concat(xc, yc, axis=-1).unsqueeze(1)
                xt_pairs = B.concat(
                    xt,
                    B.cast(xt.dtype, B.zeros(batch_size, target_set_size, 1)),
                    axis=-1,
                ).unsqueeze(2)

                diff_x = (xt_pairs - xc_pairs).reshape(
                    batch_size, -1, feature_dim + out_dim
                )
                batch_size, diff_size, filter_size = diff_x.shape
                x = diff_x.view(batch_size * diff_size, filter_size)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        x = self.net(x)

        # the values v
        x = x.view(batch_size, target_set_size, set_size, self.relational_out_dim)
        # encoded_target_x = x.sum(dim=2)

        values = self.encoder_xy(B.transpose(x)).reshape(batch_size*self.num_heads, target_set_size, set_size, -1)
        # values = x.view(batch_size*self.num_heads, target_set_size, set_size, -1)

        # the keys
        keys = self._extract_heads(self.encoder_x(B.transpose(xc)))
        # the queries
        queries = self.encoder_x(B.transpose(xt))
        activations = B.matmul(keys, self._extract_heads(queries), tr_a=True)
        activations = activations / B.sqrt(B.shape(keys, -2))
        # batch_size, set_size, target_set_size
        activations = B.transpose(B.softmax(activations, axis=-2))
        # print(activations[0, 0])
        # encoded_target_x = B.einsum('aij,aijk->aik', activations, values)
        activations = B.expand_dims(activations, axis=-1)
        encoded_target_x = activations * values
        encoded_target_x = encoded_target_x.sum(dim=2)

        encoded_target_x = B.transpose(encoded_target_x)

        z = self.mixer(self._compress_heads(encoded_target_x))

        # Apply final two residual normalisation layers.
        z = self.ln1(z + self.mlp1(queries))
        z = self.ln2(z + self.mlp2(z))
        # we should return (batch_size, encoded_feature r, target_set_size)
        return z
        # return encoded_target_x

@_dispatch
def code(coder: RelationalAttention, xz, z: B.Numeric, x, **kw_args):
    xz = coder(xz, z, x)
    return xz, z
