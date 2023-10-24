import math
from functools import partial
from typing import Tuple, Union, Optional

import lab as B
from plum import convert
import torch
import torch.nn as nn

from .. import _dispatch
from ..datadims import data_dims
from ..parallel import Parallel
from ..util import register_module, compress_batch_dimensions, with_first_last

__all__ = ["Linear", "MLP", "UNet", "ConvNet", "Conv", "ResidualBlock", "RelationalMLP", "Transformer"]


@register_module
class Linear:
    """A linear layer over channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): Linear layer.
    """

    def __init__(self, in_channels, out_channels, dtype):
        self.net = self.nn.Linear(in_channels, out_channels, dtype=dtype)


@register_module
class MLP:
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity=None,
        dtype=None,
    ):
        # Check that one of the two specifications is given.
        layers_given = layers is not None
        num_layers_given = num_layers is not None and width is not None
        if not (layers_given or num_layers_given):
            raise ValueError(
                f"Must specify either `layers` or `num_layers` and `width`."
            )
        # Make sure that `layers` is a tuple of various widths.
        if not layers_given and num_layers_given:
            layers = (width,) * num_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = self.nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = self.nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [self.nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(self.nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(self.nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = self.nn.Sequential(*net)

    def __call__(self, x):
        x = B.transpose(x)
        x, uncompress = compress_batch_dimensions(x, 2)

        x = self.net(x)

        x = uncompress(x)
        x = B.transpose(x)

        return x


@_dispatch
def code(coder: Union[Linear, MLP], xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)

    # Construct permutation to switch the channel dimension and the last dimension.
    switch = list(range(B.rank(z)))
    switch[-d - 1], switch[-1] = switch[-1], switch[-d - 1]

    # Switch, compress, apply network, uncompress, and undo switch.
    z = B.transpose(z, perm=switch)
    z, uncompress = compress_batch_dimensions(z, 2)
    z = coder.net(z)
    z = uncompress(z)
    z = B.transpose(z, perm=switch)

    return xz, z

@register_module
class Transformer:
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        num_layers: int,
        depth: int,
        num_heads: int,
        width: int,
        dtype=None,
    ):
        self.embedder = self.build_mlp(dim_x + dim_y, dim_embedding, dim_embedding, depth)

        encoder_layer = nn.TransformerEncoderLayer(dim_embedding, num_heads, width, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def build_mlp(self, dim_in, dim_hid, dim_out, depth):
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth - 2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
        return nn.Sequential(*modules)

    def create_mask(self, xc, yc, xt):
        num_ctx = xc.shape[1]
        num_tar = xt.shape[1]
        num_all = num_ctx + num_tar

        mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
        mask = B.cast(xc.dtype, mask)
        mask[:, :num_ctx] = 0.0

        return mask, num_tar

    def __call__(self,  xc, yc, xt):
        # (batch_size, set_size, feature_dim)
        xc = B.transpose(xc)
        yc = B.transpose(yc)
        xt = B.transpose(xt)
        x_y_context = B.concat(xc, yc, axis=-1)
        x_t_test = B.concat(xt, B.cast(xt.dtype, B.zeros(xt.shape[0], xt.shape[1], 1)), axis=-1)
        inp = B.concat(x_y_context, x_t_test, axis=1)
        mask, num_tar = self.create_mask(xc, yc, xt)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask)
        out = B.transpose(out[:, -num_tar:])

        return out


@_dispatch
def code(coder: Transformer, xz, z: B.Numeric, x, **kw_args):
    xz = coder(xz, z, x)
    return xz, z


@register_module
class RelationalMLP:
    """RelationalMLP.

    Args:
        in_dim (int): Input dimensionality.
        relational_out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_relational_enc_layers (int, optional): Number of relational hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        relational_out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_relational_enc_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity=None,
        dtype=None,
        comparison_function=None,
        relational_encoding_type="simple",
        non_equivariant_dim=None,
        sparse=False,
        k=50,
    ):
        # Check that one of the two specifications is given.
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
        self.sparse = sparse
        self.k = k

    def __call__(self, xc, yc, xt):
        _, set_size, _ = xc.shape
        _, target_set_size, _ = xt.shape
        xc = B.transpose(xc)
        yc = B.transpose(yc)
        xt = B.transpose(xt)

        encoding = self.comparison_function(self.relational_encoding_type,
                                            xc,
                                            yc,
                                            xt,
                                            self.sparse,
                                            self.k,
                                            self.non_equivariant_dim)

        batch_size, encoding_size, filter_size = encoding.shape
        encoding = encoding.view(batch_size * encoding_size, filter_size)

        x = self.net(encoding)
        x = x.view(batch_size, encoding_size, self.relational_out_dim)
        encoded_feature_dim = x.shape[-1]

        if self.relational_encoding_type == "simple":
            if self.sparse and set_size > self.k:
                set_size_new = self.k
            else:
                set_size_new = set_size
        elif self.relational_encoding_type == "full":
            if self.sparse and set_size > self.k:
                set_size_new = self.k * self.k
            else:
                set_size_new = set_size * set_size
        else:
            raise NotImplementedError

        x = x.view(batch_size, target_set_size, set_size_new, encoded_feature_dim)
        encoded_target_x = x.sum(dim=2)
        if self.non_equivariant_dim is not None:
            xt_non_equivariant = xt[:, :, self.non_equivariant_dim]
            encoded_target_x = B.concat(encoded_target_x, xt_non_equivariant, axis=-1)

        encoded_target_x = B.transpose(encoded_target_x)
        return encoded_target_x


@_dispatch
def code(coder: RelationalMLP, xz, z: B.Numeric, x, **kw_args):
    xz = coder(xz, z, x)
    return xz, z


# this was the 'select' option
# @_dispatch
# def code(coder: RelationalMLP, xz: Parallel, z: Parallel, x, **kw_args):
#    xz = xz[kw_args['select_channel']]
#    z = z[kw_args['select_channel']]
#    xz = coder(xz, z, x)
#    return xz, z


# this is the 'concat' option used in final experiments
# encode with all context sets, need to modify the input dimension of decoder
@_dispatch
def code(coder: RelationalMLP, xz: Parallel, z: Parallel, x, **kw_args):
    xz = [coder(xzi, zi, x) for (xzi, zi) in zip(xz, z)]
    return B.concat(*xz, axis=1), None


@register_module
class UNet:
    """UNet.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        kernels (int or tuple[int], optional): Sizes of the kernels. Defaults to `5`.
        strides (int or tuple[int], optional): Strides. Defaults to `2`.
        activations (object or tuple[object], optional): Activation functions.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make residual convolutional blocks. Defaults to
            `False`.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
        strides (tuple[int]): Strides.
        activations (tuple[function]): Activation functions.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_fields (list[float]): Receptive field for every intermediate value.
        receptive_field (float): Receptive field of the model.
        before_turn_layers (list[module]): Layers before the U-turn.
        after_turn_layers (list[module]): Layers after the U-turn
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        kernels: Union[int, Tuple[Union[int, Tuple[int, ...]], ...]] = 5,
        strides: Union[int, Tuple[int, ...]] = 2,
        activations: Union[None, object, Tuple[object, ...]] = None,
        separable: bool = False,
        residual: bool = False,
        resize_convs: bool = False,
        resize_conv_interp_method: str = "nearest",
        dtype=None,
    ):
        self.dim = dim

        # If `kernel` is an integer, repeat it for every layer.
        if not isinstance(kernels, (tuple, list)):
            kernels = (kernels,) * len(channels)
        elif len(kernels) != len(channels):
            raise ValueError(
                f"Length of `kernels` ({len(kernels)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.kernels = kernels

        # If `strides` is an integer, repeat it for every layer.
        # TODO: Change the default so that the first stride is 1.
        if not isinstance(strides, (tuple, list)):
            strides = (strides,) * len(channels)
        elif len(strides) != len(channels):
            raise ValueError(
                f"Length of `strides` ({len(strides)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.strides = strides

        # Default to ReLUs. Moreover, if `activations` is an activation function, repeat
        # it for every layer.
        activations = activations or self.nn.ReLU()
        if not isinstance(activations, (tuple, list)):
            activations = (activations,) * len(channels)
        elif len(activations) != len(channels):
            raise ValueError(
                f"Length of `activations` ({len(activations)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.activations = activations

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for stride, kernel in zip(self.strides, self.kernels):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if stride > 1:
                if after_conv % 2 == 0:
                    # If even, then subsample.
                    self.receptive_fields.append(after_conv // 2)
                else:
                    # If odd, then average pool.
                    self.receptive_fields.append((after_conv + 1) // 2)
            else:
                self.receptive_fields.append(after_conv)
        # Backward pass:
        for stride, kernel in zip(reversed(self.strides), reversed(self.kernels)):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            if stride > 1:
                after_interp = self.receptive_fields[-1] * 2 - 1
                self.receptive_fields.append(after_interp + (kernel - 1))
            else:
                self.receptive_fields.append(self.receptive_fields[-1] + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        # If none of the fancy features are used, use the standard `self.nn.Conv` for
        # compatibility with trained models. For the same reason we also don't use the
        #   `activation` keyword.
        # TODO: In the future, use `self.nps.Conv` everywhere and use the `activation`
        #   keyword.
        if residual or separable or any(isinstance(k, tuple) for k in kernels):
            Conv = partial(
                self.nps.Conv,
                dim=dim,
                residual=residual,
                separable=separable,
            )
        else:

            def Conv(*, stride=1, transposed=False, **kw_args):
                if transposed and stride > 1:
                    kw_args["output_padding"] = stride // 2
                return self.nn.Conv(
                    dim=dim,
                    stride=stride,
                    transposed=transposed,
                    **kw_args,
                )

        def construct_before_turn_layer(i):
            # Determine the configuration of the layer.
            ci = ((in_channels,) + tuple(channels))[i]
            co = channels[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    dtype=dtype,
                )
            else:
                # This is a downsampling layer.
                if self.receptive_fields[i] % 2 == 1:
                    # Perform average pooling if the previous receptive field is odd.
                    return self.nn.Sequential(
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            dtype=dtype,
                        ),
                        self.nn.AvgPool(
                            dim=dim,
                            kernel=s,
                            stride=s,
                            dtype=dtype,
                        ),
                    )
                else:
                    # Perform subsampling if the previous receptive field is even.
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        dtype=dtype,
                    )

        def construct_after_turn_layer(i):
            # Determine the configuration of the layer.
            if i == len(channels) - 1:
                # No skip connection yet.
                ci = channels[i]
            else:
                # Add the skip connection.
                ci = 2 * channels[i]
            co = ((channels[0],) + tuple(channels))[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel=k,
                    dtype=dtype,
                )
            else:
                # This is an upsampling layer.
                if resize_convs:
                    return self.nn.Sequential(
                        self.nn.UpSampling(
                            dim=dim,
                            size=s,
                            interp_method=resize_conv_interp_method,
                            dtype=dtype,
                        ),
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel=k,
                            stride=1,
                            dtype=dtype,
                        ),
                    )
                else:
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel=k,
                        stride=s,
                        transposed=True,
                        dtype=dtype,
                    )

        self.before_turn_layers = self.nn.ModuleList(
            [construct_before_turn_layer(i) for i in range(len(channels))]
        )
        self.after_turn_layers = self.nn.ModuleList(
            [construct_after_turn_layer(i) for i in range(len(channels))]
        )
        self.final_linear = self.nn.Conv(
            dim=dim,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
            dtype=dtype,
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        hs = [self.activations[0](self.before_turn_layers[0](x))]
        for layer, activation in zip(
            self.before_turn_layers[1:],
            self.activations[1:],
        ):
            hs.append(activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer, activation in zip(
            reversed(hs[:-1]),
            reversed(self.after_turn_layers[:-1]),
            reversed(self.activations[:-1]),
        ):
            h = activation(layer(B.concat(h_prev, h, axis=1)))

        return uncompress(self.final_linear(h))


@register_module
class ConvNet:
    """A regular convolutional neural network.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of channels at every intermediate layer.
        num_layers (int): Number of layers.
        points_per_unit (float, optional): Density of the discretisation corresponding
            to the inputs.
        receptive_field (float, optional): Desired receptive field.
        kernel (int, optional): Kernel size. If set, then this overrides the computation
            done by `points_per_unit` and `receptive_field`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults
            to `True`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_field (float): Receptive field.
        conv_net (module): The architecture.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        kernel: Optional[int] = None,
        points_per_unit: Optional[float] = 1,
        receptive_field: Optional[float] = None,
        separable: bool = True,
        residual: bool = False,
        dtype=None,
    ):
        self.dim = dim

        if kernel is None:
            # Compute kernel size.
            receptive_points = receptive_field * points_per_unit
            kernel = math.ceil(1 + (receptive_points - 1) / num_layers)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel  # Make kernel size odd.
            self.kernel = kernel  # Store it for reference.
        else:
            # Compute the receptive field size.
            receptive_points = kernel + num_layers * (kernel - 1)
            receptive_field = receptive_points / points_per_unit
            self.kernel = kernel

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0
        self.receptive_field = receptive_field

        # Construct basic building blocks.
        activation = self.nn.ReLU()

        self.conv_net = self.nn.Sequential(
            *(
                self.nps.Conv(
                    dim=dim,
                    in_channels=in_channels if first else channels,
                    out_channels=out_channels if last else channels,
                    kernel=kernel,
                    activation=None if first else activation,
                    separable=separable,
                    residual=residual,
                    dtype=dtype,
                )
                for first, last, _ in with_first_last(range(num_layers))
            )
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.conv_net(x))


@register_module
class Conv:
    """A flexible standard convolutional block.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int or tuple[int]): Kernel size(s). If it is a `tuple`, layers with
            those kernel sizes will be put in sequence.
        stride (int, optional): Stride.
        transposed (bool, optional): Transposed convolution. Defaults to `False`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make a residual block. Defaults to `False`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        net (object): Network.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, ...]],
        stride: int = 1,
        transposed: bool = False,
        activation=None,
        separable: bool = False,
        residual: bool = False,
        dtype=None,
    ):
        self.dim = dim

        if residual:
            self.net = self._init_residual(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                transposed=transposed,
                activation=activation,
                separable=separable,
                dtype=dtype,
            )
        else:
            if separable:
                self.net = self._init_separable_conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    dtype=dtype,
                )
            else:
                self.net = self._init_conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=1,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    dtype=dtype,
                )

    def _init_conv(
        self,
        dim,
        in_channels,
        out_channels,
        groups,
        kernel,
        stride,
        transposed,
        activation,
        dtype,
    ):
        intermediate_channels = min(in_channels, out_channels)

        # Determine the output padding.
        if transposed and stride > 1:
            if stride % 2 == 0:
                output_padding = {"output_padding": stride // 2}
            else:
                raise RuntimeError(
                    "Can only set the output padding correctly for `stride`s "
                    "which are a multiple of two."
                )
        else:
            output_padding = {}

        # Prepend the activation, if one is given.
        if activation:
            net = [activation]
        else:
            net = []

        # If `kernel` is a `tuple`, concatenate so many layers.
        net.extend(
            [
                self.nn.Conv(
                    dim=dim,
                    in_channels=in_channels if first else intermediate_channels,
                    out_channels=out_channels if last else intermediate_channels,
                    groups=groups,
                    kernel=k,
                    stride=stride if last else 1,
                    transposed=transposed if last else 1,
                    **(output_padding if last else {}),
                    dtype=dtype,
                )
                for first, last, k in with_first_last(convert(kernel, tuple))
            ]
        )

        return self.nn.Sequential(*net)

    def _init_separable_conv(
        self,
        dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        transposed,
        activation,
        dtype,
    ):
        return self.nn.Sequential(
            self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                kernel=kernel,
                stride=stride,
                transposed=transposed,
                activation=activation,
                dtype=dtype,
            ),
            self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                groups=1,
                kernel=1,
                stride=1,
                transposed=False,
                activation=None,
                dtype=dtype,
            ),
        )

    def _init_residual(
        self,
        dim,
        in_channels,
        out_channels,
        kernel,
        stride,
        transposed,
        activation,
        separable,
        dtype,
    ):
        intermediate_channels = min(in_channels, out_channels)
        if in_channels == intermediate_channels and stride == 1:
            # The input can be directly passed to the output.
            input_transform = lambda x: x
        else:
            # The input cannot be directly passed to the output, so we use an additional
            # linear layer.
            input_transform = self._init_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=intermediate_channels,
                groups=1,
                kernel=1,
                stride=stride,
                transposed=transposed,
                activation=None,
                dtype=dtype,
            )
        return self.nps.ResidualBlock(
            input_transform,
            self.nn.Sequential(
                self.nps.Conv(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel=kernel,
                    stride=stride,
                    transposed=transposed,
                    activation=activation,
                    separable=separable,
                    residual=False,
                    dtype=dtype,
                ),
                self.nn.ReLU(),
                self._init_conv(
                    dim=dim,
                    in_channels=intermediate_channels,
                    out_channels=intermediate_channels,
                    groups=1,
                    kernel=1,
                    stride=1,
                    transposed=False,
                    # TODO: Make this activation configurable.
                    activation=self.nn.ReLU(),
                    dtype=dtype,
                ),
            ),
            self._init_conv(
                dim=dim,
                in_channels=intermediate_channels,
                out_channels=out_channels,
                groups=1,
                kernel=1,
                stride=1,
                transposed=False,
                activation=None,
                dtype=dtype,
            ),
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.net(x))


@register_module
class ResidualBlock:
    """Block of a residual network.

    Args:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.

    Attributes:
        layer1 (object): Layer in the first branch.
        layer2 (object): Layer in the second branch.
        layer_post (object): Layer after adding the output of the two branches.
    """

    def __init__(self, layer1, layer2, layer_post):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer_post = layer_post

    def __call__(self, x):
        return self.layer_post(self.layer1(x) + self.layer2(x))
