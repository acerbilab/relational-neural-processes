import lab as B
import numpy as np
from plum import convert

from .. import _dispatch
from ..util import register_module
from ..parallel import Parallel

__all__ = [
    "RelationalEncode", "ConcatRelational"
]

@register_module
class RelationalEncode:
    """Replace x with xz and input into coder. Since our RNP needs to modify 'x', but original code function always
       treat 'x' unchanged. When encode xc we need to replace x with xz, and when encode xt we
       need to use inputscode to replace the original xz, z"""
    def __init__(self, coder, encode_input=False, output_index=None):
        self.coder = coder
        self.encode_input = encode_input
        self.output_index = output_index


@_dispatch
def code(coder: RelationalEncode, xz, z, x, **kw_args):
    """Now xz is what we are interested. In det_encoder part, xz is xc we want to encode, so coder is relational_encoder
       In RepeatForAggregateInputs part, xz is relational encoding for xt,
       we need to use InputsCoder to replace x with xt, now coder is InputsCoder"""

    if coder.encode_input:  # encode target inputs wrt context
        if coder.output_index is None:
            x, z = code(coder.coder, xz, z, x, **kw_args)
        else:
            x, z = code(coder.coder, xz[coder.output_index], z[coder.output_index], x, **kw_args)
        return x, x
    else:  # encode context wrt context
        xz, z = code(coder.coder, xz, z, xz, **kw_args)
        return xz, z


# todo: this was added to deal with enc_same=False. we can probably find a better solution
@register_module
class ConcatRelational:
    def __init__(self):
        pass

@_dispatch
def code(coder: ConcatRelational, xz: Parallel, z: Parallel, x, **kw_args):
    return B.concat(*xz, axis=1), B.concat(*z, axis=1)

