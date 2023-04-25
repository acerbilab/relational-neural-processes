import lab as B
import numpy as np
from plum import convert

from .. import _dispatch
from ..util import register_module

__all__ = [
    "RelationalEncode"
]

@register_module
class RelationalEncode:
    """Replace x with xz and input into coder. Since our RNP needs to modify 'x', but original code function always
       treat 'x' unchanged. When encode xc we need to replace x with xz, and when encode xt we
       need to use inputscode to replace the original xz, z"""
    def __init__(self, coder):
        self.coder = coder


@_dispatch
def code(coder: RelationalEncode, xz, z, x, **kw_args):
    """Now xz is what we are interested. In det_encoder part, xz is xc we want to encode, so coder is relational_encoder
       In RepeatForAggregateInputs part, xz is relational encoding for xt,
       we need to use InputsCoder to replace x with xt, now coder is InputsCoder"""
    xz, z = code(coder.coder, xz, z, xz, **kw_args)
    return xz, z

