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
    """Encode context or target inputs with a coder that takes as input context inputs,
    context outputs, and the inputs to be encoded."""
    def __init__(self, coder, encode_target=False, out_index=None):
        self.coder = coder
        self.encode_target = encode_target
        self.out_index = out_index


@_dispatch
def code(coder: RelationalEncode, xz, z, x, **kw_args):
    if coder.encode_target:
        # in encode target mode, encode target inputs x in the context xz and z
        if coder.out_index is None:
            # encode wrt all contexts
            x, z = code(coder.coder, xz, z, x, **kw_args)
        else:
            # encode wrt selected context
            x, z = code(coder.coder, xz[coder.out_index], z[coder.out_index], x, **kw_args)
        # return target inputs
        return x, x
    else:
        # otherwise, encode context inputs xz in the context xz and z
        xz, z = code(coder.coder, xz, z, xz, **kw_args)
        # return context
        return xz, z


# todo: this was added to deal with enc_same=False. we can probably find a better solution
@register_module
class ConcatRelational:
    def __init__(self):
        pass

@_dispatch
def code(coder: ConcatRelational, xz: Parallel, z: Parallel, x, **kw_args):
    return B.concat(*xz, axis=1), B.concat(*z, axis=1)

