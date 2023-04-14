import lab as B
import numpy as np
from plum import convert

from .. import _dispatch
from ..util import register_module

__all__ = [
    "Context_relational_encode",
    "Test_relational_encode"
]

@register_module
class Context_relational_encode:
    def __init__(self, relational_encoder):
        self.relational_encoder = relational_encoder


@_dispatch
def code(coder: Context_relational_encode, xz, z, x, **kw_args):
    xz, z = code(coder.relational_encoder, xz, z, xz, **kw_args)
    return xz, z


@register_module
class Test_relational_encode:
    def __init__(self, inputs_encoder):
        self.inputs_encoder = inputs_encoder


@_dispatch
def code(coder: Test_relational_encode, xz, z, x, **kw_args):
    xz, z = code(coder.inputs_encoder, xz, z, xz, **kw_args)
    return xz, z