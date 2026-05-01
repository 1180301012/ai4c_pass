"""
Fuses layer_norm((1024,)) for float32 models.

Pattern: matches exactly one node — torch.nn.functional.layer_norm(x,(1024,),w,b,1e-05)
Returns a SINGLE tensor (the normed result = tmp_8).

Routing: replacement_args appends "route_1024" so _dispatch_ln_wrapper
dispatches to the 1024-channel Triton kernel path.  The same _dispatch_ln_wrapper
object is shared with FuseResidualFlattenLN_768, satisfying
output_pass_replacement_func_limit=1.
"""

import torch
from pass_dir.shared_ln_kernels import _dispatch_ln_wrapper


def pattern(x, ln_weight, ln_bias):
    normed = torch.nn.functional.layer_norm(x, (1024,), ln_weight, ln_bias, 1e-05)
    return normed


def replacement_args(x, ln_weight, ln_bias):
    return (x, ln_weight, ln_bias, "route_1024")


def replacement_func():
    return _dispatch_ln_wrapper