"""
Fuses layer_norm((768,)) for bfloat16 models.

Pattern: matches exactly one node — torch.nn.functional.layer_norm(x,(768,),w,b,1e-05)
Returns a SINGLE tensor (the normed result = tmp_8).  This avoids the
multi-output assertion in custom_replacement._replace_pattern.

tmp_7 (pattern input x) stays in the graph unchanged, so the model's
original (tmp_7, tmp_10, tmp_9) return is still satisfied.
The two transpose(0,1) views remain free O(1) operations on the Triton output.

Routing: replacement_args appends "route_768" so _dispatch_ln_wrapper
dispatches to the 768-channel Triton kernel path.  Both pass files share
the SAME _dispatch_ln_wrapper object (imported from shared_ln_kernels),
which satisfies output_pass_replacement_func_limit=1.
"""

import torch
from pass_dir.shared_ln_kernels import _dispatch_ln_wrapper


def pattern(x, ln_weight, ln_bias):
    normed = torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-05)
    return normed


def replacement_args(x, ln_weight, ln_bias):
    return (x, ln_weight, ln_bias, "route_768")


def replacement_func():
    return _dispatch_ln_wrapper