import torch
import triton
import triton.language as tl

from pass_dir.shared_kernel import fused_conv_post


# ── Pattern: returns ONLY tmp_8 (single output) ──────────────────────────────
# The slice + gelu + transpose + add subgraph is matched; layer_norm stays in
# the graph untouched (it uses tmp_8 as input, which is replaced by the kernel).
# This single-output design avoids the assert len(returning_nodes)==len(copied)
# that the framework raises for multi-output patterns with the dispatch wrapper.
def pattern(x_conv, in3):
    gelu_x = torch.nn.functional.gelu(x_conv[slice(None, None, None), slice(None, None, None), slice(None, -1, None)])
    t      = gelu_x.transpose(1, 2)
    s      = in3 + t
    return s          # returns tmp_8  (dropout is outside the matched subgraph)


def replacement_args(x_conv, in3):
    return (x_conv, in3, 'route_sum')   # pass route tag for replacement_func_limit


def replacement_func():
    return fused_conv_post