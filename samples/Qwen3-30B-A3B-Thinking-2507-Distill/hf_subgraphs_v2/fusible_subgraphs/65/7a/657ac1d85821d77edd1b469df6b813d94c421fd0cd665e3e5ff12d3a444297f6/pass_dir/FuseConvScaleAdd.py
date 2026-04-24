"""
Pass 1: fuse dropout(p=0,training=False) -> layer-scale-mul -> residual-add
into a single Triton kernel.  Dropout is identity at p=0,training=False.

Pattern has 1 output (tmp_10) so it avoids the 2-output assertion crash.
The batch_norm node remains in the graph and consumes tmp_10 as its input.

replacement_args maps:
  conv_out  -> first placeholder (matched to tmp_7 / conv2d output)
  gamma     -> second placeholder (matched to in_0)
  residual  -> third placeholder (matched to in_7)
  a3 dummy  -> gamma again (passed to satisfy dispatch_wrapper's 5-arg sig)
"""
import torch
from pass_dir._kernels import dispatch_wrapper


def pattern(conv_out, gamma, residual):
    """
    conv_out = tmp_7   (conv2d output, used only within this subgraph)
    gamma    = in_0    (layer-scale weight [64,1,1])
    residual = in_7   ([B,64,H,W])
    """
    drop_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scaled   = drop_out * gamma
    result   = residual + scaled
    return result   # 1 output only


def replacement_args(conv_out, gamma, residual):
    # dispatch_wrapper(a0=conv_out, a1=gamma, a2=residual, a3=gamma, a4=gamma, route="scale_add")
    return (conv_out, gamma, residual, gamma, gamma, "scale_add")


def replacement_func():
    return dispatch_wrapper