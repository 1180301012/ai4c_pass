"""
Pass: FuseIn5Chain
Match the in_5 position-embedding processing chain (WITHOUT interpolate,
since same-size bicubic is constant-folded/eliminated by torch.compile):
  x.transpose(1,2) → view(1,32,15,15) → flatten(2) → transpose(1,2)

The chain is mathematically identity; replace it with a single
strided-copy Triton kernel to reduce kernel-launch count.
Input x: [1, 225, 32]   Output: [1, 225, 32]
"""

import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x):
    t = x.transpose(1, 2)
    v = torch.ops.aten.view.default(t, [1, 32, 15, 15])
    i = torch._C._nn.upsample_bicubic2d(v, [15, 15], False, None)
    f = i.flatten(2)
    o = f.transpose(1, 2)
    return o


def replacement_args(x):
    return (x, x, "in5_chain")   # b=x is dummy for shared-dispatch arity


def replacement_func():
    return dispatch