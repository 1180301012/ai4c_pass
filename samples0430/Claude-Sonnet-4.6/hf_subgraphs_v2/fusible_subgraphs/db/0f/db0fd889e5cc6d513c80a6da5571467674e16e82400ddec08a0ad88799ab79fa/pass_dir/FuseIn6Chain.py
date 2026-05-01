"""
Pass: FuseIn6Chain
Match the in_6 position-embedding processing chain (WITHOUT interpolate,
since same-size bicubic is constant-folded/eliminated by torch.compile):
  x.transpose(2,3) → view(4,32,15,15) → flatten(2) → transpose(1,2)
  → contiguous() → view(4,1,225,32)

The chain is mathematically identity; replace with a single
strided-copy Triton kernel to reduce kernel-launch count.
Input x: [4, 1, 225, 32]   Output: [4, 1, 225, 32]
"""

import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x):
    t = x.transpose(2, 3)
    v = torch.ops.aten.view.default(t, [4, 32, 15, 15])
    i = torch._C._nn.upsample_bicubic2d(v, [15, 15], False, None)
    f = i.flatten(2)
    t2 = f.transpose(1, 2)
    c = t2.contiguous()
    o = torch.ops.aten.view.default(c, [4, 1, 225, 32])
    return o


def replacement_args(x):
    return (x, x, "in6_chain")   # b=x is dummy for shared-dispatch arity


def replacement_func():
    return dispatch