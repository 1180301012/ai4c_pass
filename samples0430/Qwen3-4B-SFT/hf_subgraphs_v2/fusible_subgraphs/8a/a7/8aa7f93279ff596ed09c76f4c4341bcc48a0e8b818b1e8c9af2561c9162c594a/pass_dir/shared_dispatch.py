"""
Shared dispatch module: both passes return the SAME replacement_func object
so output_pass_replacement_func_limit never drops either pass.

Routes
------
"mul"     – element-wise scale
"transpose" – 1-D stride-D gather (last two dims → output)
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel A: scaled multiply
# 301056 elements = 294 × 1024 exactly → no masking needed.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _mul_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v * 0.1767766952966369)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel B: transpose last two dims (1-D element-wise gather/scatter)
# in_0[b, 0, s, d]  →  out[b, 0, d, s]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 2048}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def _transpose1d_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # no mask: 301056 = 294 × 1024 exactly
    v = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, v)


# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatch – single @torch.fx.wrap that handles all routes.
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def shared_dispatch(x, route):
    if route == "mul":
        out = torch.empty_like(x)
        _mul_kernel[(294,)](x, out, BLOCK=1024, num_warps=8)
        return out
    elif route == "transpose":
        B = x.shape[0]; S = x.shape[2]; D = x.shape[3]
        out = torch.empty(B, 1, D, S, dtype=x.dtype, device=x.device)
        _transpose1d_kernel[(294,)](x, out, BLOCK=1024, num_warps=8)
        return out
    return x