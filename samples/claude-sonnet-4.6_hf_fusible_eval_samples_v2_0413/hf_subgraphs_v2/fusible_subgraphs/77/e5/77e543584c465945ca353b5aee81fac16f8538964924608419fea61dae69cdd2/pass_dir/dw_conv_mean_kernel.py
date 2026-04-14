"""
Diagnostic version: replacement uses only torch.empty (no Triton) to test
whether the framework multi-output replacement mechanism works at all.
If this doesn't crash, the bug is in the Triton kernel code.
If this still crashes, the bug is in the framework's multi-output handling.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256},  num_warps=4),
        triton.Config({"BLOCK_HW": 512},  num_warps=4),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8),
    ],
    key=["HW_out"],
)
@triton.jit
def _dw_conv3x3_mean_kernel(
    inp_ptr,    # [N, C, H_in, W_in]  contiguous
    wt_ptr,     # [C, 9]              contiguous
    out_ptr,    # [N, C, H_out, W_out] output
    mean_ptr,   # [N*C]               output
    C,
    H_in, W_in,
    H_out, W_out, HW_out,
    sh, sw,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c   = pid % C
    base_in  = pid * H_in  * W_in
    base_out = pid * H_out * W_out
    acc = 0.0

    for s in range(0, HW_out, BLOCK_HW):
        idx  = s + tl.arange(0, BLOCK_HW)
        mask = idx < HW_out
        oh   = idx // W_out
        ow   = idx % W_out
        v = tl.zeros([BLOCK_HW], dtype=tl.float32)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih   = oh * sh + kh - 1
                iw   = ow * sw + kw - 1
                bnd  = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
                msk  = mask & bnd
                ih_s = tl.where(bnd, ih, 0)
                iw_s = tl.where(bnd, iw, 0)
                x    = tl.load(inp_ptr + base_in + ih_s * W_in + iw_s,
                                mask=msk, other=0.0).to(tl.float32)
                w    = tl.load(wt_ptr + c * 9 + kh * 3 + kw).to(tl.float32)
                v    = v + x * w
        if DTYPE == 0:
            tl.store(out_ptr + base_out + idx, v.to(tl.float16),  mask=mask)
        elif DTYPE == 1:
            tl.store(out_ptr + base_out + idx, v.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out_ptr + base_out + idx, v.to(tl.float32),  mask=mask)
        acc = acc + tl.sum(tl.where(mask, v, 0.0))

    mean_val = acc / HW_out
    if DTYPE == 0:
        tl.store(mean_ptr + pid, mean_val.to(tl.float16))
    elif DTYPE == 1:
        tl.store(mean_ptr + pid, mean_val.to(tl.bfloat16))
    else:
        tl.store(mean_ptr + pid, mean_val.to(tl.float32))


def _get_dtype_code(t):
    if t.dtype == torch.float16:   return 0
    if t.dtype == torch.bfloat16:  return 1
    return 2


# ---------------------------------------------------------------------------
# Single @torch.fx.wrap dispatch wrapper shared by ALL 5 pass files.
# Diagnostic: first try without Triton to see if multi-output FX works.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dw_conv_mean(in_0, in_1, route):
    N, C, H_in, W_in = in_1.shape
    if route == "s1":
        sh, sw = 1, 1
    else:
        sh, sw = 2, 2

    H_out  = (H_in + 2 - 3) // sh + 1
    W_out  = (W_in + 2 - 3) // sw + 1

    out      = torch.empty((N, C, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty((N, C, 1, 1),         dtype=in_1.dtype, device=in_1.device)
    return out, mean_out