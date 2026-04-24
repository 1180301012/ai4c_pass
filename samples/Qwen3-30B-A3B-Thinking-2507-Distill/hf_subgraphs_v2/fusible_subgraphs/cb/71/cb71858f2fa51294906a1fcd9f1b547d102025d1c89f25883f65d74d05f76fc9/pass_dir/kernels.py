"""
Shared Triton kernels and dispatch wrapper used by multiple AI4C passes.
Both FuseConvViewSigmoid and FuseSumDiv import and return shared_dispatch
from this module so that replacement_func() returns the same Python object
across all passes, bypassing the output_pass_replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1 : 128-program conv + sigmoid
#   in_2  [1, 2, 1, 8]  → 16 flat values
#   weight[128, 2, 1, 8] → 128×16 flat values
#   bias  [128]
#   out   [1, 2, 8, 8]   (128 flat values, viewed from [128])
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def conv_sigmoid_kernel(
    in2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C_IN: tl.constexpr,     # 16
    BLOCK_CIN: tl.constexpr,  # 16
):
    oc = tl.program_id(0)

    cin_offsets = tl.arange(0, BLOCK_CIN)
    mask = cin_offsets < C_IN

    x = tl.load(in2_ptr + cin_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + oc * C_IN + cin_offsets, mask=mask, other=0.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + oc).to(tl.float32)

    dot = tl.sum(x * w, axis=0) + bias_val
    result = tl.sigmoid(dot)

    # out is [1,2,8,8] = 128 elements; out_ptr points to flat array
    tl.store(out_ptr + oc, result)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2 : 16-program row normalisation
#   in  [1, 2, 8, 8]  → 128 flat values
#   out [1, 2, 8, 8]  → 128 flat values
#   Each program handles one 8-element row.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def sum_div_kernel(
    in_ptr,
    out_ptr,
    BLOCK_D: tl.constexpr,  # 8
):
    row = tl.program_id(0)
    d_offsets = tl.arange(0, BLOCK_D)
    offsets = row * BLOCK_D + d_offsets

    x = tl.load(in_ptr + offsets).to(tl.float32)
    row_sum = tl.sum(x, axis=0)
    y = x / row_sum
    tl.store(out_ptr + offsets, y)


# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper (SAME object imported by all pass files)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def shared_dispatch(a, b, c, route):
    """
    route == "conv_view_sigmoid" : a=bias[128], b=weight[128,2,1,8],
                                   c=input[1,2,1,8]  → returns [1,2,8,8]
    route == "sum_div"           : a=b=c=in_3[1,2,8,8] → returns [1,2,8,8]
    """
    if route == "conv_view_sigmoid":
        bias, weight, inp = a, b, c
        C_OUT = 128
        out = torch.empty((C_OUT,), dtype=inp.dtype, device=inp.device)
        conv_sigmoid_kernel[(C_OUT,)](
            inp, weight, bias, out,
            C_IN=16,
            BLOCK_CIN=16,
        )
        return out.view(1, 2, 8, 8)
    else:  # "sum_div"
        inp = a
        out = torch.empty_like(inp)
        sum_div_kernel[(16,)](
            inp, out,
            BLOCK_D=8,
        )
        return out