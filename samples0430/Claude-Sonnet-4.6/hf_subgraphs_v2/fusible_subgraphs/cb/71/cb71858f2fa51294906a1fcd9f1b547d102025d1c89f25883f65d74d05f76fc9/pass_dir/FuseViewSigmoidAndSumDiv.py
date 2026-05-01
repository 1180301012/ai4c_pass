import torch
import triton
import triton.language as tl


# ─── Pattern: both independent chains in one subgraph match ───────────────────
def pattern(x_sigmoid, x_div):
    # Chain 1 – conv output reshape + sigmoid
    t = x_sigmoid.view(1, 2, 8, 8)
    tmp_4 = t.sigmoid()
    # Chain 2 – row-normalise (sum over last dim then divide)
    s = x_div.sum(dim=3, keepdim=True)
    tmp_6 = x_div / s
    return tmp_4, tmp_6


def replacement_args(x_sigmoid, x_div):
    return (x_sigmoid, x_div)


# ─── Triton kernel 1: element-wise sigmoid (cast fp32, write back) ────────────
@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.sigmoid(x.to(tl.float32))
    tl.store(out_ptr + offs, out, mask=mask)


# ─── Triton kernel 2: fused row-sum-divide ────────────────────────────────────
@triton.jit
def _row_sum_div_kernel(x_ptr, out_ptr, ROW_LEN: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * ROW_LEN
    offs = base + tl.arange(0, ROW_LEN)
    x = tl.load(x_ptr + offs)
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)
    out = (x_f32 / row_sum).to(x.dtype)
    tl.store(out_ptr + offs, out)


# ─── Single @torch.fx.wrap boundary for BOTH kernels ─────────────────────────
@torch.fx.wrap
def fused_sigmoid_and_row_norm(x_sigmoid, x_div):
    # x_sigmoid : conv2d output  [1, 128, 1, 1]  → out_sigmoid [1, 2, 8, 8]
    # x_div     : in_3           [1, 2, 8, 8]    → out_div     [1, 2, 8, 8]
    out_sigmoid = torch.empty((1, 2, 8, 8), dtype=x_sigmoid.dtype, device=x_sigmoid.device)
    out_div = torch.empty_like(x_div)
    _sigmoid_kernel[(1,)](x_sigmoid, out_sigmoid, N=128, BLOCK=128)
    _row_sum_div_kernel[(16,)](x_div, out_div, ROW_LEN=8)
    return out_sigmoid, out_div


def replacement_func():
    return fused_sigmoid_and_row_norm