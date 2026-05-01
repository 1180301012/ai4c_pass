import torch
import triton
import triton.language as tl


def pattern(x):
    t = x.view(1, 2, 8, 8)
    out = t.sigmoid()
    return out


def replacement_args(x):
    return (x, "view_sigmoid")


# ── kernel: view + sigmoid (cast to fp32 for tl.sigmoid, cast back) ──────────
@triton.jit
def _view_sigmoid_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    # tl.sigmoid only supports fp32/fp64 – upcast, compute, downcast
    x_f32 = x.to(tl.float32)
    out_f32 = tl.sigmoid(x_f32)
    # Store back (Triton coerces fp32 → bf16/fp16 to match out_ptr dtype)
    tl.store(out_ptr + offs, out_f32, mask=mask)


# ── kernel: reduce-sum then divide (last dim, keepdim) ────────────────────────
@triton.jit
def _sum_div_kernel(
    x_ptr,
    out_ptr,
    ROW_LEN: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * ROW_LEN
    offs = row_start + tl.arange(0, ROW_LEN)
    x = tl.load(x_ptr + offs)
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)
    out_f32 = x_f32 / row_sum
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offs, out)


# ── shared dispatch wrapper ───────────────────────────────────────────────────
@torch.fx.wrap
def _dispatch(x, route: str):
    if route == "view_sigmoid":
        # x: conv2d output [1,128,1,1] → sigmoid → [1,2,8,8]
        out = torch.empty((1, 2, 8, 8), dtype=x.dtype, device=x.device)
        _view_sigmoid_kernel[(1,)](x, out, N=128, BLOCK=128)
        return out
    elif route == "sum_div":
        # x: [1,2,8,8] → normalise last dim
        out = torch.empty_like(x)
        _sum_div_kernel[(16,)](x, out, ROW_LEN=8)
        return out
    # unreachable
    return x


def replacement_func():
    return _dispatch