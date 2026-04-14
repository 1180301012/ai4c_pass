import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel A: weighted coord sums (route="weighted")
#   sm4d [B*C, 4096], in_0 [64], in_1 [64] → out [B*C, 2]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['N_ROWS'],
)
@triton.jit
def weighted_sum_kernel(
    sm4d_ptr, in0_ptr, in1_ptr, out_ptr,
    N_ROWS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    sm     = tl.load(sm4d_ptr + base + offsets)
    sm_f32 = sm.to(tl.float32)

    x_coords = tl.load(in0_ptr + (offsets % 64)).to(tl.float32)
    y_coords = tl.load(in1_ptr + (offsets // 64)).to(tl.float32)

    wx = tl.sum(sm_f32 * x_coords, axis=0)
    wy = tl.sum(sm_f32 * y_coords, axis=0)

    tl.store(out_ptr + pid * 2,     wx.to(sm.dtype))
    tl.store(out_ptr + pid * 2 + 1, wy.to(sm.dtype))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel B: double-sum + cat fallback (route="double_sum")
#   a [B*C, 4096], b [B*C, 4096] → out [B*C, 2]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['N_ROWS'],
)
@triton.jit
def double_sum_cat_kernel(
    a_ptr, b_ptr, out_ptr,
    N_ROWS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    raw_a = tl.load(a_ptr + base + offsets)
    raw_b = tl.load(b_ptr + base + offsets)

    sum_a = tl.sum(raw_a.to(tl.float32), axis=0)
    sum_b = tl.sum(raw_b.to(tl.float32), axis=0)

    tl.store(out_ptr + pid * 2,     sum_a.to(raw_a.dtype))
    tl.store(out_ptr + pid * 2 + 1, sum_b.to(raw_a.dtype))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (not FX-wrapped; called only from dispatch_fused)
# ─────────────────────────────────────────────────────────────────────────────
def _weighted_sum_impl(in_0, in_1, sm4d):
    B, C = sm4d.shape[0], sm4d.shape[1]
    N_ROWS = B * C
    out = torch.empty(B, C, 2, dtype=sm4d.dtype, device=sm4d.device)
    weighted_sum_kernel[(N_ROWS,)](
        sm4d_ptr=sm4d, in0_ptr=in_0, in1_ptr=in_1,
        out_ptr=out, N_ROWS=N_ROWS, BLOCK_SIZE=4096,
    )
    return out


def _double_sum_impl(tmp_5, tmp_8):
    B, C = tmp_5.shape[0], tmp_5.shape[1]
    N_ROWS = B * C
    out = torch.empty(B, C, 2, dtype=tmp_5.dtype, device=tmp_5.device)
    double_sum_cat_kernel[(N_ROWS,)](
        a_ptr=tmp_5, b_ptr=tmp_8, out_ptr=out,
        N_ROWS=N_ROWS, BLOCK_SIZE=4096,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Single shared dispatch wrapper (ALL passes return this from replacement_func)
#
#   Route "weighted":    dispatch_fused(in_0, in_1, sm4d, "weighted")
#   Route "double_sum":  dispatch_fused(tmp_5, tmp_8, "double_sum")
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def dispatch_fused(a, b, c_or_route, route_or_none=None):
    if route_or_none is not None:
        # weighted: a=in_0, b=in_1, c_or_route=sm4d, route_or_none="weighted"
        return _weighted_sum_impl(a, b, c_or_route)
    else:
        # double_sum: a=tmp_5, b=tmp_8, c_or_route="double_sum"
        return _double_sum_impl(a, b)


# Keep legacy wrappers for any direct callers
@torch.fx.wrap
def fused_weighted_sum(in_0, in_1, sm4d):
    return _weighted_sum_impl(in_0, in_1, sm4d)


@torch.fx.wrap
def fused_double_sum_cat(tmp_5, tmp_8):
    return _double_sum_impl(tmp_5, tmp_8)