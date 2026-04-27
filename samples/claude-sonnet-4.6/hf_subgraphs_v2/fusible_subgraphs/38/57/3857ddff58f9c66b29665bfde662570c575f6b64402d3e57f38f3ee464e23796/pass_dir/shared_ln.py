# Shared Triton layer-norm kernel and dispatch helpers used by both passes.
import torch
import triton
import triton.language as tl


# ── single kernel handles any N via constexpr ─────────────────────────────

@triton.jit
def _ln_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx    = tl.program_id(0)
    row_offset = row_idx * N
    cols       = tl.arange(0, BLOCK_SIZE)
    mask       = cols < N

    # Load row (invalid lanes get 0.0 → don't affect mean)
    x     = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Mean: invalid lanes are 0 so sum is correct
    mean = tl.sum(x_f32, axis=0) / N

    # Variance: zero out invalid lanes before squaring
    diff = x_f32 - mean
    diff = tl.where(mask, diff, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    # Affine transform
    weight   = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)

    out = x_norm * weight + bias_val
    tl.store(out_ptr + row_offset + cols, out.to(x.dtype), mask=mask)


# ── per-size Python wrappers ──────────────────────────────────────────────

def _do_ln_768(bias, weight, x):
    N, BS = 768, 1024
    num_rows = x.numel() // N
    out = torch.empty(num_rows, N, dtype=x.dtype, device=x.device)
    _ln_kernel[(num_rows,)](x, weight, bias, out,
                            N=N, eps=1e-05, BLOCK_SIZE=BS, num_warps=4)
    return out


def _do_ln_16(bias, weight, x):
    N, BS = 16, 16
    num_rows = x.numel() // N
    out = torch.empty(num_rows, N, dtype=x.dtype, device=x.device)
    _ln_kernel[(num_rows,)](x, weight, bias, out,
                            N=N, eps=1e-05, BLOCK_SIZE=BS, num_warps=1)
    return out


# ── shared dispatch (ONE object imported by both pass files) ──────────────

@torch.fx.wrap
def fused_dispatch(in_0, in_1, x, route):
    """Route to the correct layer-norm kernel based on the route tag.
    in_0 = bias, in_1 = weight, x = input tensor (tmp_3).
    Returns single normalised output tensor (tmp_4).
    """
    if route == "route_768":
        return _do_ln_768(in_0, in_1, x)
    elif route == "route_16":
        return _do_ln_16(in_0, in_1, x)