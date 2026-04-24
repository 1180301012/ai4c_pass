import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _add3_mean_kernel(
    a_ptr, b_ptr, c_ptr,
    out_ptr, mean_ptr,
    HW, BC,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = a+b+c, mean = spatial mean of out."""
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    base    = pid_bc * HW
    start   = pid_hw * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = (start + offsets) < HW

    a = tl.load(a_ptr + base + start + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + base + start + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + base + start + offsets, mask=mask, other=0.0)

    x   = a + b + c
    acc = tl.where(mask, x, 0.0).to(tl.float32)

    tl.store(out_ptr + base + start + offsets, x, mask=mask)

    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(mean_ptr + pid_bc, mean_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _add2_mean_kernel(
    a_ptr, b_ptr,
    out_ptr, mean_ptr,
    HW, BC,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = a+b, mean = spatial mean of out."""
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    base    = pid_bc * HW
    start   = pid_hw * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = (start + offsets) < HW

    a = tl.load(a_ptr + base + start + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + base + start + offsets, mask=mask, other=0.0)

    x   = a + b
    acc = tl.where(mask, x, 0.0).to(tl.float32)

    tl.store(out_ptr + base + start + offsets, x, mask=mask)

    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(mean_ptr + pid_bc, mean_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _add_scalar0_mean_kernel(
    a_ptr,
    out_ptr, mean_ptr,
    HW, BC,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = 0+a+0=a, mean = spatial mean of a."""
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    base    = pid_bc * HW
    start   = pid_hw * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = (start + offsets) < HW

    a   = tl.load(a_ptr + base + start + offsets, mask=mask, other=0.0)
    acc = tl.where(mask, a, 0.0).to(tl.float32)

    tl.store(out_ptr + base + start + offsets, a, mask=mask)

    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(mean_ptr + pid_bc, mean_val)


@torch.fx.wrap
def add_mean_dispatch(a, b, c, route, out_dtype):
    """
    Dispatch wrapper for fused add+mean kernels.
      route="add3"   -> out = a+b+c
      route="add2"   -> out = a+b  (c is unused)
      route="add0"   -> out = a    (b, c are unused zeros)
    Returns (out, mean) with mean shape [B, C, 1, 1].
    """
    B, C, H, W = a.shape
    HW = H * W
    BC = B * C

    out      = torch.empty_like(a)
    mean_out = torch.empty((B, C, 1, 1), dtype=out_dtype, device=a.device)
    grid     = lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_SIZE']))

    if route == "add3":
        _add3_mean_kernel[grid](a, b, c, out, mean_out, HW, BC)
    elif route == "add2":
        _add2_mean_kernel[grid](a, b, out, mean_out, HW, BC)
    elif route == "add0":
        _add_scalar0_mean_kernel[grid](a, out, mean_out, HW, BC)

    return out, mean_out


# ── Mean-only wrapper (used by FuseMeanDim23 pass) ─────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _mean_dim23_kernel(
    x_ptr, mean_ptr,
    HW, BC,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    base    = pid_bc * HW
    start   = pid_hw * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = (start + offsets) < HW
    x    = tl.load(x_ptr + base + start + offsets, mask=mask, other=0.0)
    acc  = tl.where(mask, x, 0.0).to(tl.float32)
    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(mean_ptr + pid_bc, mean_val)


@torch.fx.wrap
def triton_mean_dim23(x):
    """Fast Triton spatial mean over dims (2, 3), keepdim=True. Input: [B,C,H,W]. Output: [B,C,1,1]."""
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C
    mean_out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    grid     = lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_SIZE']))
    _mean_dim23_kernel[grid](x, mean_out, HW, BC)
    return mean_out