import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: simple 1D reduction (one program per channel pair, simple code)
# Good for small N where maximum program count is beneficial.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _avgpool_1d_kernel(
    x_ptr, out_ptr,
    C, HW, stride_n, stride_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n   = pid // C
    c   = pid % C
    base = n * stride_n + c * stride_c
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    x    = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    total = tl.sum(x.to(tl.float32), axis=0)
    tl.store(out_ptr + pid, total / HW)


# ---------------------------------------------------------------------------
# Kernel B: 2D channel-batched reduction (one program per N*C_tiles groups)
# Good for large N where fewer, heavier programs give better occupancy.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 8,  'BLOCK_HW': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=1),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def _avgpool_2d_kernel(
    x_ptr, out_ptr,
    N, C, HW, stride_n, stride_c,
    BLOCK_C:  tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid     = tl.program_id(0)
    num_ct  = tl.cdiv(C, BLOCK_C)
    n       = pid // num_ct
    c_start = (pid % num_ct) * BLOCK_C

    offs_c  = tl.arange(0, BLOCK_C)
    offs_hw = tl.arange(0, BLOCK_HW)
    mask_c  = (c_start + offs_c) < C
    mask_hw = offs_hw < HW
    mask    = mask_c[:, None] & mask_hw[None, :]

    c_offs = (c_start + offs_c) * stride_c
    x_ptrs = x_ptr + n * stride_n + c_offs[:, None] + offs_hw[None, :]

    x      = tl.load(x_ptrs, mask=mask, other=0.0)
    totals = tl.sum(x.to(tl.float32), axis=1)
    result = totals / HW

    out_offs = n * C + c_start + offs_c
    tl.store(out_ptr + out_offs, result, mask=mask_c)


@torch.fx.wrap
def avgpool_flatten_fast(x):
    N, C, H, W = x.shape
    HW  = H * W
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)

    if N >= 16:
        # Small+medium batch: 2D kernel (fewer programs, better occupancy)
        def grid(meta):
            return (N * triton.cdiv(C, meta['BLOCK_C']),)
        _avgpool_2d_kernel[grid](x, out, N, C, HW, C * HW, HW)
    else:
        # Tiny batch (N=1,2,4,8): 1D kernel (simplest, lowest overhead)
        _avgpool_1d_kernel[(N * C,)](x, out, C, HW, C * HW, HW)
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return avgpool_flatten_fast