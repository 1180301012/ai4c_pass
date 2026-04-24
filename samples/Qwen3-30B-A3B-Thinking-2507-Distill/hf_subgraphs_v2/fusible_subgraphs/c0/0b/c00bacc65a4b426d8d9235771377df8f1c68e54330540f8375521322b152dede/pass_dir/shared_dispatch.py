"""
Shared kernels and unified dispatch wrapper for all passes.
Both FuseBNInference.py and FuseAvgPool2x2Stride2.py import `dispatch` from
here so the framework sees exactly ONE unique replacement_func.
"""
import torch
import triton
import triton.language as tl


# =========================================================================
# Batch-Norm inference kernel  (2-D grid: pid0=N, pid1=C)
# -----------------------------------------------------------------------
# One Triton program per (n, c) slice — processes all HW elements.
# BN params are loaded as scalars once per program (no gather).
# Contiguous HW elements per warp → fully coalesced memory access.
# =========================================================================
@triton.jit
def _bn_inf_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    n   = tl.program_id(0)
    c   = tl.program_id(1)

    # Scalar BN-parameter loads (broadcast over the HW block)
    mu  = tl.load(mean_ptr + c).to(tl.float32)
    v   = tl.load(var_ptr  + c).to(tl.float32)
    wt  = tl.load(w_ptr    + c).to(tl.float32)
    bt  = tl.load(b_ptr    + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(v + eps)
    scale   = wt * inv_std
    shift   = bt - mu * scale

    base = (n * C + c) * HW
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    y = x.to(tl.float32) * scale + shift
    tl.store(out_ptr + base + offs, y.to(x.dtype), mask=mask)


# =========================================================================
# 2×2 average-pool, stride 2  (1-D flat grid)
# =========================================================================
@triton.jit
def _avg_pool2x2_s2_kernel(
    x_ptr,
    out_ptr,
    N, C, OH, OW,
    H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask  = offs < total

    ow  = offs % OW
    oh  = (offs // OW) % OH
    c   = (offs // (OW * OH)) % C
    n   = offs // (OW * OH * C)

    base_nc = n * C * H * W + c * H * W
    x00 = tl.load(x_ptr + base_nc + oh * 2 * W + ow * 2,       mask=mask, other=0.0)
    x01 = tl.load(x_ptr + base_nc + oh * 2 * W + ow * 2 + 1,   mask=mask, other=0.0)
    x10 = tl.load(x_ptr + base_nc + (oh * 2 + 1) * W + ow * 2, mask=mask, other=0.0)
    x11 = tl.load(x_ptr + base_nc + (oh * 2 + 1) * W + ow * 2 + 1, mask=mask, other=0.0)

    avg = (x00.to(tl.float32) + x01.to(tl.float32) + x10.to(tl.float32) + x11.to(tl.float32)) * 0.25
    tl.store(out_ptr + offs, avg.to(x00.dtype), mask=mask)


# =========================================================================
# Unified dispatch  (returned by replacement_func() in ALL passes)
# =========================================================================
@torch.fx.wrap
def dispatch(*args):
    route = args[-1]
    if route == "bn":
        x, running_mean, running_var, weight, bias = args[0], args[1], args[2], args[3], args[4]
        N, C, H, W = x.shape
        HW       = H * W
        BLOCK_HW = triton.next_power_of_2(HW)
        if BLOCK_HW < 16:
            BLOCK_HW = 16
        out  = torch.empty_like(x)
        grid = (N, C)
        _bn_inf_kernel[grid](
            x, running_mean, running_var, weight, bias, out,
            C, HW, 1e-5,
            BLOCK_HW=BLOCK_HW,
        )
        return out
    elif route == "avgpool":
        x       = args[0]
        N, C, H, W = x.shape
        OH  = (H + 1) // 2
        OW  = (W + 1) // 2
        out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
        total = N * C * OH * OW
        grid  = ((total + 511) // 512,)
        _avg_pool2x2_s2_kernel[grid](
            x, out,
            N, C, OH, OW, H, W,
            BLOCK_SIZE=512,
        )
        return out
    return args[0]