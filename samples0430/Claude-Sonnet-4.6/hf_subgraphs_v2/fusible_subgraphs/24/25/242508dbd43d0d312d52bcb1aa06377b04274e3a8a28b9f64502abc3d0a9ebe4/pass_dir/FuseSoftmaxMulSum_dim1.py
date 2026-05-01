import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused kernel  (B=1 fast-path, no integer division in hot-path)
#
# Grid : (C, n_tiles)   c = program_id(0)  directly (no division)
# ● BLOCK_SIZE ≥ HW  → 1 tile/channel → minimum scheduling waves
# ● num_warps = 8    → 256 blocks × 8 = 2048 warps; fits in 1 wave on A30
# ● Binary softmax (sigmoid trick) → 1 exp() instead of 2
# ● num_stages = 2   → instruction-level pipelining for the two vector loads
# ---------------------------------------------------------------------------
@triton.jit
def _fused_b1_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    HW,            # H * W
    CHW,           # C * H * W
    C,             # channel count
    BLOCK_SIZE: tl.constexpr,
):
    c   = tl.program_id(0)
    pid = tl.program_id(1)

    off  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < HW

    # Scalar softmax weights (broadcast in every warp of this block)
    x0 = tl.load(in1_ptr + c    ).to(tl.float32)
    x1 = tl.load(in1_ptr + C + c).to(tl.float32)

    diff  = tl.minimum(tl.maximum(x0 - x1, -80.0), 80.0)
    exp_d = tl.exp(diff)
    inv   = 1.0 / (exp_d + 1.0)
    s0    = exp_d * inv
    s1    = inv

    base = c * HW + off
    a0 = tl.load(in0_ptr + base,       mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(in0_ptr + CHW + base, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + base, a0 * s0 + a1 * s1, mask=mask)


# General kernel for B > 1 (correctness fallback)
@triton.jit
def _fused_general_kernel(
    in0_ptr, in1_ptr, out_ptr,
    HW, CHW, C,
    BLOCK_SIZE: tl.constexpr,
):
    bc  = tl.program_id(0)
    pid = tl.program_id(1)
    b   = bc // C
    c   = bc - b * C

    off  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < HW

    in1_base = b * (2 * C) + c
    x0 = tl.load(in1_ptr + in1_base    ).to(tl.float32)
    x1 = tl.load(in1_ptr + in1_base + C).to(tl.float32)

    diff  = tl.minimum(tl.maximum(x0 - x1, -80.0), 80.0)
    exp_d = tl.exp(diff)
    inv   = 1.0 / (exp_d + 1.0)

    in0_base = b * (2 * CHW) + c * HW + off
    a0 = tl.load(in0_ptr + in0_base,       mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(in0_ptr + in0_base + CHW, mask=mask, other=0.0).to(tl.float32)

    out_base = b * CHW + c * HW + off
    tl.store(out_ptr + out_base, a0 * (exp_d * inv) + a1 * inv, mask=mask)


# ---------------------------------------------------------------------------
# Module-level param cache  →  avoid recomputing grid/BS every call
# ---------------------------------------------------------------------------
_param_cache: dict = {}


def _get_params(HW: int, C: int, B: int):
    key = (HW, C, B)
    if key in _param_cache:
        return _param_cache[key]
    if HW <= 256:
        BS, NW = 256, 8
    elif HW <= 512:
        BS, NW = 512, 8
    else:
        BS, NW = 1024, 8
    n_tiles = (HW + BS - 1) // BS
    grid_b1  = (C, n_tiles)
    grid_gen = (B * C, n_tiles)
    result = (BS, NW, grid_b1, grid_gen)
    _param_cache[key] = result
    return result


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    """
    out[b,c,h,w] = sum_k( in_0[b,k,c,h,w] * softmax(in_1[b,:,c,0,0])[k] )
    in_0 : [B, K=2, C, H, W]
    in_1 : [B, K=2, C, 1, 1]
    out  : [B, C, H, W]
    """
    B, K, C, H, W = in_0.shape
    HW  = H * W
    CHW = C * H * W

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    BS, NW, grid_b1, grid_gen = _get_params(HW, C, B)

    if B == 1:
        _fused_b1_kernel[grid_b1](
            in_0, in_1, out,
            HW, CHW, C,
            BLOCK_SIZE=BS, num_warps=NW, num_stages=2,
        )
    else:
        _fused_general_kernel[grid_gen](
            in_0, in_1, out,
            HW, CHW, C,
            BLOCK_SIZE=BS, num_warps=NW, num_stages=2,
        )

    return out


def replacement_func():
    return fused_softmax_mul_sum