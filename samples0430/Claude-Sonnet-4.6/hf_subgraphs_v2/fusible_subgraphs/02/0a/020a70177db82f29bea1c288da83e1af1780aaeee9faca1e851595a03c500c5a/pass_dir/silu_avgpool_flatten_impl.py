import torch
import triton
import triton.language as tl


@triton.jit
def _silu_global_avgpool_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused SiLU + Global Average Pool kernel.
    Each program handles one (B, C) pair, reducing over H*W spatial elements.
    Input layout: [B, C, H, W] contiguous -> (b*C + c) * HW is the base offset.
    Output layout: [B, C] -> one scalar per program.
    """
    pid = tl.program_id(0)
    base = pid * HW

    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    # Load H*W elements for this (b, c) channel slice
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # SiLU activation: x * sigmoid(x)  (computed in float32 for precision)
    x_f32 = x.to(tl.float32)
    silu_x = x_f32 * tl.sigmoid(x_f32)

    # Reduce: sum over valid H*W elements (masked positions loaded as 0.0)
    total = tl.sum(silu_x, axis=0)

    # Average
    avg = total / HW

    # Write result, cast back to original dtype
    tl.store(out_ptr + pid, avg.to(x.dtype))


@torch.fx.wrap
def silu_avgpool_flatten(x):
    """
    Fused: SiLU -> global avg pool -> flatten.
    Dropout (training=False) is identity and is absorbed here.

    Input:  [B, C, H, W]
    Output: [B, C]
    """
    B, C, H, W = x.shape
    HW = H * W
    n_programs = B * C

    out = torch.empty((B, C), dtype=x.dtype, device=x.device)

    # Choose smallest power-of-2 BLOCK_HW that covers all spatial elements
    if HW <= 64:
        BLOCK_HW = 64
    elif HW <= 128:
        BLOCK_HW = 128
    else:
        BLOCK_HW = 256

    _silu_global_avgpool_kernel[(n_programs,)](
        x, out, HW, BLOCK_HW,
        num_warps=4,
    )

    return out