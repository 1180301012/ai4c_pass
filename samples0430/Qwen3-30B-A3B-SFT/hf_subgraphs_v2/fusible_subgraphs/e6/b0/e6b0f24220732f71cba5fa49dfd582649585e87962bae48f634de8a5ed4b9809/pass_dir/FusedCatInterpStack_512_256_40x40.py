import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['C_half', 'HW'],
)
@triton.jit
def fused_cat_kernel(
    in2_ptr,    # [B, C_half, H, W]
    in3_ptr,    # [B, C_half, H, W]
    out_ptr,    # [B, 2*C_half, H, W]
    B, C_half, H, W,
    HW,
    BLOCK_HW: tl.constexpr,
):
    bc_id     = tl.program_id(0)   # 0 .. B*C_half - 1
    hw_blk_id = tl.program_id(1)

    b = bc_id // C_half
    c = bc_id % C_half

    offsets = hw_blk_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    HW_i     = tl.cast(HW, tl.int32)
    C_half_i = tl.cast(C_half, tl.int32)

    src = b * (C_half_i * HW_i) + c * HW_i + offsets
    val2 = tl.load(in2_ptr + src, mask=mask, other=0.0)
    val3 = tl.load(in3_ptr + src, mask=mask, other=0.0)

    out1 = b * (2 * C_half_i * HW_i) + c * HW_i + offsets
    out2 = b * (2 * C_half_i * HW_i) + (c + C_half_i) * HW_i + offsets
    tl.store(out_ptr + out1, val2, mask=mask)
    tl.store(out_ptr + out2, val3, mask=mask)


@torch.fx.wrap
def triton_cat_dim1(in_2, in_3):
    B, C_half, H, W = in_2.shape
    HW  = H * W
    C   = 2 * C_half
    out = torch.empty((B, C, H, W), dtype=in_2.dtype, device=in_2.device)
    grid = lambda meta: (B * C_half, triton.cdiv(HW, meta['BLOCK_HW']))
    fused_cat_kernel[grid](in_2, in_3, out, B, C_half, H, W, HW)
    return out


# ── Pattern & Replacement ─────────────────────────────────────────────────────

def pattern(in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    return tmp_0


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return triton_cat_dim1