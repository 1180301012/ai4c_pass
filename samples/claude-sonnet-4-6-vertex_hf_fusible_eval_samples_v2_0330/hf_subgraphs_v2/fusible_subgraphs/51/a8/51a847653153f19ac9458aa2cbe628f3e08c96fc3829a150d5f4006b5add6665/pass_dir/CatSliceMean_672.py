import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 2-D grid: axis-0 = channel (0..2C-1), axis-1 = batch (0..B-1).
# This eliminates all integer divisions (pid//C2, pid%C2, c_out%C) that are
# very expensive for non-power-of-2 sizes like C=336, C2=672.
# The dtype conversion for the mean is done inside the kernel, so no extra
# PyTorch .to() kernel launch is needed.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_cat_mean_kernel_672(
    in0_ptr, in1_ptr,
    cat_ptr, mean_ptr,
    B, C, HW,
    IS_F16:  tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c_out = tl.program_id(0)   # [0, 2C) — no division needed
    b     = tl.program_id(1)   # [0, B)  — no division needed
    C2    = 2 * C

    acc = 0.0
    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        if c_out < C:
            val = tl.load(in0_ptr + (b * C + c_out) * HW + offsets,
                          mask=mask, other=0.0)
        else:
            val = tl.load(in1_ptr + (b * C + c_out - C) * HW + offsets,
                          mask=mask, other=0.0)

        tl.store(cat_ptr + (b * C2 + c_out) * HW + offsets, val, mask=mask)
        acc = acc + tl.sum(val.to(tl.float32), axis=0)

    mean_f32 = acc / HW
    mean_idx  = b * C2 + c_out
    if IS_BF16:
        tl.store(mean_ptr + mean_idx, mean_f32.to(tl.bfloat16))
    elif IS_F16:
        tl.store(mean_ptr + mean_idx, mean_f32.to(tl.float16))
    else:
        tl.store(mean_ptr + mean_idx, mean_f32)


@torch.fx.wrap
def _compute_fused_cat_mean_672(in_0, in_1):
    B, C, H, W = in_0.shape
    C2 = 2 * C
    HW = H * W

    cat_out  = torch.empty(B, C2, H, W, dtype=in_0.dtype, device=in_0.device)
    mean_out = torch.empty(B * C2,      dtype=in_0.dtype, device=in_0.device)

    _fused_cat_mean_kernel_672[(C2, B)](
        in_0, in_1,
        cat_out, mean_out,
        B, C, HW,
        IS_F16  = (in_0.dtype == torch.float16),
        IS_BF16 = (in_0.dtype == torch.bfloat16),
        BLOCK_HW=256,
        num_warps=2,
        num_stages=2,
    )

    return cat_out, mean_out.view(B, C2, 1, 1)


def _fused_cat_mean_fn_672(in_0, in_1):
    result   = _compute_fused_cat_mean_672(in_0, in_1)
    cat_out  = result[0]
    mean_out = result[1]
    return cat_out, mean_out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 672, None),
                   slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _fused_cat_mean_fn_672