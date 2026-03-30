import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused cat + spatial-mean in one memory pass
# Each program handles one (batch, channel_out) pair.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_cat_mean_kernel(
    in0_ptr, in1_ptr,
    cat_ptr, mean_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid   = tl.program_id(0)
    C2    = 2 * C
    b     = pid // C2
    c_out = pid % C2

    local_c  = c_out % C
    src_base = (b * C + local_c) * HW
    dst_base = (b * C2 + c_out) * HW

    acc = 0.0

    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        if c_out < C:
            val = tl.load(in0_ptr + src_base + offsets, mask=mask, other=0.0)
        else:
            val = tl.load(in1_ptr + src_base + offsets, mask=mask, other=0.0)

        tl.store(cat_ptr + dst_base + offsets, val, mask=mask)
        acc = acc + tl.sum(val.to(tl.float32), axis=0)

    # Store mean in float32; wrapper casts to input dtype
    tl.store(mean_ptr + b * C2 + c_out, acc / HW)


# ---------------------------------------------------------------------------
# FX-atomic wrapper (returns a tuple so the outer function can unpack)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _run_fused_cat_mean(in_0, in_1):
    B, C, H, W = in_0.shape
    C2 = 2 * C
    HW = H * W

    cat_out  = torch.empty(B, C2, H, W, dtype=in_0.dtype, device=in_0.device)
    mean_f32 = torch.empty(B * C2,      dtype=torch.float32, device=in_0.device)

    BLOCK_HW = 256
    _fused_cat_mean_kernel[(B * C2,)](
        in_0, in_1,
        cat_out, mean_f32,
        B, C, HW,
        BLOCK_HW=BLOCK_HW,
    )

    mean_out = mean_f32.to(in_0.dtype).view(B, C2, 1, 1)
    return cat_out, mean_out


# Regular (non-wrapped) function — FX traces into this,
# producing 2 separate operator.getitem returning nodes.
def _fused_cat_mean_fn(in_0, in_1):
    result   = _run_fused_cat_mean(in_0, in_1)
    cat_out  = result[0]
    mean_out = result[1]
    return cat_out, mean_out


# ---------------------------------------------------------------------------
# Pattern: cat + any getitem + mean.
# Using all-None slice so the FX pattern graph matches the generic structure.
# The framework may treat constant values in getitem args as wildcards.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, None, None),
                   slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _fused_cat_mean_fn