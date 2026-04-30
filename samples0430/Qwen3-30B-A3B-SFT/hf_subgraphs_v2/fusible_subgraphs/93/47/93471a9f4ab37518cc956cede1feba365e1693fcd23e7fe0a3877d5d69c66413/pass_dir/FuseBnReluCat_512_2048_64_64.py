import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: BN (inference) + ReLU + cat
#
# Inputs:
#   in_a, in_b, in_c, in_d : [1, C, H, W]   (bfloat16 / float16 / float32)
#   in_e                   : [1, C, H, W]   (already post-relu BN output)
#   mean, var, weight, bias: [C]
#
# Output:
#   out[0, 0:C,       :, :] = relu(BN(in_a))
#   out[0, C:2C,      :, :] = relu(BN(in_b))
#   out[0, 2C:3C,     :, :] = relu(BN(in_c))
#   out[0, 3C:4C,     :, :] = relu(BN(in_d))
#   out[0, 4C:5C,     :, :] = relu(BN(in_e))
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_bn_relu_cat_kernel(
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    in_a_ptr,
    in_b_ptr,
    in_c_ptr,
    in_d_ptr,
    in_e_ptr,
    out_ptr,
    C,
    HW,
    C_total,
    c_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c  = tl.program_id(0)   # channel group index  [0, C_total)
    pid_hw = tl.program_id(1)   # spatial block index

    # Global channel index for BN parameters
    c_global = pid_c + c_offset

    # Spatial offsets within this block
    hw_start = pid_hw * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW

    # ---- Load BN parameters (one scalar per channel) ----
    mean_val = tl.load(mean_ptr   + c_global).to(tl.float32)
    var_val  = tl.load(var_ptr    + c_global).to(tl.float32)
    wt_val   = tl.load(weight_ptr + c_global).to(tl.float32)
    bias_val = tl.load(bias_ptr   + c_global).to(tl.float32)

    # Pre-compute inverse std once (same for all BLOCK_SIZE elements)
    eps     = 1e-5
    inv_std = 1.0 / tl.sqrt(var_val + eps)

    # ---- Determine source tensor ----
    # Each program handles channels in exactly one of the 5 ranges
    # so these are scalar conditions – no warp divergence.
    if pid_c < C:
        x = tl.load(in_a_ptr + c_global * HW + offsets, mask=mask)
    elif pid_c < 2 * C:
        x = tl.load(in_b_ptr + (c_global - C) * HW + offsets, mask=mask)
    elif pid_c < 3 * C:
        x = tl.load(in_c_ptr + (c_global - 2 * C) * HW + offsets, mask=mask)
    elif pid_c < 4 * C:
        x = tl.load(in_d_ptr + (c_global - 3 * C) * HW + offsets, mask=mask)
    else:
        x = tl.load(in_e_ptr + (c_global - 4 * C) * HW + offsets, mask=mask)

    # ---- Batch-norm (inference) + ReLU ----
    x_f32    = x.to(tl.float32)
    normalized = (x_f32 - mean_val) * inv_std * wt_val + bias_val
    out_f32    = tl.maximum(normalized, 0.0)
    out_val    = out_f32.to(x.dtype)

    # ---- Store to output ----
    tl.store(out_ptr + pid_c * HW + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_bn_relu_cat(mean, var, bias, weight, in_a, in_b, in_c, in_d, in_e):
    """
    Replaces:
        bn  = batch_norm(x, mean, var, weight, bias, False, 0.1, 1e-05)
        rel = relu(bn, inplace=False)
        out = cat([a, b, c, d, rel], dim=1)
    """
    C    = 512
    H    = 64
    W    = 64
    HW   = H * W       # 4096
    C5   = 5 * C       # 2560

    out = torch.empty((1, C5, H, W), dtype=in_a.dtype, device=in_a.device)

    grid = (C5, triton.cdiv(HW, 1024))   # will be overridden by autotune

    fused_bn_relu_cat_kernel[lambda meta: (C5, triton.cdiv(HW, meta['BLOCK_SIZE']))](
        mean,
        var,
        bias,
        weight,
        in_a, in_b, in_c, in_d, in_e,
        out,
        C, HW, C5,
        0,        # c_offset for in_a
        BLOCK_SIZE=1024,   # overridden by autotune
    )

    return (out,)


# ---------------------------------------------------------------------------
# Pattern – must mirror the exact call signatures in model.py
# ---------------------------------------------------------------------------
def pattern(mean, var, bias, weight, in_a, in_b, in_c, in_d, in_e):
    bn  = torch.nn.functional.batch_norm(
              in_e, mean, var, weight, bias, False, 0.1, 1e-05)
    rel = torch.nn.functional.relu(bn, inplace=False)
    out = torch.cat([in_a, in_b, in_c, in_d, rel], dim=1)
    return (out,)


def replacement_args(mean, var, bias, weight, in_a, in_b, in_c, in_d, in_e):
    return (mean, var, bias, weight, in_a, in_b, in_c, in_d, in_e)


def replacement_func():
    return fused_bn_relu_cat