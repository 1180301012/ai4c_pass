import torch
import triton
import triton.language as tl


def pattern(cat_left, cat_right, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight):
    cat_out = torch.cat([cat_left, cat_right], 1)
    bn_out = torch.nn.functional.batch_norm(cat_out, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    pool_out = torch.nn.functional.adaptive_avg_pool2d(prelu_out, 1)
    return prelu_out, pool_out


def replacement_args(cat_left, cat_right, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight):
    return (cat_left, cat_right, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_cat_bn_prelu_pool_kernel(
    cat_left_ptr,
    cat_right_ptr,
    prelu_out_ptr,
    pool_out_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    prelu_weight_ptr,
    C_left: tl.int32,
    C_right: tl.int32,
    C_total: tl.int32,
    H: tl.int32,
    W: tl.int32,
    HW: tl.int32,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    c_idx = nc_idx % C_total
    n_idx = nc_idx // C_total

    # Load BN parameters for this channel (compute in float32 for accuracy)
    mean_val = tl.load(bn_mean_ptr + c_idx).to(tl.float32)
    var_val = tl.load(bn_var_ptr + c_idx).to(tl.float32)
    weight_val = tl.load(bn_weight_ptr + c_idx).to(tl.float32)
    bias_val = tl.load(bn_bias_ptr + c_idx).to(tl.float32)
    prelu_w = tl.load(prelu_weight_ptr + c_idx).to(tl.float32)

    # Compute BN scale and offset (inference mode)
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    bn_scale = weight_val * inv_std
    bn_offset = bias_val - weight_val * mean_val * inv_std

    # Determine which input tensor to read from based on channel index
    # For c < C_left: read from cat_left at channel c
    # For c >= C_left: read from cat_right at channel (c - C_left)
    if c_idx < C_left:
        src_ptr = cat_left_ptr
        src_c = c_idx
        src_n_stride = C_left * HW
    else:
        src_ptr = cat_right_ptr
        src_c = c_idx - C_left
        src_n_stride = C_right * HW

    # Compute base offsets for source and output
    src_base = n_idx * src_n_stride + src_c * HW
    out_base = n_idx * C_total * HW + c_idx * HW

    # Accumulator for adaptive avg pool
    pool_sum = 0.0

    # Process spatial positions in blocks
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW

        # Read from source tensor
        x = tl.load(src_ptr + src_base + hw_offsets, mask=mask, other=0.0).to(tl.float32)

        # Apply BN: y = scale * x + offset
        y = bn_scale * x + bn_offset

        # Apply PReLU: z = max(0, y) + prelu_w * min(0, y)
        z = tl.where(y > 0.0, y, prelu_w * y)

        # Mask out invalid positions for accumulation
        z_valid = tl.where(mask, z, 0.0)

        # Write PReLU output
        tl.store(prelu_out_ptr + out_base + hw_offsets, z_valid, mask=mask)

        # Accumulate for pool (only valid positions contribute)
        pool_sum += tl.sum(z_valid)

    # Pool output: mean over spatial dimensions
    pool_val = pool_sum / (H * W)

    # Store pool output at [n, c, 0, 0] (flat offset = n * C_total + c)
    tl.store(pool_out_ptr + n_idx * C_total + c_idx, pool_val)


@torch.fx.wrap
def fused_cat_bn_prelu_pool(cat_left, cat_right, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight):
    N = cat_left.shape[0]
    C_left = cat_left.shape[1]
    C_right = cat_right.shape[1]
    C_total = C_left + C_right
    H = cat_left.shape[2]
    W = cat_left.shape[3]
    HW = H * W
    eps = 0.001

    # Create output tensors
    prelu_out = torch.empty((N, C_total, H, W), dtype=cat_left.dtype, device=cat_left.device)
    pool_out = torch.empty((N, C_total, 1, 1), dtype=cat_left.dtype, device=cat_left.device)

    # Grid: one program per (n, c) pair
    num_programs = N * C_total
    grid = (num_programs,)

    BLOCK_HW = 1024
    # Ensure BLOCK_HW >= HW for efficiency
    while BLOCK_HW < HW:
        BLOCK_HW *= 2

    fused_cat_bn_prelu_pool_kernel[grid](
        cat_left_ptr=cat_left,
        cat_right_ptr=cat_right,
        prelu_out_ptr=prelu_out,
        pool_out_ptr=pool_out,
        bn_mean_ptr=bn_running_mean,
        bn_var_ptr=bn_running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        prelu_weight_ptr=prelu_weight,
        C_left=C_left,
        C_right=C_right,
        C_total=C_total,
        H=H,
        W=W,
        HW=HW,
        eps=eps,
        BLOCK_HW=BLOCK_HW,
    )

    return prelu_out, pool_out


def replacement_func():
    return fused_cat_bn_prelu_pool