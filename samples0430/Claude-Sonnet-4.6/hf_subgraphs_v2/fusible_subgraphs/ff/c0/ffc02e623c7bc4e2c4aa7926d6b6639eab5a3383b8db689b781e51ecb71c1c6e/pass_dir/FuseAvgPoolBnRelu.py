import torch
import triton
import triton.language as tl


@triton.jit
def fused_avgpool_bn_relu_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Load HW spatial values for this (b, c)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    base = b * C * HW + c * HW
    x_vals = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0)

    # Adaptive average pooling: mean over H*W
    avg = tl.sum(x_vals.to(tl.float32), axis=0) / HW

    # Batch norm inference: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    mean  = tl.load(running_mean_ptr + c).to(tl.float32)
    var   = tl.load(running_var_ptr   + c).to(tl.float32)
    w     = tl.load(weight_ptr        + c).to(tl.float32)
    b_val = tl.load(bias_ptr          + c).to(tl.float32)

    inv_std = tl.rsqrt(var + eps)
    result  = (avg - mean) * inv_std * w + b_val

    # ReLU
    result = tl.maximum(result, 0.0)

    # Store to output [B, C, 1, 1]  (strides: C, 1, 1, 1)
    tl.store(out_ptr + b * C + c, result)


@torch.fx.wrap
def fused_avgpool_bn_relu(x, running_mean, running_var, weight, bias):
    B, C, H, W = x.shape
    HW = H * W

    # Output shape [B, C, 1, 1]
    out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    grid = (B * C,)
    fused_avgpool_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        B, C, HW,
        eps=1e-5,
        BLOCK_HW=64,
        num_warps=4,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_avgpool_bn_relu