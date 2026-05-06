import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────
# Pattern: adaptive_avg_pool2d → batch_norm → relu
# Matches both float16/bfloat16 and float32 variants
# ──────────────────────────────────────────────
def pattern(in_5, in_1, in_2, in_4, in_3):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


# ──────────────────────────────────────────────
# Triton kernel: fused global-avgpool + BN + ReLU
#
# Grid  : (B * C,)  — one program per (batch, channel) pair
# Thread: BLOCK_HW = 64 threads, each handles one spatial pixel
# ──────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64},  num_warps=2),
        triton.Config({"BLOCK_HW": 64},  num_warps=4),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
    ],
    key=["C", "HW"],
)
@triton.jit
def _avgpool_bn_relu_kernel(
    input_ptr,          # [B, C, H, W]
    mean_ptr,           # [C]  running_mean
    var_ptr,            # [C]  running_var
    weight_ptr,         # [C]  BN weight (scale)
    bias_ptr,           # [C]  BN bias
    output_ptr,         # [B, C]  (will be viewed as [B, C, 1, 1])
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base pointer offset for element (b, c, 0, 0)
    base = (b * C + c) * HW

    # Load all HW spatial values for this (b, c) pair
    offsets = tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    # Global average pooling
    x_sum = tl.sum(x, axis=0)
    x_avg = x_sum / HW

    # Batch Normalization (inference: training=False)
    #   y = gamma * (x - mean) / sqrt(var + eps) + beta
    mean    = tl.load(mean_ptr   + c).to(tl.float32)
    var     = tl.load(var_ptr    + c).to(tl.float32)
    weight  = tl.load(weight_ptr + c).to(tl.float32)
    bias    = tl.load(bias_ptr   + c).to(tl.float32)

    eps     = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = weight * (x_avg - mean) * inv_std + bias

    # ReLU
    y = tl.maximum(y, 0.0)

    # Store scalar result → output[b, c, 0, 0]
    tl.store(output_ptr + pid, y)


# ──────────────────────────────────────────────
# Python wrapper (must be decorated @torch.fx.wrap)
# ──────────────────────────────────────────────
@torch.fx.wrap
def fused_avgpool_bn_relu(in_5, in_1, in_2, in_4, in_3):
    """
    in_5  : [B, C, H, W]  input feature map
    in_1  : [C]            running_mean         (BN)
    in_2  : [C]            running_var           (BN)
    in_4  : [C]            weight (gamma)        (BN)
    in_3  : [C]            bias   (beta)         (BN)
    """
    B, C, H, W = in_5.shape
    HW = H * W
    output_fp32 = torch.empty((B * C,), dtype=torch.float32, device=in_5.device)

    grid = (B * C,)
    _avgpool_bn_relu_kernel[grid](
        in_5, in_1, in_2, in_4, in_3,
        output_fp32,
        C, HW,
    )

    # Return tensor with original dtype
    return output_fp32.to(in_5.dtype)


def replacement_func():
    return fused_avgpool_bn_relu