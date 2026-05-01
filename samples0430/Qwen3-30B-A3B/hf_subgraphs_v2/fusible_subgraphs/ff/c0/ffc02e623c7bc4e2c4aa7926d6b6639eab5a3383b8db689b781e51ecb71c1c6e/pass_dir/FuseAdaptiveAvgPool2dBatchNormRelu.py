import torch
import triton
import triton.language as tl

@triton.jit
def fused_avg_pool_bn_relu_kernel(
    in_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    channels,
    spatial_h,
    spatial_w,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // channels
    c = pid % channels

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    total = 0.0
    for h in range(spatial_h):
        for w in range(spatial_w):
            idx = b * channels * spatial_h * spatial_w + c * spatial_h * spatial_w + h * spatial_w + w
            val = tl.load(in_ptr + idx)
            total += val
    avg = total / (spatial_h * spatial_w)

    normalized = (avg - mean) / tl.sqrt(var + eps)
    normalized = normalized * weight + bias
    out_val = tl.maximum(normalized, 0.0)

    out_idx = b * channels + c
    tl.store(out_ptr + out_idx, out_val)

@torch.fx.wrap
def fused_avg_pool_bn_relu(in_5, in_1, in_2, in_4, in_3):
    batch, channels, spatial_h, spatial_w = in_5.shape
    eps = 1e-05
    out_flat = torch.empty(batch * channels, dtype=in_5.dtype, device=in_5.device)
    grid_size = (batch * channels + 255) // 256

    fused_avg_pool_bn_relu_kernel[grid_size](
        in_5, in_1, in_2, in_4, in_3,
        out_flat, batch, channels, spatial_h, spatial_w, eps, 256
    )

    out = out_flat.view(batch, channels, 1, 1)
    return out

def pattern(in_5, in_1, in_2, in_4, in_3):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8

def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)

def replacement_func():
    return fused_avg_pool_bn_relu