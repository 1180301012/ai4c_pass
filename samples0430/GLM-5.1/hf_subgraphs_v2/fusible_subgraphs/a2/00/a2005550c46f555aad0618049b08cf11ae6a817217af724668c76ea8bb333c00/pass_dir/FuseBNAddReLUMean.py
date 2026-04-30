import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_bn_add_relu_mean_kernel(
    in4_ptr, in5_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    out_relu_ptr, out_mean_ptr,
    N, C, H, W,
    eps,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) channel
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    # Load BN parameters for this channel, upcast to float32 for precision
    mean_val = tl.load(bn_mean_ptr + c).to(tl.float32)
    var_val = tl.load(bn_var_ptr + c).to(tl.float32)
    weight_val = tl.load(bn_weight_ptr + c).to(tl.float32)
    bias_val = tl.load(bn_bias_ptr + c).to(tl.float32)

    # Precompute BN scale and offset for inference mode
    # BN output = input * scale + offset
    # where scale = weight / sqrt(var + eps), offset = bias - mean * scale
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale = weight_val * inv_std
    offset = bias_val - mean_val * scale

    HW = H * W
    sum_val = 0.0

    # Base index for this channel's spatial data
    base_idx = n * C * H * W + c * H * W

    # Iterate over spatial elements in blocks
    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW

        # Load input values, upcast to float32
        in4_val = tl.load(in4_ptr + base_idx + offsets, mask=mask, other=0.0).to(tl.float32)
        in5_val = tl.load(in5_ptr + base_idx + offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused computation: BN(in4) + in5, then ReLU
        bn_out = in4_val * scale + offset
        add_out = in5_val + bn_out
        relu_out = tl.maximum(add_out, 0.0)

        # Store relu output at valid positions only
        tl.store(out_relu_ptr + base_idx + offsets, relu_out, mask=mask)

        # Accumulate for mean computation - zero out masked positions
        valid_relu = tl.where(mask, relu_out, 0.0)
        sum_val += tl.sum(valid_relu, axis=0)

    # Compute mean and store at [n, c, 0, 0] position
    mean_result = sum_val / (H * W)
    mean_idx = n * C + c
    tl.store(out_mean_ptr + mean_idx, mean_result)


@torch.fx.wrap
def fused_bn_add_relu_mean(bn_mean, bn_var, bn_bias, bn_weight, in4, in5):
    N, C, H, W = in4.shape

    out_relu = torch.empty_like(in4)
    out_mean = torch.empty((N, C, 1, 1), dtype=in4.dtype, device=in4.device)

    grid = (N * C,)

    fused_bn_add_relu_mean_kernel[grid](
        in4, in5, bn_mean, bn_var, bn_weight, bn_bias,
        out_relu, out_mean,
        N, C, H, W,
        eps=1e-05,
    )

    return out_relu, out_mean


def replacement_func():
    return fused_bn_add_relu_mean