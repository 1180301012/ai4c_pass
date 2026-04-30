import torch
import triton
import triton.language as tl


def pattern(conv_input, weight, bias, feature):
    conv2d = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_out = torch.sigmoid(conv2d)
    result = feature * sigmoid_out
    return result


def replacement_args(conv_input, weight, bias, feature):
    return (conv_input, weight, bias, feature)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SPATIAL': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=4, num_stages=3),
    ],
    key=['B', 'H', 'W'],
)
@triton.jit
def fused_conv_sigmoid_mul_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, feature_ptr, output_ptr,
    B, C_in: tl.constexpr, C_out: tl.constexpr, H, W,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    # Program IDs for 2D grid
    bc_id = tl.program_id(0)
    spatial_block_id = tl.program_id(1)

    b = bc_id // C_out
    c = bc_id % C_out

    # Compute conv output for this (b, c) pair
    # conv_out = sum_k(weight[c,k] * input[b,k]) + bias[c]
    conv_out = tl.load(bias_ptr + c).to(tl.float32)  # bias[c]

    # Dot product over input channels
    k_offsets = tl.arange(0, BLOCK_CIN)
    k_mask = k_offsets < C_in

    w_offset = c * C_in + k_offsets
    w_vals = tl.load(weight_ptr + w_offset, mask=k_mask, other=0.0).to(tl.float32)

    i_offset = b * C_in + k_offsets
    i_vals = tl.load(conv_input_ptr + i_offset, mask=k_mask, other=0.0).to(tl.float32)

    conv_out += tl.sum(w_vals * i_vals, axis=0)

    # Apply sigmoid
    sigmoid_val = tl.sigmoid(conv_out)

    # Process spatial block of the feature map
    spatial_size = H * W
    s_start = spatial_block_id * BLOCK_SPATIAL
    s_offsets = s_start + tl.arange(0, BLOCK_SPATIAL)
    s_mask = s_offsets < spatial_size

    # Load feature[b, c, spatial_positions]
    feature_base = b * C_out * spatial_size + c * spatial_size
    feature_vals = tl.load(feature_ptr + feature_base + s_offsets, mask=s_mask, other=0.0).to(tl.float32)

    # Multiply and store
    output_vals = feature_vals * sigmoid_val
    tl.store(output_ptr + feature_base + s_offsets, output_vals, mask=s_mask)


@torch.fx.wrap
def fused_conv_sigmoid_mul(conv_input, weight, bias, feature):
    conv_input = conv_input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    feature = feature.contiguous()

    B = feature.shape[0]
    C_out = feature.shape[1]
    H = feature.shape[2]
    W = feature.shape[3]
    C_in = conv_input.shape[1]

    output = torch.empty(feature.shape, dtype=feature.dtype, device=feature.device)

    spatial_size = H * W

    grid = lambda META: (
        B * C_out,
        triton.cdiv(spatial_size, META['BLOCK_SPATIAL']),
    )

    fused_conv_sigmoid_mul_kernel[grid](
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        feature_ptr=feature,
        output_ptr=output,
        B=B, C_in=C_in, C_out=C_out, H=H, W=W,
        BLOCK_CIN=16,
    )

    return output


def replacement_func():
    return fused_conv_sigmoid_mul