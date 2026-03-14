import torch
import triton
import triton.language as tl


def pattern(in_6, tmp_4, in_0, in_1, in_3, in_2, in_5):
    """
    Pattern: Conv2D + BatchNorm + Add (residual connection)
    This matches the exact computation from model.py
    """
    tmp_5 = torch.conv2d(in_6, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = tmp_6 + in_5
    return tmp_5, tmp_6


def replacement_args(in_6, tmp_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, tmp_4, in_0, in_1, in_3, in_2, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=3, num_warps=4),
    ],
    key=['C_out'],
)
@triton.jit
def fused_conv_bn_add_kernel(
    input_ptr, weight_ptr, mean_ptr, var_ptr, weight_bn_ptr, bias_bn_ptr, residual_ptr,
    output_ptr,
    N, C_in, C_out, H, W,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_co, stride_weight_ci,
    stride_residual_n, stride_residual_c, stride_residual_h, stride_residual_w,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: Conv2D (1x1) + BatchNorm + Add
    """
    pid = tl.program_id(0)
    num_positions = N * H * W
    
    pos = pid
    n = pos // (H * W)
    pos = pos % (H * W)
    h = pos // W
    w = pos % W
    
    if n >= N:
        return
    
    # Load BN params
    mean = tl.load(mean_ptr + tl.arange(0, C_out))
    var = tl.load(var_ptr + tl.arange(0, C_out))
    weight_bn = tl.load(weight_bn_ptr + tl.arange(0, C_out))
    bias_bn = tl.load(bias_bn_ptr + tl.arange(0, C_out))
    
    # Compute BN scale and bias term
    inv_std = tl.rsqrt(var + eps)
    scale = weight_bn * inv_std
    bias_term = bias_bn - mean * scale
    
    # Load input slice: (C_in,)
    input_slice = tl.load(
        input_ptr + n * stride_input_n + 
        h * stride_input_h + w * stride_input_w +
        tl.arange(0, C_in) * stride_input_c
    )
    
    # Process in channel chunks
    for ch_start in range(0, C_out, BLOCK_N):
        ch_end = min(ch_start + BLOCK_N, C_out)
        ch_mask = ch_end - ch_start
        
        # Load weight slice: (BLOCK_N, C_in)
        weight_slice = tl.load(
            weight_ptr + ch_start * stride_weight_co +
            tl.arange(0, BLOCK_N)[:, None] * stride_weight_co +
            tl.arange(0, C_in)[None, :] * stride_weight_ci
        )
        
        # Matrix multiply: (1, C_in) @ (C_in, BLOCK_N) = (1, BLOCK_N)
        conv_out = tl.dot(input_slice[None, :], weight_slice)
        
        # Apply BN
        scale_slice = scale[ch_start:ch_end]
        bias_slice = bias_term[ch_start:ch_end]
        bn_out = conv_out * scale_slice + bias_slice
        
        # Add residual
        residual = tl.load(
            residual_ptr + n * stride_residual_n + ch_start * stride_residual_c +
            h * stride_residual_h + w * stride_residual_w +
            tl.arange(0, BLOCK_N)[:ch_mask] * stride_residual_c
        )
        
        final_out = bn_out + residual
        
        # Store result
        tl.store(
            output_ptr + n * stride_output_n + ch_start * stride_output_c +
            h * stride_output_h + w * stride_output_w +
            tl.arange(0, BLOCK_N)[:ch_mask] * stride_output_c,
            final_out
        )


def fused_conv_bn_add(x, weight, mean, var, weight_bn, bias_bn, residual):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    output = torch.empty_like(residual)
    grid = (N * H * W,)
    
    fused_conv_bn_add_kernel[grid](
        x, weight, mean, var, weight_bn, bias_bn, residual, output,
        N, C_in, C_out, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        1e-05,
    )
    
    return output


fused_conv_bn_add = torch.fx.wrap(fused_conv_bn_add)


def replacement_func():
    return fused_conv_bn_add