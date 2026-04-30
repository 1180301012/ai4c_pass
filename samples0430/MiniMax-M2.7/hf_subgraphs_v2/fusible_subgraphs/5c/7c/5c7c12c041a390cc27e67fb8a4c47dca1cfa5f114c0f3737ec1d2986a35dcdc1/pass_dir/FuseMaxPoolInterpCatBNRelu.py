import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    pooled_ptr, concat_ptr, output_ptr,
    N, C_pool, C_concat, H_out, W_out,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    stride_pool_N, stride_pool_C, stride_pool_H, stride_pool_W,
    stride_concat_N, stride_concat_C, stride_concat_H, stride_concat_W,
    stride_out_N, stride_out_C, stride_out_H, stride_out_W,
    bn_channels, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. concat with another tensor
    2. batch_norm
    3. relu
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elems = N * (C_pool + C_concat) * H_out * W_out
    mask = offsets < total_elems
    
    # Calculate indices for output tensor
    n = offsets // ((C_pool + C_concat) * H_out * W_out)
    rem = offsets % ((C_pool + C_concat) * H_out * W_out)
    c = rem // (H_out * W_out)
    h = (rem % (H_out * W_out)) // W_out
    w = rem % W_out
    
    # Load from pooled/interpolated tensor
    pool_offset = n * stride_pool_N + c * stride_pool_C + h * stride_pool_H + w * stride_pool_W
    pool_val = tl.load(pooled_ptr + pool_offset, 
                       mask=mask & (c < C_pool) & (h < H_out) & (w < W_out), 
                       other=0.0)
    
    # Load from concat tensor
    concat_c = c - C_pool
    concat_offset = n * stride_concat_N + concat_c * stride_concat_C + h * stride_concat_H + w * stride_concat_W
    concat_val = tl.load(concat_ptr + concat_offset, mask=mask & (c >= C_pool), other=0.0)
    
    # Select based on channel
    val = tl.where(c < C_pool, pool_val, concat_val)
    
    # Apply batch_norm + relu
    if bn_channels > 0:
        ch_idx = c
        mean = tl.load(bn_mean_ptr + ch_idx)
        var = tl.load(bn_var_ptr + ch_idx)
        weight = tl.load(bn_weight_ptr + ch_idx)
        bias = tl.load(bn_bias_ptr + ch_idx)
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
        inv_std = tl.rsqrt(var + eps)
        val = (val - mean) * inv_std * weight + bias
    
    # ReLU activation
    val = tl.where(val > 0, val, 0.0)
    
    # Store output
    out_offset = n * stride_out_N + c * stride_out_C + h * stride_out_H + w * stride_out_W
    tl.store(output_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def triton_fused_cat_bn_relu(
    pooled, concat, bn_mean, bn_var, bn_weight, bn_bias, output_shape, eps=0.001
):
    """
    Fused kernel: cat + batch_norm + relu
    All in a single kernel launch.
    """
    N, C_pool, H_out, W_out = pooled.shape
    N2, C_concat, _, _ = concat.shape
    _, C_out, _, _ = output_shape
    
    BLOCK_SIZE = 1024
    num_elements = N * C_out * H_out * W_out
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(output_shape, dtype=pooled.dtype, device=pooled.device)
    
    fused_kernel[(num_programs,)](
        pooled_ptr=pooled,
        concat_ptr=concat,
        output_ptr=output,
        N=N, C_pool=C_pool, C_concat=C_concat, H_out=H_out, W_out=W_out,
        bn_mean_ptr=bn_mean,
        bn_var_ptr=bn_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        stride_pool_N=pooled.stride(0),
        stride_pool_C=pooled.stride(1),
        stride_pool_H=pooled.stride(2),
        stride_pool_W=pooled.stride(3),
        stride_concat_N=concat.stride(0),
        stride_concat_C=concat.stride(1),
        stride_concat_H=concat.stride(2),
        stride_concat_W=concat.stride(3),
        stride_out_N=output.stride(0),
        stride_out_C=output.stride(1),
        stride_out_H=output.stride(2),
        stride_out_W=output.stride(3),
        bn_channels=bn_mean.shape[0],
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: max_pool2d -> interpolate -> cat -> batch_norm -> relu
    """
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    """
    Return the fused kernel function.
    """
    return triton_fused_cat_bn_relu