import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_layer_norm_kernel_32_64_48(
    # Input pointers
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    # Output pointers
    out_0_ptr, out_1_ptr,
    # Shapes
    B, C, H, W, norm_dim,
    # Strides for in_2
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    # Strides for in_3
    stride_in3_b, stride_in3_c, stride_in3_n,
    # Output strides
    stride_out0_b, stride_out0_n, stride_out0_c,
    stride_out1_b, stride_out1_h, stride_out1_w, stride_out1_c,
    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    EPS: tl.constexpr,
):
    # Get batch and channel indices
    batch_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    # Compute output position
    out_pos = batch_idx * stride_out0_b + n_idx * stride_out0_n
    
    # Load in_3 (skip connection) - shape [B, N, C]
    in3_offset = batch_idx * stride_in3_b + n_idx * stride_in3_c + tl.arange(0, C)
    in3_mask = in3_offset < batch_idx * stride_in3_b + stride_in3_c * stride_in3_n
    skip_val = tl.load(in_3_ptr + in3_offset, mask=in3_mask, other=0.0).to(tl.float32)
    
    # Compute gelu activation on in_2
    # in_2 shape: [B, C, H, W], we need to map output position to in_2 position
    h = n_idx // W
    w = n_idx % W
    
    # Load and activate in_2
    in2_offset = (batch_idx * stride_in2_b + 
                  tl.arange(0, C) * stride_in2_c + 
                  h * stride_in2_h + 
                  w * stride_in2_w)
    in2_mask = (batch_idx * stride_in2_b + 
                tl.arange(0, C) * stride_in2_c + 
                h * stride_in2_h + 
                w * stride_in2_w) < batch_idx * stride_in2_b + stride_in2_c * C + stride_in2_h * H + stride_in2_w * W
    x = tl.load(in_2_ptr + in2_offset, mask=in2_mask, other=0.0).to(tl.float32)
    
    # GELU approximation using error function
    cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    gelu_out = x * cdf
    
    # Add skip connection
    fused_sum = gelu_out + skip_val
    
    # Store intermediate output (tmp_10)
    tl.store(out_0_ptr + out_pos + tl.arange(0, C) * stride_out0_c, fused_sum, mask=tl.arange(0, C) < C)
    
    # Compute layer norm statistics
    mean = tl.sum(fused_sum, axis=0) / float(norm_dim)
    centered = fused_sum - mean
    var = tl.sum(centered * centered, axis=0) / float(norm_dim)
    
    # Compute normalized output
    inv_std = 1.0 / tl.sqrt(var + EPS)
    norm_out = (fused_sum - mean) * inv_std
    
    # Load weight and bias
    w_weight = tl.load(weight_ptr + tl.arange(0, C)).to(tl.float32)
    w_bias = tl.load(bias_ptr + tl.arange(0, C)).to(tl.float32)
    
    # Apply affine transform
    norm_out = norm_out * w_weight + w_bias
    
    # Reshape for out_1: view to [B, H, W, C]
    out1_offset = batch_idx * stride_out1_b + h * stride_out1_w * stride_out1_c + w * stride_out1_c + tl.arange(0, C) * 1
    out1_mask = batch_idx * stride_out1_b + h * stride_out1_w * stride_out1_c + w * stride_out1_c + tl.arange(0, C) < batch_idx * stride_out1_b + stride_out1_h * H * stride_out1_w * stride_out1_c
    tl.store(out_1_ptr + out1_offset, norm_out, mask=tl.arange(0, C) < C)


@torch.fx.wrap
def fused_kernel_wrapper_32_64_48(in_2, in_3, weight, bias, output_0, output_1, H, W, C, eps=1e-6):
    B = 1
    N = H * W
    
    # Compute strides
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w = in_2.stride()
    stride_in3_b, stride_in3_c, stride_in3_n = in_3.stride()
    stride_out0_b, stride_out0_n, stride_out0_c = output_0.stride()
    stride_out1_b, stride_out1_h, stride_out1_w, stride_out1_c = output_1.stride()
    
    # Grid configuration
    grid = (B, N, 1)
    
    fused_gelu_layer_norm_kernel_32_64_48[grid](
        in_2, in_3, weight, bias,
        output_0, output_1,
        B, C, H, W, C,
        stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
        stride_in3_b, stride_in3_c, stride_in3_n,
        stride_out0_b, stride_out0_n, stride_out0_c,
        stride_out1_b, stride_out1_h, stride_out1_w, stride_out1_c,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_C=C,
        EPS=eps,
    )
    
    return output_0, output_1


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern with C=32, H=64, W=48
    """
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 32, 64, 48)
    tmp_9 = tmp_8.view(1, 32, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 64, 48, 32)
    return tmp_10, tmp_12


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract the dimension parameters for C=32, H=64, W=48.
    """
    B, C, H, W = in_2.shape
    N = H * W
    output_0 = torch.empty(1, N, C, dtype=in_2.dtype, device=in_2.device)
    output_1 = torch.empty(1, H, W, C, dtype=in_2.dtype, device=in_2.device)
    
    return (in_0, in_1, in_2, in_3, output_0, output_1, H, W, C)


def replacement_func():
    return fused_kernel_wrapper_32_64_48