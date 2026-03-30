import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    """
    Pattern matching the computation graph:
    conv2d -> conv2d -> cat -> batch_norm -> silu -> mean
    """
    conv2d_1 = torch.conv2d(in_8, in_4, None)
    conv2d_2 = torch.conv2d(in_9, in_5, None)
    tmp_8 = torch.cat([in_6, in_7, conv2d_1, conv2d_2], 1)
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    return (tmp_10, tmp_11)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9)

@triton.jit
def fused_batch_norm_silu_kernel(
    input_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    spatial_size, num_channels,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    num_pid_c = tl.cdiv(num_channels, BLOCK_SIZE_C)
    num_pid_s = tl.cdiv(spatial_size, BLOCK_SIZE_S)
    
    if pid_c >= num_pid_c or pid_s >= num_pid_s:
        return
    
    # Compute channel and spatial offsets
    c_start = pid_c * BLOCK_SIZE_C
    s_start = pid_s * BLOCK_SIZE_S
    
    # Process channel block
    for c in range(c_start, min(c_start + BLOCK_SIZE_C, num_channels)):
        # Load batch norm parameters
        bn_mean = tl.load(bn_mean_ptr + c)
        bn_var = tl.load(bn_var_ptr + c)
        bn_weight = tl.load(bn_weight_ptr + c)
        bn_bias = tl.load(bn_bias_ptr + c)
        
        # Inverse standard deviation
        inv_std = 1.0 / tl.sqrt(bn_var + eps)
        
        # Process spatial positions
        for s in range(s_start, min(s_start + BLOCK_SIZE_S, spatial_size)):
            pos = c * spatial_size + s
            
            # Load input value
            x = tl.load(input_ptr + pos)
            
            # Batch normalization
            bn_val = (x - bn_mean) * inv_std
            bn_val = bn_val * bn_weight + bn_bias
            
            # SiLU activation
            silu_val = bn_val * (1.0 / (1.0 + tl.exp(-bn_val)))
            
            # Store result
            tl.store(output_ptr + pos, silu_val)

@torch.fx.wrap
def fused_conv_bn_silu_mean(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    """
    Fused implementation of conv2d + cat + batch_norm + silu + mean
    """
    # Get tensor shapes
    batch_size, c_out1, h_out1, w_out1 = in_8.shape
    c_out2 = in_5.shape[0]
    
    # Get input channel count
    c_in = in_4.shape[1]  # Input channels for first conv
    
    # Concatenated dimensions
    c_cat = in_6.shape[1] + in_7.shape[1] + c_out1 + c_out2
    h_cat, w_cat = in_6.shape[2], in_6.shape[3]
    
    # Determine output dimensions based on convolution parameters
    # For now, assume same spatial dimensions as inputs
    H_out, W_out = h_out1, w_out1
    
    # Create output tensors
    output = torch.zeros((batch_size, c_cat, h_cat, w_cat), dtype=in_8.dtype, device=in_8.device)
    mean_output = torch.zeros((batch_size, c_cat, 1, 1), dtype=in_8.dtype, device=in_8.device)
    
    # Block sizes for Triton kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Launch Triton kernel
    grid = lambda meta: (
        triton.cdiv(c_cat, meta['BLOCK_SIZE_M']),
        triton.cdiv(h_cat * w_cat, meta['BLOCK_SIZE_N'])
    )
    
    fused_conv_bn_silu_mean_kernel[grid](
        in_8, in_4, in_5, in_0, in_1, in_3, in_2,
        output, mean_output.view(-1),
        batch_size, c_in, h_out1, w_out1,
        c_out1, c_out2, h_out1, w_out1, h_out1, w_out1,
        h_cat, w_cat, c_cat,
        (1, 1), (3, 3), (1, 1),  # Conv1 parameters
        (1, 1), (4, 4), (1, 1),  # Conv2 parameters
        1e-5,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return (output, mean_output)

def replacement_func():
    return fused_conv_bn_silu_mean