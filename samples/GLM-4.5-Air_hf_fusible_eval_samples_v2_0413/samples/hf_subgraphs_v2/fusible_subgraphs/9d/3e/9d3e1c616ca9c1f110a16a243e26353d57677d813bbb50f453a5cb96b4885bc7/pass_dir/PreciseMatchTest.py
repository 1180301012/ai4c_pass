import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    """
    Match the exact computation sequence from the graph:
    1. Conv2D: in_6 -> [256] -> [1024] with kernel [1024, 256, 1, 1]
    2. BatchNorm: with parameters [in_0, in_1, in_3, in_2]
    3. Addition: tmp_6 += in_5
    4. Return: tmp_7 = tmp_6
    """
    # Exact Conv2D operation from the graph
    conv2d_output = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    
    # Exact BatchNorm operation from the graph (note parameter order!)
    tmp_6 = torch.nn.functional.batch_norm(conv2d_output, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Exact addition operation from the graph
    tmp_6 += skip_connection
    
    # Final assignment that gets returned
    tmp_7 = tmp_6
    
    # Return ONLY what the original function returns: (tmp_7,)
    return (tmp_7,)

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    return (conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection)

@triton.jit
def precise_fused_kernel(
    # Pointers to input tensors
    input_ptr,        # [1, 256, 24, 24] - conv_input (in_6)
    weight_ptr,       # [1024, 256, 1, 1] - conv_weight (in_4)
    running_mean_ptr, # [1024] - bn_running_mean (in_0)
    running_var_ptr,  # [1024] - bn_running_var (in_1)
    bn_weight_ptr,    # [1024] - bn_weight (in_3)
    bn_bias_ptr,      # [1024] - bn_bias (in_2)
    skip_ptr,         # [1, 1024, 24, 24] - skip_connection (in_5)
    
    # Pointer to output tensor
    output_ptr,       # [1, 1024, 24, 24] - tmp_7
    
    # Tensor dimensions
    N, C_in, C_out, H, W,
    
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # C_out dimension
    BLOCK_SIZE_N: tl.constexpr,  # Spatial dimensions (H*W)
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel dimension
    pid_n = tl.program_id(1)  # Spatial dimension
    
    # Calculate ranges
    m_range = tl.arange(0, BLOCK_SIZE_M)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    m_mask = m_range < C_out
    
    # Current output channel indices
    m = pid_m * BLOCK_SIZE_M + m_range
    
    # Load BatchNorm parameters
    running_mean = tl.load(running_mean_ptr + m, mask=m_mask, other=0.0)
    running_var = tl.load(running_var_ptr + m, mask=m_mask, other=1.0)
    bn_weight = tl.load(bn_weight_ptr + m, mask=m_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + m, mask=m_mask, other=0.0)
    
    # Precompute normalization factors
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = bn_weight * inv_std
    bias = bn_bias - running_mean * bn_weight * inv_std
    
    # Process spatial location
    spatial_idx = pid_n
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Process each output channel in the block
    for m_idx in m:
        if m_idx >= C_out:
            continue
            
        # Load conv weight for this output channel
        weight_offset = m_idx * C_in  # weight is [C_out, C_in, 1, 1]
        conv_val = 0.0
        
        # Sum over input channels for this spatial location
        for c_in in range(C_in):
            input_offset = spatial_idx * C_in + c_in  # input layout [N, C_in, H, W] -> flattened spatial first
            weight_val = tl.load(weight_ptr + weight_offset + c_in, other=0.0)
            input_val = tl.load(input_ptr + input_offset, other=0.0)
            conv_val += input_val * weight_val
        
        # Apply BatchNorm
        bn_val = conv_val * scale[m_idx - pid_m * BLOCK_SIZE_M] + bias[m_idx - pid_m * BLOCK_SIZE_M]
        
        # Add skip connection
        skip_val = tl.load(skip_ptr + m_idx * (H * W) + spatial_idx, other=0.0)
        output_val = bn_val + skip_val
        
        # Store result
        output_offset = m_idx * (H * W) + spatial_idx
        tl.store(output_ptr + output_offset, output_val)

@torch.fx.wrap
def precise_fused_function(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias, skip_connection):
    # Get tensor shapes
    N, C_in, H, W = conv_input.shape
    C_out = conv_weight.shape[0]
    
    # Flatten spatial dimensions for simpler indexing
    num_spatial = H * W
    
    # Block sizes
    BLOCK_SIZE_M = min(256, C_out)  # Output channels
    BLOCK_SIZE_N = min(1024, num_spatial)  # Spatial locations
    
    # Grid size
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (num_spatial + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Launch kernel
    precise_fused_kernel[(grid_m, grid_n)](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        skip_ptr=skip_connection,
        output_ptr=output,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return (output,)  # Return as tuple to match original return

def replacement_func():
    return precise_fused_function