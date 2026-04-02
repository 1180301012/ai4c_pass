import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation sequence
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused convolution + sigmoid + multiply + hardtanh
@triton.jit
def fused_conv_activation_kernel(
    # Conv2D input and weights
    x_ptr,                      # Input tensor [N, C_in, H_in, W_in]
    weight_ptr,                # Weight tensor [C_out, C_in, KH, KW]
    bias_ptr,                  # Bias tensor [C_out]
    scale_ptr,                 # Scaling tensor [N, C_out, 1, 1]
    output_ptr,                # Output tensor [N, C_out, H_out, W_out]
    
    # Conv2D parameters
    N, C_in, H_in, W_in,
    C_out, KH, KW,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    
    # Activation parameters
    hardtanh_min, hardtanh_max,
    
    # Metadata
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial locations
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Spatial location block
    
    # Range of output channels this program handles
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, C_out)
    
    # Range of spatial locations this program handles
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, H_out * W_out)
    
    # Initialize accumulator
    accumulator = tl.zeros((m_end - m_start), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, C_in, groups):
        # Compute base pointers
        weight_base = weight_ptr + m_start * C_in * KH * KW + k * KH * KW
        bias_base = bias_ptr + m_start
        
        # Load bias for this output channel block
        bias = tl.load(bias_base, mask=(m_start < C_out), other=0.0)
        
        # Process spatial locations in this block
        for n in range(n_start, n_end):
            # Convert linear n to spatial coordinates
            h_out = n // W_out
            w_out = n % W_out
            
            # Compute input coordinates
            h_in = h_out * stride_h - pad_h
            w_in = w_out * stride_w - pad_w
            
            # Only process valid spatial locations
            if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                continue
            
            # Base pointers for this spatial location
            x_base = x_ptr + k * H_in * W_in + h_in * W_in + w_in
            weight_spatial_base = weight_base + h_in * W_in + w_in
            scale_base = scale_ptr + m_start + m_end
            
            # Load input spatial patches
            x_vals = tl.load(x_ptr + k * H_in * W_in + h_in * W_in + w_in, mask=True, other=0.0)
            weight_vals = tl.load(weight_spatial_base, mask=True, other=0.0)
            scale_vals = tl.load(scale_ptr + m_start + m_end, mask=True, other=1.0)
            
            # Compute convolution for this spatial location
            conv_val = tl.sum(x_vals * weight_vals) + bias
            
            # Apply sigmoid activation
            sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
            
            # Apply scaling multiplication
            scaled_val = sigmoid_val * scale_vals
            
            # Apply hardtanh
            hardtanh_val = tl.maximum(tl.minimum(scaled_val, hardtanh_max), hardtanh_min)
            
            # Store result
            if h_out < H_out and w_out < W_out:
                spatial_idx = h_out * W_out + w_out
                output_base = output_ptr + m_start * H_out * W_out + spatial_idx
                tl.store(output_base + (0 * H_out * W_out hardtanh_val, mask=(spatial_idx < H_out * W_out))
    
    # Synchronize threads
    tl.debug_barrier()

# Wrapper function that handles different data types and tensor configurations
@torch.fx.wrap
def fused_conv_activation(in_0, in_1, in_2, in_3):
    # Get input shapes and data types
    N, C_in, H_in, W_in = in_2.shape      # Main input tensor (scaling factor)
    N2, C_out, _, _ = in_3.shape         # Conv input tensor
    
    # Create output tensor
    output = torch.empty((N, C_out, H_in, W_in), dtype=in_2.dtype, device=in_2.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 64   # Output channels per block
    BLOCK_SIZE_N = 1024 # Spatial locations per block
    
    num_blocks_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (H_in * W_in + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with appropriate data type specialization
    if in_2.dtype == torch.float32:
        fused_conv_activation_kernel[(num_blocks_m, num_blocks_n)](
            in_3, in_1, in_0, in_2,
            N, C_in, H_in, W_in,
            C_out, 1, 1, H_in, W_in,
            1, 1, 0, 0, 1, 1, 1,
            0.0, 6.0,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )
    else:
        # For float16/bfloat16, we can use the same kernel but with appropriate types
        fused_conv_activation_kernel[(num_blocks_m, num_blocks_n)](
            in_3, in_1, in_0, in_2,
            N, C_in, H_in, W_in,
            C_out, 1, 1, H_in, W_in,
            1, 1, 0, 0, 1, 1, 1,
            0.0, 6.0,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )
    
    return output

# Replacement function - must return a function reference 
def replacement_func():
    return fused_conv_activation