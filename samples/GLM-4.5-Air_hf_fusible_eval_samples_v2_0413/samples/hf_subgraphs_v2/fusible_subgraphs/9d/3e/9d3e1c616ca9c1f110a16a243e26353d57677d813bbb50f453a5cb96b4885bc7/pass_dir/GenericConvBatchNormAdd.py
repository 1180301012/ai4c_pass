import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    """
    Generic pattern that matches Conv2D + BatchNorm + Add operations
    with flexible argument ordering to handle different graph structures
    """
    # Conv2D with 1x1 kernel (parameters match all graph variations)
    conv_output = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (0, 0), (1, 1), 1)
    
    # BatchNorm (using the parameter order that appears most consistently)
    bn_output = torch.nn.functional.batch_norm(conv_output, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Addition with skip connection (both orderings should work due to commutativity)
    # Addition is commutative, so tmp_6 += in_5 is equivalent to in_5 += tmp_6
    final_output = bn_output + skip_connection
    
    return final_output

def replacement_args(input_tensor, weight_tensor, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    return (input_tensor, weight_tensor, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection)

@triton.jit
def generic_fused_conv_bn_add_kernel(
    # Input pointers
    input_ptr,      # [N, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1]
    running_mean_ptr,  # [C_out]
    running_var_ptr,    # [C_out]
    bn_weight_ptr,  # [C_out]
    bn_bias_ptr,    # [C_out]
    skip_ptr,       # [N, C_out, H, W]
    
    # Output pointer
    output_ptr,     # [N, C_out, H, W]
    
    # Tensor sizes
    N, C_in, C_out, H, W,
    
    # Triton constants
    BLOCK_SIZE_M: tl.constexpr,  # Output channels
    BLOCK_SIZE_N: tl.constexpr,  # Spatial dimensions
    BLOCK_SIZE_K: tl.constexpr   # Input channels
):
    # Get program ids
    pid_m = tl.program_id(0)  # C_out dimension
    pid_n = tl.program_id(1)  # Spatial (H, W) dimension
    pid_k = tl.program_id(2)  # C_in dimension
    
    # Calculate ranges
    m_range = tl.arange(0, BLOCK_SIZE_M)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    # Masks for bounds checking
    m_mask = m_range < C_out
    k_mask = k_range < C_in
    
    # Current output channel indices
    m = pid_m * BLOCK_SIZE_M + m_range
    
    # Current spatial location
    spatial_idx = pid_n
    
    # Load BatchNorm parameters for this output channel group
    running_mean = tl.load(running_mean_ptr + m, mask=m_mask, other=0.0)
    running_var = tl.load(running_var_ptr + m, mask=m_mask, other=1.0)
    bn_weight = tl.load(bn_weight_ptr + m, mask=m_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + m, mask=m_mask, other=0.0)
    
    # Precompute normalization factors
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = bn_weight * inv_std
    bias = bn_bias - running_mean * bn_weight * inv_std
    
    # Calculate spatial location indices
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load skip connection values for this spatial location
    skip_ptrs = skip_ptr + m * (H * W) + spatial_idx
    skip_vals = tl.load(skip_ptrs, mask=m_mask, other=0.0)
    
    # Process input channels in blocks
    conv_val = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float16)
    
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_block = k_range + k
        k_block_mask = k_block < C_in
        
        # Load weights for current output channel group and input channel block
        weight_ptrs = weight_ptr + m[:, None] * C_in + k_block
        weight_vals = tl.load(weight_ptrs, mask=(m_mask[:, None] & k_block_mask[None, :]), other=0.0)
        
        # Load input values for this spatial location and input channel block
        input_ptrs = input_ptr + (slice(None), k_block) + h_idx * (W * C_in) + w_idx
        input_vals = tl.load(input_ptrs, mask=k_mask, other=0.0)
        
        # Convolution operation (sum over input channels)
        conv_val += input_vals * tl.sum(weight_vals, axis=1)
    
    # Apply BatchNorm and add skip connection
    output_vals = conv_val * scale + bias + skip_vals
    
    # Store results
    output_ptrs = output_ptr + m * (H * W) + spatial_idx
    tl.store(output_ptrs, output_vals, mask=m_mask)

@torch.fx.wrap
def generic_fused_conv_bn_add(input_tensor, weight_tensor, running_mean, running_var, bn_weight, bn_bias, skip_connection):
    # Get tensor dimensions
    N, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Calculate number of spatial locations
    num_spatial = H * W
    
    # Optimal block sizes
    BLOCK_SIZE_M = min(128, C_out)  # Output channels
    BLOCK_SIZE_N = min(1024, num_spatial)  # Spatial locations
    BLOCK_SIZE_K = min(64, C_in)  # Input channels
    
    # Calculate grid size (2D grid for output channels and spatial locations)
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (num_spatial + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (C_in + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with 3D grid (output channels, spatial locations, input channels)
    generic_fused_conv_bn_add_kernel[(grid_m, grid_n, grid_k)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        skip_ptr=skip_connection,
        output_ptr=output,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return generic_fused_conv_bn_add