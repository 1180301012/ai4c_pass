import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    # Conv2D: conv input first, then weight (used in resnet10t patterns)
    conv_output = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    
    # BatchNorm (note the different parameter order)
    bn_output = torch.nn.functional.batch_norm(conv_output, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Addition: add batch_norm output to skip_connection (in_6 += tmp_6 pattern)
    final_output = skip_connection + bn_output
    
    return final_output

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    return (conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection)

@triton.jit
def fused_conv_bn_add_kernel_variant(
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
    BLOCK_SIZE_M: tl.constexpr,  # Number of programs to process C_out dimension
    BLOCK_SIZE_N: tl.constexpr,  # Number of programs to process spatial dimensions
    BLOCK_SIZE_K: tl.constexpr   # Number of programs to process C_in dimension
):
    # Get program ids for different dimensions
    pid_m = tl.program_id(0)  # C_out dimension
    pid_n = tl.program_id(1)  # Spatial (H, W) dimension
    pid_k = tl.program_id(2)  # C_in dimension
    
    # Calculate ranges for each dimension
    m_range = tl.arange(0, BLOCK_SIZE_M)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    m_mask = m_range < C_out
    k_mask = k_range < C_in
    
    # Compute current output channel
    m = pid_m * BLOCK_SIZE_M + m_range
    
    # Load BatchNorm parameters for current output channel
    running_mean = tl.load(running_mean_ptr + m, mask=m_mask, other=0.0)
    running_var = tl.load(running_var_ptr + m, mask=m_mask, other=1.0)
    bn_weight = tl.load(bn_weight_ptr + m, mask=m_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + m, mask=m_mask, other=0.0)
    
    # Precompute the normalization factors
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = bn_weight * inv_std
    bias = bn_bias - running_mean * bn_weight * inv_std
    
    # Process spatial locations
    h_idx = pid_n // ((H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    w_idx = pid_n % ((H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    spatial_block = n_range + pid_n * BLOCK_SIZE_N
    h_mask = spatial_block < H
    w_mask = spatial_block < W
    
    # Load skip connection values
    skip_ptrs = skip_ptr + m[:, None] * (H * W) + spatial_block[:, None] * W
    skip_vals = tl.load(skip_ptrs, mask=(m_mask[:, None] & h_mask[None, :] & w_mask[None, :]), other=0.0)
    
    # Process input channels for each output channel
    acc = tl.zeros((BLOCK_SIZE_M, H, W), dtype=tl.float16)
    
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_block = k_range + k
        k_block_mask = k_block < C_in
        
        # Load conv weights
        weight_ptrs = weight_ptr + m[:, None, None] * (C_in * 1 * 1) + k_block[None, :, None] * (1 * 1) + 0
        weight_vals = tl.load(weight_ptrs, mask=(m_mask[:, None, None] & k_block_mask[:, None, None]), other=0.0)
        
        # Load input values (exploit 1x1 kernel structure)
        input_ptrs = input_ptr + (slice(None), k_block[None, :], h_idx, w_idx)
        input_vals = tl.load(input_ptrs, mask=(k_block_mask[:, None] & h_mask[None, :] & w_mask[None, :]), other=0.0)
        
        # Compute convolution (using 1x1 kernel property)
        conv_vals = input_vals * weight_vals
        
        # Sum over input channels (C_in dimension)
        acc += tl.sum(conv_vals, axis=2)
    
    # Apply BatchNorm and add skip connection (order matters for precision)
    output_vals = acc * scale[:, None, None] + bias[:, None, None] + skip_vals
    
    # Store results
    output_ptrs = output_ptr + m[:, None] * (H * W) + spatial_block[:, None] * W
    tl.store(output_ptrs, output_vals, mask=(m_mask[:, None] & h_mask[None, :] & w_mask[None, :]))

@torch.fx.wrap
def fused_conv_bn_add_variant(input_tensor, weight_tensor, running_mean, running_var, bn_weight, bn_bias, skip_connection):
    # Get tensor dimensions
    N, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Calculate optimal block sizes
    MAX_BLOCK_SIZE = 1024
    BLOCK_SIZE_M = min(128, C_out)
    BLOCK_SIZE_N = min(MAX_BLOCK_SIZE // BLOCK_SIZE_M, H * W)
    BLOCK_SIZE_K = min(64, C_in)
    
    # Calculate grid size
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = ((H * W) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (C_in + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the fused kernel
    fused_conv_bn_add_kernel_variant[(grid_m, grid_n, grid_k)](
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
    return fused_conv_bn_add_variant