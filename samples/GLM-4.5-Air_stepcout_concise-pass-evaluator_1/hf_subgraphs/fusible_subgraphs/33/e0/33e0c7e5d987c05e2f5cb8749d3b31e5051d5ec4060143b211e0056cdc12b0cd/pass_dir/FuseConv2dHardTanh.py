import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matches the exact model structure with Conv2D + HardTanh"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return (tmp_3, tmp_2)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the optimized kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv2d_hardtanh_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, tanh_input_ptr,
    conv_output_ptr, tanh_output_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Fused Conv2D + HardTanh kernel optimized for GPU performance"""
    
    # Get program ID for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute convolution output ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offset ranges for matrix operations
    conv_m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    conv_n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    conv_k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    conv_m_mask = conv_m_offsets < C_out
    conv_n_mask = conv_n_offsets < (C_in * H * W)
    conv_k_mask = conv_k_offsets < C_in
    
    # Compute spatial dimensions for convolution
    conv_h = H
    conv_w = W
    
    # Process Conv2D operation
    if conv_m_mask.any():
        # Load bias from global memory
        bias_values = tl.load(bias_ptr + conv_n_offsets[:1], mask=conv_m_mask[:1], other=0.0)
        
        # Initialize accumulator with bias
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator += bias_values
        
        # Load kernel weights and input data in tiles
        for k_offset in range(0, C_in, BLOCK_SIZE_K):
            k_offsets_block = k_offset + conv_k_offsets
            k_full_mask = (k_offsets_block < C_in) & conv_k_mask
            
            # Load weight tiles (C_out, C_in, KH, KW)
            weight_values = tl.load(
                weight_ptr + 
                ((conv_m_offsets[:, None] * C_in + k_offsets_block[None, :]) * 1 * 1),
                mask=conv_m_mask[:, None] & k_full_mask[None, :],
                other=0.0
            )
            
            # Load input tiles (B, C_in, H, W)
            input_data = tl.load(
                conv_input_ptr +
                (k_offsets_block[None, :] * conv_h * conv_w),
                mask=k_full_mask[None, :],
                other=0.0
            )
            
            # Convolution operation: matrix multiplication with 1x1 kernel
            accumulator += tl.dot(weight_values, input_data)
        
        # Store Conv2D output
        conv_output_base = conv_output_ptr + (
            conv_m_offsets[:, None] * conv_h * conv_w +
            conv_n_offsets[None, :]
        )
        
        tl.store(conv_output_base, accumulator, mask=conv_m_mask[:, None] & conv_n_mask[None, :])
    
    # Process HardTanh operation on different input stream
    if tl.program_id(1) == 0:  # First program set handles tanh
        total_elements = N * C_out * conv_h * conv_w
        start_idx = tl.program_id(0) * BLOCK_SIZE
        end_idx = min(start_idx + BLOCK_SIZE, total_elements)
        
        if start_idx < total_elements:
            # Load tanh input data
            tanh_offsets = start_idx + tl.arange(0, BLOCK_SIZE)
            tanh_mask = tanh_offsets < total_elements
            
            tanh_values = tl.load(tanh_input_ptr + tanh_offsets, mask=tanh_mask, other=0.0)
            
            # Apply HardTanh: max(0, min(6, x))  
            tanh_result = tl.where(tanh_values < 0.0, 0.0, tanh_values)
            tanh_result = tl.where(tanh_result > 6.0, 6.0, tanh_result)
            
            # Store HardTanh output
            tl.store(tanh_output_ptr + tanh_offsets, tanh_result, mask=tanh_mask)

@torch.fx.wrap
def fused_conv2d_hardtanh(in_0, in_1, in_2, in_3):
    """High-performance fused Conv2D + HardTanh kernel wrapper"""
    # Map parameters to semantic names for clarity
    bias = in_0
    weight = in_1
    conv_input = in_2
    tanh_input = in_3
    
    # Get input dimensions
    N, C_in, H, W = conv_input.shape
    C_out = weight.shape[0]
    
    # Create output tensors
    conv_output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    tanh_output = torch.empty_like(tanh_input)
    
    # Optimal block sizes for GPU architecture
    BLOCK_SIZE_M = 128  # Output channels per block
    BLOCK_SIZE_N = 32   # Input elements per block
    BLOCK_SIZE_K = 32   # Input channels per block
    BLOCK_SIZE = 1024   # Elements per program for HardTanh
    
    # Calculate grid dimensions
    conv_grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    conv_grid_n = (C_in * H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    tanh_grid = (N * C_out * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with appropriate grid
    fused_conv2d_hardtanh_kernel[(conv_grid_m, conv_grid_n, tanh_grid)](
        conv_input, weight, bias, tanh_input,
        conv_output, tanh_output,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE
    )
    
    return (tanh_output, conv_output)

def replacement_func():
    """Returns the optimized fused function"""
    return fused_conv2d_hardtanh