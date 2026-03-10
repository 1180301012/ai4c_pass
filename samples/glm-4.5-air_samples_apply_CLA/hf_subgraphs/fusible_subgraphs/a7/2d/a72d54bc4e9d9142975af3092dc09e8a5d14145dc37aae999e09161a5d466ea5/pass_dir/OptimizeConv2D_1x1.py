import torch
import triton
import triton.language as tl

# Let me try to match exactly what's in the graph
def pattern(x, weight):
    """Match conv2d operation with input and weight"""
    # The original: torch.conv2d(in_6, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    return torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
def conv2d_1x1_kernel(
    x_ptr, 
    weight_ptr, 
    out_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    C_IN_EXPR: tl.constexpr,
):
    """Simplified 1x1 convolution kernel"""
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking
    m_mask = m_offsets < N
    n_mask = n_offsets < C_out
    
    # Load weights: [C_OUT, C_IN] 
    weight_ptrs = weight_ptr + n_offsets[:, None] * C_IN_EXPR + tl.arange(0, C_IN_EXPR)[None, :]
    weight = tl.load(weight_ptrs, mask=n_mask[:, None], other=0.0)
    
    # For spatial position (0,0) - since it's 1x1 conv
    spatial_pos = 0  # Linear index for first spatial position
    
    # Compute output for this block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m in range(BLOCK_SIZE_M):
        if m_offsets[m] < N:
            for n in range(BLOCK_SIZE_N):
                if n_offsets[n] < C_out:
                    # Load input for this batch and spatial position
                    input_ptr = x_ptr + m_offsets[m] * C_IN_EXPR * H * W + spatial_pos
                    x_val = tl.load(input_ptr + tl.arange(0, C_IN_EXPR), 
                                  mask=tl.arange(0, C_IN_EXPR) < C_IN_EXPR, 
                                  other=0.0)
                    
                    # Compute dot product for 1x1 conv
                    result = tl.sum(x_val * weight[n, :])
                    acc[m, n] = result
    
    # Store results
    output_ptrs = out_ptr + \
        (m_offsets[:, None] * C_out + n_offsets[None, :]) * H * W + spatial_pos
    tl.store(output_ptrs, acc, mask=m_mask[:, None] and n_mask[None, :])

@torch.fx.wrap
def optimized_conv2d_1x1(x, weight):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Output shape: [N, C_out, H, W]
    out = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x.device)
    
    # Use block sizes that are power of 2
    BLOCK_SIZE_M = 64   # Batch dimension
    BLOCK_SIZE_N = 128  # Output channels dimension
    
    # Calculate grid (2D grid for batch and output channels)
    num_M = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_N = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch simplified kernel
    grid = (num_M, num_N)
    conv2d_1x1_kernel[grid](
        x, weight, out, N, C_in, H, W, C_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, C_in
    )
    
    return out

def replacement_func():
    return optimized_conv2d_1x1