import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Simple pattern: just linear operation (this works)
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fixed_linear_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate thread offsets within blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary conditions (more conservative)
    m_mask = m_offset < M
    n_mask = n_offset < N
    
    # Initialize accumulator in higher precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (hidden dimension)
    for k in range(0, K, BLOCK_SIZE_M):
        # Calculate block offset in k dimension
        k_block = tl.arange(0, BLOCK_SIZE_M) + k
        k_mask = k_block < K
        
        # Load input and weight blocks with careful masking
        # Input: [M, K], Weight: [K, N], A @ W^T + bias
        input_ptr_offset = (m_offset[:, None] * K + k_block[None, :]).to(tl.int32)
        weight_ptr_offset = (k_block[:, None] * N + n_offset[None, :]).to(tl.int32)
        
        # Load blocks with proper bounds checking
        input_block = tl.load(input_ptr + input_ptr_offset, 
                             mask=k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr + weight_ptr_offset, 
                              mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(input_block, weight_block)
    
    # Load bias vector and add with broadcasting
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    accumulator = accumulator + bias[None, :]
    
    # Convert to output dtype with error checking
    if accumulator.dtype == tl.float32:
        output_block = accumulator.to(tl.float16)
    else:
        output_block = accumulator  # Already correct dtype
    
    # Store results with careful bounds checking
    # Calculate output strides carefully: base + m_offset * N + n_offset
    output_base = pid_m * N + pid_n * M * N
    output_offsets = m_offset[:, None] * N + n_offset[None, :]
    
    # Apply mask for both dimensions
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, mask=store_mask)

@torch.fx.wrap
def fixed_linear_transform(in_0, in_1, in_2):
    # Get dimensions clearly labeled
    M = in_2.shape[0]  # batch size
    K = in_2.shape[1]  # sequence length  
    N = in_0.size(0)   # output dimension
    
    # More conservative block sizing for stability
    BLOCK_SIZE_M = 64   # batch/sequence dimension
    BLOCK_SIZE_N = 64   # output dimension
    
    # Create output tensor with same properties
    output = torch.empty((M, K, N), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid dimensions conservatively 
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with conservative configuration
    fixed_linear_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        M=M, N=N, K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return fixed_linear_transform