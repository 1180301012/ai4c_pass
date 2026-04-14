import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Simple pattern: just linear operation (this definitely works)
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_linear_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)  # along sequence dimension
    pid_n = tl.program_id(1)  # along output dimension
    
    # Calculate offsets within blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundaries
    m_mask = m_offset < S
    n_mask = n_offset < D
    
    # Load bias vector [D]
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Accumulator in float32 for precision and reduced numerical errors
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over hidden dimension with optimized blocking
    for k in range(0, H, BLOCK_SIZE_M):
        # Get block offsets in k dimension
        k_block = tl.arange(0, BLOCK_SIZE_M) + k
        k_mask = k_block < H
        
        # Load input block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        input_ptr_offset = (m_offset[:, None] * H + k_block[None, :]).to(tl.int64)
        input_block = tl.load(input_ptr + input_ptr_offset, mask=k_mask[None, :], other=0.0)
        
        # Load weight block [BLOCK_SIZE_K, BLOCK_SIZE_N]  
        weight_ptr_offset = (k_block[:, None] * D + n_offset[None, :]).to(tl.int64)
        weight_block = tl.load(weight_ptr + weight_ptr_offset, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # High-performance matrix multiplication
        accumulator += tl.dot(input_block, weight_block, out_dtype=tl.float32)
    
    # Add bias with broadcasting
    accumulator = accumulator + bias[None, :]
    
    # Convert to target precision efficiently
    output_block = accumulator.to(tl.float16)
    
    # Store results with efficient stride calculation
    # Stride pattern: batch * (S * D + m_offset * D + n_offset)
    output_base = pid_m * D + pid_n * S * D
    output_offsets = m_offset[:, None] * D + n_offset[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, 
             mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap  
def optimized_linear_transform(in_0, in_1, in_2):
    # Extract dimensions from input tensors
    B, S, H = in_2.shape
    D = in_0.shape[0]  # bias dimension = output dimension
    
    # Adaptive block sizing for optimal GPU performance
    # Uses NVIDIA A30 GPU characteristics for optimal occupancy and memory bandwidth
    if S > 128 and D > 512:  # Large transformer matrices
        BLOCK_SIZE_M = 32   # Sequence dimension (smaller for better occupancy)
        BLOCK_SIZE_N = 64   # Output dimension
    elif S > 64 or D > 256:  # Medium matrices
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    else:  # Small matrices
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 64
    
    # Create output tensor with same properties as input
    output = torch.empty((B, S, D), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid dimensions for GPU launch
    num_blocks_m = (S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch highly optimized kernel
    optimized_linear_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        B=B, S=S, H=H, D=D,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_linear_transform