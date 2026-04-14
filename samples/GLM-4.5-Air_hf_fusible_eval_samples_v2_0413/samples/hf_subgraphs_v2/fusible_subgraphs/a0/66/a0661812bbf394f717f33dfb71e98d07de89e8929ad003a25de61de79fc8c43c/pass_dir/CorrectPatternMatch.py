import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the EXACT computation pattern including dropout with p=0.0
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)  # Dropout with p=0.0 is no-op
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_fused_kernel(
    bias_ptr, weight_ptr, input_ptr,
    output_ptr, transposed_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program IDs for 2D grid
    pid_m = tl.program_id(0)  # sequence dimension
    pid_n = tl.program_id(1)  # output dimension
    
    # Calculate thread offsets within blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create boundary masks
    m_mask = m_offset < S
    n_mask = n_offset < D
    
    # Load bias vector
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Initialize accumulator in float32 for precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over hidden dimension with blocking for memory efficiency
    for k in range(0, H, BLOCK_SIZE_M):
        # Calculate block offset in k dimension
        k_block = tl.arange(0, BLOCK_SIZE_M) + k
        k_mask = k_block < H  # Mask for boundary conditions
        
        # Load blocks with proper stride calculation and masking
        input_ptr_offset = m_offset[:, None] * H + k_block[None, :]
        weight_ptr_offset = k_block[:, None] * D + n_offset[None, :]
        
        input_block = tl.load(input_ptr + input_ptr_offset, 
                             mask=k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr + weight_ptr_offset, 
                              mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication with precise accumulation
        accumulator += tl.dot(input_block, weight_block)
    
    # Add bias (broadcasted across sequence dimension)
    accumulator = accumulator + bias[None, :]
    
    # Convert to original data type (float16 or bfloat16)
    output_block = accumulator.to(tl.float16)
    
    # Store results in original layout [B, S, D]
    output_base = pid_m * D + pid_n * S * D
    output_offsets = m_offset[:, None] * D + n_offset[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, 
             mask=m_mask[:, None] & n_mask[None, :])
    
    # Store transposed results in [B, D, S] layout
    transposed_base = pid_n * S + pid_m * D * S
    transposed_offsets = n_offset[:, None] * S + m_offset[None, :]
    tl.store(transposed_ptr + transposed_base + transposed_offsets, output_block.T, 
             mask=n_mask[:, None] & m_mask[None, :])

@torch.fx.wrap
def optimized_fusion_wrapper(in_0, in_1, in_2):
    # Extract tensor dimensions
    B, S, H = in_2.shape
    D = in_0.shape[0]  # bias dimension
    
    # Adaptive block sizing based on problem characteristics
    if S > 150 and D > 512:  # Large transformer-like matrices
        BLOCK_SIZE_M = 32   # Sequence dimension (smaller for GPU occupancy)
        BLOCK_SIZE_N = 64   # Output dimension
    elif S > 80 or D > 256:  # Medium size matrices
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
    else:  # Small matrices
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
    
    # Create output tensors with same properties as inputs
    output = torch.empty((B, S, D), device=in_2.device, dtype=in_2.dtype)
    transposed = torch.empty((B, D, S), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid dimensions for optimal GPU utilization
    num_blocks_m = (S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Execute the fused kernel
    optimized_fused_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        transposed_ptr=transposed,
        B=B, S=S, H=H, D=D,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output, transposed

def replacement_func():
    return optimized_fusion_wrapper