import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the full computation pattern: linear + dropout + transpose
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear  # Represent dropout output
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_fused_kernel(
    bias_ptr, weight_ptr, input_ptr,
    output_ptr, transposed_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # sequence dimension (M)
    pid_n = tl.program_id(1)  # output dimension (N)
    
    # Offsets within blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for boundaries
    m_mask = m_offset < S
    n_mask = n_offset < D
    
    # Load bias vector (this is broadcasted across sequence dimension)
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Accumulator in higher precision for accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over hidden dimension with proper blocking
    for k in range(0, H, BLOCK_SIZE_K):
        k_block = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_block < H
        
        # Load input and weight blocks with proper masking
        input_ptr_offset = (m_offset[:, None] * H + k_block[None, :]).to(tl.int64)
        weight_ptr_offset = (k_block[:, None] * D + n_offset[None, :]).to(tl.int64)
        
        input_block = tl.load(input_ptr + input_ptr_offset, 
                             mask=k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr + weight_ptr_offset, 
                              mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(input_block, weight_block)
    
    # Add bias (broadcasted)
    accumulator = accumulator + bias[None, :]
    
    # Convert to original data type based on input
    output_block = accumulator.to(tl.float16)  # Most models use float16
    
    # Apply dropout only if probability > 0
    if dropout_p > 0.0:
        # Generate good quality random numbers
        # Use thread indices + program ID for better distribution
        x_coord = m_offset[:, None]
        y_coord = n_offset[None, :]
        random_seed = ((pid_m * 73856093 + pid_n * 19349663) ^ 
                      (x_coord + y_coord * 81999) ^ 
                      (x_coord * y_coord * 123456789)) % 1000000
        
        # Linear Congruential Generator
        random_numbers = (random_seed * 1664525 + 1013904223) % (2**32) / (2**32)
        dropout_mask = random_numbers > dropout_p
        
        # Apply dropout and invert dropout for training
        output_block = output_block * dropout_mask.to(tl.float16)
        if dropout_p > 0.0:
            output_block = output_block * (1.0 / (1.0 - dropout_p))
    
    # Store output in [B, S, D] layout (B=1)
    output_base = pid_m * D + pid_n * S * D
    output_offsets = m_offset[:, None] * D + n_offset[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, 
             mask=m_mask[:, None] & n_mask[None, :])
    
    # Store transposed in [B, D, S] layout
    transposed_base = pid_n * S + pid_m * D * S
    transposed_offsets = n_offset[:, None] * S + m_offset[None, :]
    tl.store(transposed_ptr + transposed_base + transposed_offsets, output_block.T, 
             mask=n_mask[:, None] & m_mask[None, :])

@torch.fx.wrap
def optimized_fusion_wrapper(in_0, in_1, in_2):
    # Get tensor dimensions
    B, S, H = in_2.shape
    D = in_0.shape[0]
    
    # Optimized block sizes for A30 GPU characteristics
    # Balance between occupancy and memory bandwidth
    if S > 150 and D > 500:  # Large matrices typical of transformer models
        BLOCK_SIZE_M = 16  # Sequence dimension (smaller for better occupancy)
        BLOCK_SIZE_N = 64  # Output dimension
        BLOCK_SIZE_K = 32  # Hidden dimension (for loop unrolling)
    elif S > 80 or D > 200:  # Medium matrices
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 64
    else:  # Small matrices
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    
    # Create output tensors with same dtype as inputs
    output = torch.empty((B, S, D), device=in_2.device, dtype=in_2.dtype)
    transposed = torch.empty((B, D, S), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid dimensions
    num_blocks_m = (S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # For now, use dropout_p=0.0 (we can make this parameterized later)
    dropout_p = 0.01  # Small non-zero value to enable dropout logic
    
    # Launch the fused kernel
    optimized_fused_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        transposed_ptr=transposed,
        B=B, S=S, H=H, D=D,
        dropout_p=dropout_p,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output, transposed

def replacement_func():
    return optimized_fusion_wrapper