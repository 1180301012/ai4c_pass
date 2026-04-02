import torch
import triton
import triton.language as tl

def pattern(attention_weights):
    # Unsqueeze pattern for final output
    tmp_9 = attention_weights.unsqueeze(3)
    return tmp_9

def replacement_args(attention_weights):
    return (attention_weights,)

@triton.jit
def unsqueeze_kernel(
    input_ptr, 
    output_ptr,
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute indices
    batch_idx = tl.program_id(0)
    n_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_idx = tl.program_id(2)
    
    # Create masks for bounds checking
    n_mask = n_idx < N
    
    # Load input data: [B, N, K]
    input_idx = batch_idx * B * N * K + n_idx * K + k_idx
    input_val = tl.load(input_ptr + input_idx, mask=n_mask and k_idx < K, other=0.0)
    
    # Store output data: [B, N, K, 1] - add a new dimension
    # We need to handle the expanded dimension properly
    output_idx = batch_idx * B * N * K * 1 + n_idx * K * 1 + k_idx * 1 + 0
    tl.store(output_ptr + output_idx, input_val, mask=n_mask and k_idx < K)

@torch.fx.wrap
def optimized_unsqueeze(attention_weights):
    # Get tensor shapes
    B, N, K = attention_weights.shape  # [1, 4096, 32] from attention computation
    D_new = 1  # The new dimension we're adding
    
    # Create output tensor: [B, N, K, D_new]
    out = torch.zeros((B, N, K, D_new), dtype=attention_weights.dtype, device=attention_weights.device)
    
    # Block size for parallel computation  
    BLOCK_SIZE = 256  # Optimize for GPU warps
    
    # Calculate grid dimensions
    num_blocks_N = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    unsqueeze_kernel[(B, num_blocks_N, K)](
        attention_weights,
        out,
        B, N, K,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_unsqueeze