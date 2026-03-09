import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern matching for mean operation: input_tensor.mean(-2)"""
    output = input_tensor.mean(-2)
    return output

def replacement_args(input_tensor):
    """Extract arguments for the optimized kernel"""
    return (input_tensor,)

@triton.jit
def mean_kernel(
    input_ptr,    # [batch_size, seq_len, features]
    output_ptr,   # [batch_size, features]
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE_M: tl.constexpr,  # batch_size tile
    BLOCK_SIZE_N: tl.constexpr,  # features tile
):
    """Optimized mean reduction kernel using Triton"""
    # Get program indices
    m = tl.program_id(0)  # batch dimension
    n = tl.program_id(1)  # feature dimension
    
    # Compute memory addresses
    input_base_ptr = input_ptr + m * seq_len * features + n
    output_ptr_base = output_ptr + m * features + n
    
    # Initialize accumulator
    acc = 0.0
    
    # Sequential loop over sequence length with vectorized feature access
    for k in range(seq_len):
        # Load feature value for this sequence position
        val = tl.load(input_base_ptr + k * features)
        acc += val
    
    # Compute mean by dividing by sequence length
    mean_val = acc / seq_len
    
    # Store result
    tl.store(output_ptr_base, mean_val)

@torch.fx.wrap
def optimized_mean(input_tensor):
    """Wrapper function to launch the optimized mean kernel"""
    batch_size, seq_len, features = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty(batch_size, features, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose optimal block sizes
    BLOCK_SIZE_M = min(32, triton.next_power_of_2(batch_size))
    BLOCK_SIZE_N = min(32, triton.next_power_of_2(features))
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    mean_kernel[(grid_m, grid_n)](
        input_tensor,
        output,
        batch_size,
        seq_len,
        features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Return the optimized mean function"""
    return optimized_mean