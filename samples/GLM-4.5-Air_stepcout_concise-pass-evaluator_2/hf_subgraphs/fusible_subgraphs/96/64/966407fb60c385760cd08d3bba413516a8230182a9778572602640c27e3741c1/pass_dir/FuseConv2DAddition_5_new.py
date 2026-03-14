import torch
import triton
import triton.language as tl

def pattern(conv_weight, context_layer, value_layer):
    """Pattern matching for conv2d followed by in-place addition"""
    tmp_1 = torch.conv2d(value_layer, conv_weight, None, (1, 1), (32, 0), (1, 1), 4)
    context_layer += tmp_1
    return context_layer

def replacement_args(conv_weight, context_layer, value_layer):
    """Extract arguments for the replacement kernel"""
    return (conv_weight, context_layer, value_layer)

@triton.jit
def optimized_add_kernel(
    context_ptr,
    value_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    seq_len: tl.constexpr,
    heads: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized Triton kernel for tensor addition"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Process in 2D grid for better GPU utilization
    m_range = min(BLOCK_SIZE_M, batch_size - pid_m * BLOCK_SIZE_M)
    n_range = min(BLOCK_SIZE_N, channels * seq_len * heads - pid_n * BLOCK_SIZE_N)
    
    for m in range(m_range):
        for n in range(n_range):
            batch_idx = pid_m * BLOCK_SIZE_M + m
            element_idx = pid_n * BLOCK_SIZE_N + n
            
            if batch_idx < batch_size and element_idx < channels * seq_len * heads:
                # Calculate flat indices
                context_idx = batch_idx * channels * seq_len * heads + element_idx
                value_idx = batch_idx * channels * seq_len * heads + element_idx
                
                # Load values and add
                context_val = tl.load(context_ptr + context_idx)
                value_val = tl.load(value_ptr + value_idx)
                result = context_val + value_val
                tl.store(context_ptr + context_idx, result)

@torch.fx.wrap
def optimized_add(context_layer, value_layer):
    """Wrapper for optimized tensor addition"""
    # Get tensor shapes
    batch_size, channels, seq_len, heads = context_layer.shape
    total_elements = channels * seq_len * heads
    
    # Set optimal block sizes for better GPU occupancy
    BLOCK_SIZE_M = 32    # Process 32 batches per thread
    BLOCK_SIZE_N = 1024  # Process 1024 elements per thread
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    optimized_add_kernel[(grid_m, grid_n)](
        context_layer,
        value_layer,
        batch_size,
        channels,
        seq_len,
        heads,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return context_layer

def replacement_func():
    """Return the optimized add function"""
    return optimized_add