import torch
import triton
import triton.language as tl

@triton.jit
def simple_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    input_dim,
    output_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Each thread computes one output position
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load bias for the output block
    bias_block = tl.load(bias_ptr + n_offsets)
    
    # Compute output base position
    output_base = batch_idx * seq_len * output_dim + seq_idx * output_dim
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input dimension
    for k_block in tl.range(0, input_dim, BLOCK_SIZE_K):
        # Load input block
        x_block = tl.load(x_ptr + batch_idx * seq_len * input_dim + seq_idx * input_dim + k_block + m_offsets)
        
        # Load weight block
        weight_block = tl.load(weight_ptr + (k_block + n_offsets) * input_dim)
        
        # Update accumulator
        acc += x_block[:, None] * weight_block[None, :]
    
    # Add bias
    acc += bias_block[None, :]
    
    # Store results
    output_offsets = output_base + n_offsets[None, :] + m_offsets[:, None] * output_dim
    tl.store(out_ptr + output_offsets, acc)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """Simple optimized linear operation"""
    batch_size, seq_len, input_dim = x.shape
    output_dim = bias.shape[0]
    
    output = torch.empty(batch_size, seq_len, output_dim, dtype=x.dtype, device=x.device)
    
    # Use reasonable block sizes
    BLOCK_SIZE_M = 1  # Process one output position at a time
    BLOCK_SIZE_N = 64  # Process 64 output features per thread
    BLOCK_SIZE_K = 32  # Process 32 input features per iteration
    
    # Grid size: one program per (batch, seq) pair
    grid_size = batch_size * seq_len
    
    simple_linear_kernel[(grid_size, 1, 1)](
        x, weight, bias, output,
        batch_size, seq_len, input_dim, output_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def pattern(in_3, in_2, in_1):
    """
    Simple test pattern - just the linear operation
    """
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    return linear

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

def replacement_func():
    return optimized_linear