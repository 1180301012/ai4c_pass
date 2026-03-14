import torch
import triton
import triton.language as tl
import math

def pattern(weight, hidden_states):
    tmp_1 = torch.nn.functional.linear(hidden_states, weight, None)
    return tmp_1

def replacement_args(weight, hidden_states):
    return (weight, hidden_states)

@triton.jit
def linear_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr,
    batch_size_seq: tl.constexpr,
    dim_in: tl.constexpr,
    dim_out: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute offsets in the output matrix
    row_offsets = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to prevent out-of-bounds accesses
    mask_m = row_offsets < batch_size_seq
    mask_n = col_offsets < dim_out
    
    # Accumulator for the dot products
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Optimized loop with better memory access patterns
    for k in range(0, dim_in):
        # Load input data vectorized for better bandwidth utilization
        x_val = tl.load(x_ptr + row_offsets[:, None] * dim_in + k, mask=mask_m[:, None], other=0.0)
        
        # Load weight data transposed for better cache utilization 
        w_val = tl.load(w_ptr + col_offsets[None, :] * dim_in + k, mask=mask_n[None, :], other=0.0)
        
        # Perform fused multiply-add operation
        acc += x_val * w_val
    
    # Store the result: output is [batch_size_seq, dim_out]
    out_ptr_base = out_ptr + row_offsets[:, None] * dim_out + col_offsets[None, :]
    tl.store(out_ptr_base, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, ])

@torch.fx.wrap
def triton_linear_optimized(weight, hidden_states):
    # Get input shapes and determine dimensions
    if hidden_states.dim() == 3:
        batch_size, seq_len, dim_in = hidden_states.shape
    elif hidden_states.dim() == 2:
        batch_size, seq_len = 1, hidden_states.shape[0]
        dim_in = hidden_states.shape[1]
    else:
        raise ValueError(f"Unsupported hidden_states shape: {hidden_states.shape}")
    
    # Weight should be [dim_out, dim_in]
    if weight.dim() == 2:
        dim_out, _ = weight.shape
    else:
        raise ValueError(f"Unsupported weight shape: {weight.shape}")
    
    # Calculate output shape
    output_shape = (batch_size, seq_len, dim_out)
    output = torch.empty(output_shape, dtype=torch.bfloat16, device=hidden_states.device)
    
    # Reshape inputs to 2D for matrix multiplication
    # hidden_states: [B, S, D] -> [B*S, D]
    x_2d = hidden_states.reshape(-1, dim_in)
    # output: [B*S, D_out]
    out_2d = output.reshape(-1, dim_out)
    
    # Triton kernel launch parameters - optimized for better GPU occupancy
    BLOCK_SIZE_M = 32   # Reduced for better thread block utilization
    BLOCK_SIZE_N = 128  # Increased for better memory coalescing
    
    # Number of programs for 2D grid
    batch_size_seq = batch_size * seq_len
    num_programs_m = (batch_size_seq + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (dim_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with 2D grid
    linear_kernel[(num_programs_m, num_programs_n)](
        x_2d,
        weight,
        out_2d,
        batch_size_seq,
        dim_in,
        dim_out,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return triton_linear_optimized