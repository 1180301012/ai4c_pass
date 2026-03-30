import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    tmp_0 = input_tensor / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_divide_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get flattened program ID
    pid_flat = tl.program_id(0)
    
    # Calculate grid dimensions
    grid_per_bh = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_per_b = grid_per_bh * ((head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Extract batch and head IDs from flattened program ID
    pid_b = pid_flat // grid_per_b
    pid_flat_remainder = pid_flat % grid_per_b
    pid_h = pid_flat_remainder // grid_per_bh
    pid_mn = pid_flat_remainder % grid_per_bh
    
    # Extract sequence and block IDs
    pid_m = pid_mn // ((head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    pid_n = pid_mn % ((head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Get thread IDs within the block
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate the sequence and head dimension this thread handles
    s_pos = m_offsets[:, None] + BLOCK_SIZE_M * pid_m
    d_pos = n_offsets[None, :] + BLOCK_SIZE_N * pid_n
    
    # Create mask for boundary conditions
    mask_s = s_pos < seq_len
    mask_d = d_pos < head_dim
    mask = mask_s & mask_d
    
    # Calculate input tensor offsets [batch, heads, seq, head]
    input_offsets = (pid_b * num_heads + pid_h) * seq_len * head_dim + \
                   s_pos * head_dim + d_pos
    
    # Calculate output tensor offsets [batch, heads, head, seq] (transposed)
    output_offsets = (pid_b * num_heads + pid_h) * head_dim * seq_len + \
                    d_pos * seq_len + s_pos
    
    # Load input data and apply division
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    output_vals = input_vals / scale
    
    # Store to transposed position
    tl.store(output_ptr + output_offsets, output_vals, mask=mask)

@torch.fx.wrap
def fused_divide_transpose(input_tensor):
    # Get tensor shape
    batch_size, num_heads, seq_len, head_dim = input_tensor.shape
    
    # Configure block sizes for optimal GPU performance
    BLOCK_SIZE_M = 32  # Process 32 sequence elements per block
    BLOCK_SIZE_N = 8   # Process 8 head_dim elements per block
    
    # Calculate grid dimensions
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_grid = batch_size * num_heads * grid_m * grid_n
    
    # Create output tensor with transposed dimensions
    output_tensor = torch.empty((batch_size, num_heads, head_dim, seq_len), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get scale from the computation pattern
    scale = 2.8284271247461903
    
    # Launch kernel with flattened 1D grid
    fused_divide_transpose_kernel[(total_grid,)](
        input_tensor,
        output_tensor,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output_tensor

def replacement_func():
    return fused_divide_transpose