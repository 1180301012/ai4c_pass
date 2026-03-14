import torch
import triton
import triton.language as tl

# Pattern matching function - matches the duplicate addition and transpose pattern
def pattern(in_0, in_1):
    """
    Match the computation pattern with duplicate additions and transposes.
    The pattern has:
    - Reshape in_1 to [1, 64, -1]
    - Two identical additions: in_0 + tmp_0 (compute once and reuse)
    - Two identical transposes on the addition results
    - One transpose on in_0
    Returns all three outputs in the original order.
    """
    tmp_0 = in_1.reshape(1, 64, -1)
    tmp_1 = in_0 + tmp_0
    tmp_2 = in_0 + tmp_0
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)
    tmp_5 = in_0.transpose(0, 1)
    return tmp_4, tmp_3, tmp_5


# Extract arguments needed for replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Autotune configurations for optimal performance across different tensor sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr, in_1_ptr, 
    out_4_ptr, out_3_ptr, out_5_ptr,
    in_0_batch, in_0_seq, in_0_dim,
    in_1_batch, in_1_seq, in_1_dim,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Reshape in_1 from [64, 2, 128] to [1, 64, 256]
    2. Add in_0 + reshaped_in_1 (with broadcasting)
    3. Transpose the result (0, 1)
    4. Also transpose in_0 directly
    
    Uses 1D grid with tiling for simplicity and correctness.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute indices for output tensor [in_0_seq, in_0_batch, in_0_dim]
    # We process the transposed output directly
    out_batch_size = in_0_batch * in_0_dim
    
    # Output indices: seq_idx = offsets // (in_0_batch * in_0_dim)
    #                 remainder = offsets % (in_0_batch * in_0_dim)
    #                 batch_idx = remainder // in_0_dim
    #                 dim_idx = remainder % in_0_dim
    seq_idx = offsets // out_batch_size
    remainder = offsets % out_batch_size
    batch_idx = remainder // in_0_dim
    dim_idx = remainder % in_0_dim
    
    # Compute offset in original in_0 tensor: [batch, seq, dim]
    # This is for loading in_0 and computing in_0 + tmp_0
    in_0_offsets = batch_idx * in_0_seq * in_0_dim + seq_idx * in_0_dim + dim_idx
    
    # Load in_0 values
    in_0 = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=0.0)
    
    # Compute tmp_0 value: reshape in_1 [64, 2, 128] -> [1, 64, 256]
    # tmp_0[0, seq_idx, dim_idx] = in_1[seq_idx // 2, seq_idx % 2, dim_idx]
    # in_1_shape is [in_1_batch, in_1_seq, in_1_dim] = [64, 2, 128]
    in_1_s = seq_idx // in_1_seq  # seq_idx // 2
    in_1_r = seq_idx % in_1_seq   # seq_idx % 2
    in_1_offsets = in_1_s * in_1_seq * in_1_dim + in_1_r * in_1_dim + dim_idx
    
    tmp_0 = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0)
    
    # Compute add_result = in_0 + tmp_0 (broadcasting)
    add_result = in_0 + tmp_0
    
    # Store transposed results to out_4 and out_3 (identical)
    # The output is already in transposed layout [seq, batch, dim]
    tl.store(out_4_ptr + offsets, add_result, mask=mask)
    tl.store(out_3_ptr + offsets, add_result, mask=mask)
    
    # Store transposed in_0 to out_5
    tl.store(out_5_ptr + offsets, in_0, mask=mask)


@torch.fx.wrap
def fused_add_transpose_wrapper(in_0, in_1):
    """
    Wrapper function that uses Triton kernel to compute the fused operation.
    Optimizations:
    1. Compute addition once instead of twice
    2. Compute transpose once instead of twice  
    3. Reuse the same transposed result for both output positions
    """
    # Get shapes - convert to list to avoid FX tracing issues
    in_0_shape = in_0.shape
    in_1_shape = in_1.shape
    
    # Extract individual dimensions to avoid tuple tracing issues
    in_0_batch = in_0_shape[0]
    in_0_seq = in_0_shape[1]
    in_0_dim = in_0_shape[2]
    in_1_batch = in_1_shape[0]
    in_1_seq = in_1_shape[1]
    in_1_dim = in_1_shape[2]
    
    # Output shapes after transpose(0, 1): [in_0_seq, in_0_batch, in_0_dim]
    out_4_shape = (in_0_seq, in_0_batch, in_0_dim)
    out_3_shape = out_4_shape
    out_5_shape = out_4_shape
    
    # Total elements in output
    total_elements = in_0_seq * in_0_batch * in_0_dim
    
    # Get dtype and device - use torch API to avoid tracing issues
    dtype = in_0.dtype
    device = in_0.device
    
    # Allocate output tensors using empty_like for correct shape and device
    # We'll overwrite the values anyway
    out_4 = torch.empty(in_0_seq, in_0_batch, in_0_dim, dtype=dtype, device=device)
    out_3 = torch.empty(in_0_seq, in_0_batch, in_0_dim, dtype=dtype, device=device)
    out_5 = torch.empty(in_0_seq, in_0_batch, in_0_dim, dtype=dtype, device=device)
    
    # Calculate grid
    grid = (total_elements,)
    
    # Launch kernel
    fused_add_transpose_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_4_ptr=out_4,
        out_3_ptr=out_3,
        out_5_ptr=out_5,
        in_0_batch=in_0_batch,
        in_0_seq=in_0_seq,
        in_0_dim=in_0_dim,
        in_1_batch=in_1_batch,
        in_1_seq=in_1_seq,
        in_1_dim=in_1_dim,
        total_elements=total_elements,
    )
    
    return out_4, out_3, out_5


def replacement_func():
    return fused_add_transpose_wrapper