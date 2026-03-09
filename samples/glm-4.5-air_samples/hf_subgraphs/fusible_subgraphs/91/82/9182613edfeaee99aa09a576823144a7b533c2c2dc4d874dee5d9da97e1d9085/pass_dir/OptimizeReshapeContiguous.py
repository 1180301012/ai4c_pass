import torch
import triton
import triton.language as tl

# Pattern matching function for reshape followed by contiguous
def pattern(in_1):
    tmp_0 = in_1.reshape(-1, 16, -1, 128)  # General reshape pattern
    tmp_4 = tmp_0.contiguous()
    return tmp_4

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel that combines reshape and contiguous
@triton.jit
def reshape_contiguous_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a block of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate memory offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create offsets within the block
    offsets_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for bounds checking
    mask_m = offsets_m < batch_size
    mask_n = offsets_n < 16 * seq_len * 128
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Reshape input to [batch_size, 16, seq_len, 128] layout in memory
    # We need to compute the flat index from the 4D layout
    flat_idx = offsets_m[:, None] * (16 * seq_len * 128) + offsets_n
    
    # Load data with proper reshaping applied at memory level
    input_data = tl.load(in_ptr + flat_idx, mask=mask, other=0.0)
    
    # Store contiguous output
    out_flat_idx = offsets_m[:, None] * (16 * seq_len * 128) + offsets_n
    tl.store(out_ptr + out_flat_idx, input_data, mask=mask)

@torch.fx.wrap
def optimized_reshape_contiguous(in_1):
    # Get input tensor metadata
    batch_size = in_1.shape[0]
    original_shape = in_1.shape
    
    # Determine target dimensions based on input
    if len(original_shape) == 5:  # Original input format
        # Input is [batch, ..., seq_len, head_dim] -> [batch, 16, seq_len, 128]
        target_head_dim = 128
        target_heads = 16
        seq_len = original_shape[3] if original_shape[3] == original_shape[4] else original_shape[4]
        
        # Calculate new shape
        new_shape = (batch_size, target_heads, seq_len, target_head_dim)
        total_elements = batch_size * target_heads * seq_len * target_head_dim
    else:
        # Already in correct shape, just make contiguous
        return in_1.contiguous()
    
    # Create output tensor
    out = torch.empty(new_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Calculate kernel launch configuration
    if seq_len > 128:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 128
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (16 * seq_len * 128 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Run the kernel
    reshape_contiguous_kernel[(num_blocks_m, num_blocks_n)](
        in_1,
        out,
        batch_size,
        seq_len,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return optimized_reshape_contiguous