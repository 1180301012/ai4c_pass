import torch
import triton
import triton.language as tl

# Pattern matching function for reshape operation
def pattern(in_1):
    # The input is 5D [batch, a, b, seq_len, head_dim] 
    # and we want to reshape to 4D [batch, a*b, seq_len, head_dim]
    batch_size = in_1.shape[0]
    new_shape = (batch_size, in_1.shape[1]*in_1.shape[2], in_1.shape[3], in_1.shape[4])
    tmp_0 = in_1.reshape(new_shape)
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel for reshape operation
@triton.jit
def reshape_contiguous_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    head_dim,
    a_dim,
    b_dim,
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
    total_elements = a_dim * b_dim * seq_len * head_dim
    mask_m = offsets_m < batch_size
    mask_n = offsets_n < total_elements
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Calculate 5D to 4D reshape indices
    # Input: [batch, a, b, seq_len, head_dim]
    # Output: [batch, a*b, seq_len, head_dim]
    
    # Calculate batch and output indices
    batch_idx = offsets_m[:, None]
    output_idx = offsets_n
    
    # Calculate coordinates from flat output index
    # For output [batch, combined_ab, seq_len, head_dim]
    head_idx = output_idx % head_dim
    seq_idx = (output_idx // head_dim) % seq_len
    combined_ab_idx = (output_idx // (head_dim * seq_len)) % (a_dim * b_dim)
    
    # Split combined_ab_idx back into a and b components
    a_idx = combined_ab_idx // b_dim  # Note: b_dim is the second dimension size
    b_idx = combined_ab_idx % b_dim
    
    # Calculate flat index for 5D input tensor [batch, a, b, seq_len, head_dim]
    input_5d_flat_idx = (batch_idx * (a_dim * b_dim * seq_len * head_dim) + 
                        a_idx * (b_dim * seq_len * head_dim) + 
                        b_idx * (seq_len * head_dim) + 
                        seq_idx * head_dim + 
                        head_idx)
    
    # Output flat index is straightforward - it's already in the right order
    output_4d_flat_idx = offsets_n
    
    # Load input data
    input_data = tl.load(in_ptr + input_5d_flat_idx, mask=mask, other=0.0)
    
    # Store output data
    tl.store(out_ptr + output_4d_flat_idx, input_data, mask=mask)

@torch.fx.wrap
def optimized_reshape(in_1):
    # Get input tensor metadata
    original_shape = in_1.shape
    
    if len(original_shape) == 5:  # 5D input [batch, a, b, seq_len, head_dim]
        # Parse dimensions from input shape
        batch_size = original_shape[0]
        a_dim = original_shape[1]
        b_dim = original_shape[2]
        seq_len = original_shape[3]
        head_dim = original_shape[4]
        
        # Target shape: [batch, a*b, seq_len, head_dim]
        new_shape = (batch_size, a_dim * b_dim, seq_len, head_dim)
        
        # Create output tensor
        out = torch.empty(new_shape, dtype=in_1.dtype, device=in_1.device)
        
        # Calculate kernel launch configuration based on actual dimensions
        if seq_len > 128:
            BLOCK_SIZE_M = 16
            BLOCK_SIZE_N = 128
        else:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 64
            
        num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_blocks_n = (a_dim * b_dim * seq_len * head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        # Run the optimized reshape kernel
        reshape_contiguous_kernel[(num_blocks_m, num_blocks_n)](
            in_1,
            out,
            batch_size,
            seq_len,
            head_dim,
            a_dim,
            b_dim,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N
        )
        
        return out
    else:
        # If already in the right shape, just return it
        return in_1

def replacement_func():
    return optimized_reshape