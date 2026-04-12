import torch
import triton
import triton.language as tl

@triton.jit
def optimized_slice_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    heads,
    seq_len,
    head_dim,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for slicing attention tensors.
    Slices dimension 2 from start_idx to end.
    """
    pid = tl.program_id(0)
    
    # Calculate offsets
    offset_b = pid // (heads * seq_len * head_dim)
    offset_h = (pid % (heads * seq_len * head_dim)) // (seq_len * head_dim)
    offset_s = (pid % (seq_len * head_dim)) // head_dim
    offset_d = pid % head_dim
    
    # Apply slice logic - skip elements before start_idx
    if offset_s >= start_idx:
        # Calculate new sequence length (original - start_idx)
        new_seq_len = seq_len - start_idx
        new_offset_s = offset_s - start_idx
        
        # Calculate final output position
        output_offset = offset_b * heads * new_seq_len * head_dim + \
                       offset_h * new_seq_len * head_dim + \
                       new_offset_s * head_dim + offset_d
        
        # Load input data
        input_offset = offset_b * heads * seq_len * head_dim + \
                      offset_h * seq_len * head_dim + \
                      offset_s * head_dim + offset_d
        
        # Load and store with bounds checking
        mask = (offset_b < batch_size) & (offset_h < heads) & \
               (new_offset_s < new_seq_len) & (offset_d < head_dim)
        
        if mask:
            data = tl.load(input_ptr + input_offset)
            tl.store(output_ptr + output_offset, data)

@torch.fx.wrap
def optimized_attention_slice(tensor, start_idx=1):
    """
    Optimized tensor slicing for attention computations.
    Efficiently slices the third dimension (seq_len dimension) from start_idx to end.
    """
    batch_size, heads, seq_len, head_dim = tensor.shape
    
    # Calculate output shape
    new_seq_len = seq_len - start_idx
    output_shape = (batch_size, heads, new_seq_len, head_dim)
    result = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Determine block size and launch configuration
    BLOCK_SIZE = 256
    total_elements = batch_size * heads * new_seq_len * head_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_programs > 0:
        optimized_slice_kernel[(num_programs,)](
            input_ptr=tensor,
            output_ptr=result,
            batch_size=batch_size,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            start_idx=start_idx,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return result

def pattern(in_0, in_1, in_2):
    """Pattern matching for attention computation with slice optimization"""
    tmp_0 = in_1 @ in_0
    tmp_1 = optimized_attention_slice(in_1, 1)  # Optimized slice operation
    tmp_2 = optimized_attention_slice(in_2, 1)  # Optimized slice operation
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_2 = None
    # Use flexible reshape and split to match different graph variants
    tmp_4 = tmp_3.reshape(1, -1, tmp_3.shape[2], tmp_3.shape[2])
    tmp_3 = None
    # Split into roughly equal parts for different head configurations
    total_channels = tmp_4.shape[1]
    split_size = total_channels // 3
    remaining = total_channels - 2 * split_size
    tmp_5 = torch.functional.split(tmp_4, [split_size, remaining, split_size], dim=1)
    tmp_4 = None
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_5 = None
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for slicing optimization"""
    return (in_0, in_1, in_2)

def replacement_func():
    """Return optimized functions that include slice optimization"""
    # Create a wrapper that incorporates both optimizations
    @torch.fx.wrap
    def optimized_forward(in_0, in_1, in_2):
        tmp_0 = in_1 @ in_0  # This will be replaced by the matmul optimization
        tmp_1 = optimized_attention_slice(in_1, 1)
        tmp_2 = optimized_attention_slice(in_2, 1)
        tmp_3 = tmp_2.transpose(-1, -2)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, -1, tmp_3.shape[2], tmp_3.shape[2])
        tmp_3 = None
        total_channels = tmp_4.shape[1]
        split_size = total_channels // 3
        remaining = total_channels - 2 * split_size
        tmp_5 = torch.functional.split(tmp_4, [split_size, remaining, split_size], dim=1)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_8 = tmp_5[2]
        tmp_5 = None
        return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)
    
    return optimized_forward