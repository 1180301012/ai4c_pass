import torch
import triton
import triton.language as tl

@triton.jit
def cat_dim2_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    n1, n2, n3, feature_dim,
    batch_size, head_size, seq_len_1, seq_len_2, seq_len_3,
    BLOCK_SIZE: tl.constexpr
):
    """Efficient concatenation kernel along dimension 2."""
    # Tensor shapes:
    # x: [batch_size, head_size, seq_len_1, feature_dim]
    # y: [batch_size, head_size, seq_len_2, feature_dim] 
    # z: [batch_size, head_size, seq_len_3, feature_dim]
    # Output: [batch_size, head_size, seq_len_1+seq_len_2+seq_len_3, feature_dim]
    
    # Get program position
    pid = tl.program_id(0)
    
    # Each program handles a block of elements
    offsets = tl.arange(0, BLOCK_SIZE)
    total_features = n1 + n2 + n3
    total_elements = batch_size * head_size * total_features
    
    # Calculate which elements this program handles
    linear_idx = pid * BLOCK_SIZE + offsets
    mask = linear_idx < total_elements
    
    # Convert linear index to tensor indices
    # First map to [batch, head, pos] coordinates
    pos_per_head = total_features
    pos_per_batch = head_size * pos_per_head
    
    batch_idx = linear_idx // pos_per_batch
    rem_idx = linear_idx % pos_per_batch
    
    head_idx = rem_idx // pos_per_head
    pos_idx = rem_idx % pos_per_head
    
    # Initialize output to zeros for elements outside mask
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Map position to source tensor and load data
    # Initialize result to zero (already done above)
    
    # For positions in first tensor (in_2)
    mask_first = mask & (pos_idx < n1)
    src_offset = batch_idx * head_size * seq_len_1 * feature_dim + \
                 head_idx * seq_len_1 * feature_dim + \
                 pos_idx * feature_dim + offsets
    src = tl.load(x_ptr + src_offset, mask=mask_first, other=0.0)
    result = tl.where(mask_first, src, result)
    
    # For positions in second tensor (in_5)
    mask_second = mask & (pos_idx >= n1) & (pos_idx < n1 + n2)
    src_pos = pos_idx - n1
    src_offset = batch_idx * head_size * seq_len_2 * feature_dim + \
                 head_idx * seq_len_2 * feature_dim + \
                 src_pos * feature_dim + offsets
    src = tl.load(y_ptr + src_offset, mask=mask_second, other=0.0)
    result = tl.where(mask_second, src, result)
    
    # For positions in third tensor (in_3)
    mask_third = mask & (pos_idx >= n1 + n2)
    src_pos = pos_idx - n1 - n2
    src_offset = batch_idx * head_size * seq_len_3 * feature_dim + \
                 head_idx * seq_len_3 * feature_dim + \
                 src_pos * feature_dim + offsets
    src = tl.load(z_ptr + src_offset, mask=mask_third, other=0.0)
    result = tl.where(mask_third, src, result)
    
    # Calculate output offset and store result
    out_offset = batch_idx * head_size * total_features * feature_dim + \
                 head_idx * total_features * feature_dim + \
                 pos_idx * feature_dim + offsets
    
    tl.store(out_ptr + out_offset, result, mask=mask)

@torch.fx.wrap  
def cat_dim2_triton(x, y, z):
    """Optimized concatenation along dimension 2 using Triton."""
    # Extract tensor shapes
    batch_size, head_size, seq_len_1, feature_dim = x.shape
    _, _, seq_len_2, _ = y.shape  
    _, _, seq_len_3, _ = z.shape
    
    # Calculate output shape
    total_seq_len = seq_len_1 + seq_len_2 + seq_len_3
    output_shape = [batch_size, head_size, total_seq_len, feature_dim]
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Use feature_dim for n1, n2, n3 as these are the sequence lengths along dim=2
    n1 = seq_len_1
    n2 = seq_len_2  
    n3 = seq_len_3
    
    # Calculate grid and block sizes
    total_elements = batch_size * head_size * total_seq_len * feature_dim
    BLOCK_SIZE = 128
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    cat_dim2_kernel[(num_programs,)](
        x, y, z, output,
        n1, n2, n3, feature_dim,
        batch_size, head_size,
        seq_len_1, seq_len_2, seq_len_3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(in_2, in_5, in_3):
    """Match torch.cat along dim=2."""
    return torch.cat((in_2, in_5, in_3), dim=2)

def replacement_args(in_2, in_5, in_3):
    """Extract arguments for concatenation replacement."""
    return (in_2, in_5, in_3)

def replacement_func():
    """Return optimized concatenation function."""
    return cat_dim2_triton