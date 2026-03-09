import torch
import triton
import triton.language as tl

def pattern(in_0, in_3, in_2, in_1):
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    return tmp_3

def replacement_args(in_0, in_3, in_2, in_1):
    return (in_0, in_3, in_2, in_1)

@triton.jit
def fused_attention_arithmetic_kernel(
    x_ptr,
    y_ptr,  
    z_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    seq_len: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load main tensor data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate position for mask loading based on broadcasting pattern
    # For mask [batch, 1, 1, seq_len] -> load [batch, num_heads, seq_len, seq_len]
    # We need to map each offset to its corresponding mask value
    offset_idx = tl.arange(0, BLOCK_SIZE)
    
    # Extract coordinates: (batch, head, query_pos, key_pos)
    # Total elements per sequence: num_heads * seq_len * seq_len
    total_per_seq = seq_len * seq_len
    
    # Calculate which sequence (batch element) this offset belongs to
    seq_idx = offset_idx // total_per_seq
    remaining_idx = offset_idx % total_per_seq
    
    # Calculate head and positions within sequence
    head_idx = remaining_idx // (seq_len * seq_len)
    seq_pos_idx = remaining_idx % (seq_len * seq_len)
    query_pos = seq_pos_idx // seq_len
    key_pos = seq_pos_idx % seq_len
    
    # Calculate mask offset: for each batch, we want mask[batch, 0, 0, key_pos]
    # The mask has shape [batch, 1, 1, seq_len], so we need to flatten it appropriately
    mask_offset = seq_idx * seq_len + key_pos
    
    # Load mask values
    mask_val = tl.load(mask_ptr + mask_offset, mask=tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)
    
    # Fused computation: (x + y + z) * scale + mask_val
    # Note: division by 8.0 = multiplication by 0.125
    result = (x + y + z) * 0.125 + mask_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_attention_arithmetic(x, y, z, mask):
    # Determine total number of elements and grid size
    n_elements = x.numel()
    
    # Extract sequence length from tensor shape
    seq_len = x.shape[-1]
    
    BLOCK_SIZE = 1024  # Adjust based on GPU properties
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_attention_arithmetic_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        mask_ptr=mask,
        out_ptr=out,
        n_elements=n_elements,
        seq_len=seq_len,
        scale=0.125,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_attention_arithmetic