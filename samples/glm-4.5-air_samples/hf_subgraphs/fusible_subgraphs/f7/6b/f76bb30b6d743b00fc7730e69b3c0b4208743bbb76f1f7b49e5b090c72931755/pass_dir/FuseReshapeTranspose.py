import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern: reshape(16, -1, 64) -> transpose(1, 2)
    return x.reshape(16, -1, 64).transpose(1, 2)

def replacement_args(x):
    return (x,)

@triton.jit
def fused_reshape_transpose_kernel(
    x_ptr,
    out_ptr,
    orig_elements,
    seq_len,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reshape from flat format to (16, seq_len, 64) and then transpose to (16, 64, seq_len)
    # Input is flat, we need to map: flat_idx -> (head_idx, seq_pos, elem_idx) -> (head_idx, elem_idx, seq_pos)
    
    if seq_len == 1:
        # Special case: (16, 1, 64) -> (16, 64, 1) - just swap last two dimensions
        # flat_idx = head_idx * 64 + elem_idx
        # new_flat_idx = head_idx * 64 + elem_idx (same, but we handle in reshape)
        pass
    else:
        # General case: (16, seq_len, 64) -> (16, 64, seq_len)  
        # Inverse mapping: need to reorder from flat storage to transposed layout
        pos_in_seq = offsets % seq_len
        head_idx = (offsets // seq_len) % 16
        elem_idx = (offsets // seq_len) // 16
        
        # Reorder for transpose: (16, seq_len, 64) -> (16, 64, seq_len)
        new_offset = head_idx * (seq_len * 64) + elem_idx * seq_len + pos_in_seq
        offsets = new_offset
    
    # Store reordered data
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_reshape_transpose(x):
    # Get input shape and calculate dimensions
    total_elements = x.numel()
    
    # Determine sequence length based on input size
    # Input could be: [1, 1, 1024] or [1, 577, 1024] shaped after flattening
    # We need to determine seq_len from total_elements
    if total_elements % (16 * 64) == 0:
        seq_len = 1  # This case: (16, 1, 64)
    else:
        seq_len = total_elements // (16 * 64)
    
    # The input is expected to be in a format that when reshaped gives (16, seq_len, 64)
    assert total_elements == 16 * seq_len * 64, f"Input size {total_elements} incompatible with reshape(16,{seq_len},64)"
    
    # Determine output shape: (16, 64, seq_len)
    output_shape = (16, 64, seq_len)
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate optimal block size
    if seq_len <= 32:
        BLOCK_SIZE = 1024  # Use larger block for small sequences
    else:
        BLOCK_SIZE = 1024  # Standard block size
    
    # Launch kernel
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_reshape_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        orig_elements=total_elements,
        seq_len=seq_len,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_reshape_transpose