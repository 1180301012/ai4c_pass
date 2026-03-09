import torch
import triton
import triton.language as tl

def pattern(x):
    return x.view(1, -1, 16, 64).transpose(1, 2)

def replacement_args(x):
    return (x,)

@triton.jit
def fused_view_transpose_kernel(
    x_ptr,
    out_ptr,
    orig_shape0,
    orig_shape1, 
    orig_shape2,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate output shape: (1, seq_len, 16, 64) -> (1, 16, seq_len, 64)
    # This is equivalent to view(1,-1,16,64).transpose(1,2)
    orig_elements = orig_shape0 * orig_shape1 * orig_shape2
    
    # Special handling for different input shapes
    if orig_shape1 == 1 and orig_shape2 == 1024:  # Input: [1, 1, 1024]
        # Special case for sequence length 1: [1, 1, 1024] -> [1, 16, 1, 64]
        # Direct mapping: no reordering needed, just reshape
        flat_idx = offsets
    elif orig_shape1 > 1 and orig_shape2 == 1024:  # Input: [1, seq_len, 1024]
        # General case: [1, seq_len, 1024] -> [1, 16, seq_len, 64]
        seq_len = orig_shape1
        pos_in_seq = tl.arange(0, BLOCK_SIZE) // 1024
        head_idx = (tl.arange(0, BLOCK_SIZE) % 1024) // 64
        elem_idx = (tl.arange(0, BLOCK_SIZE) % 1024) % 64
        
        # Reorder to (1, 16, seq_len, 64) layout
        seq_len_total = (total_elements + 1023) // 1024
        valid_pos = pos_in_seq < seq_len_total
        flat_idx = pos_in_seq * 1024 + head_idx * 64 + elem_idx
        flat_idx = tl.where(valid_pos, flat_idx, 0)  # Handle out of bounds
    else:
        # Fallback: simple copy (shouldn't happen with our patterns)
        flat_idx = offsets
    
    # Store reordered data
    tl.store(out_ptr + flat_idx, x, mask=mask)

@torch.fx.wrap
def fused_view_transpose(x):
    # Get original shape and calculate dimensions
    orig_shape = x.shape
    total_elements = x.numel()
    
    # Original input could be: [1, 1, 1024] or [1, seq_len, 1024]
    if len(orig_shape) == 3 and orig_shape[0] == 1 and orig_shape[2] == 1024:
        # This is the expected input format from our pattern
        orig_shape0, orig_shape1, orig_shape2 = orig_shape
    else:
        # Try to infer from total elements  
        if total_elements == 1024:
            orig_shape0, orig_shape1, orig_shape2 = 1, 1, 1024
        else:
            # Infer sequence length
            seq_len = total_elements // 1024
            orig_shape0, orig_shape1, orig_shape2 = 1, seq_len, 1024
    
    # Determine output shape from pattern: view(1,-1,16,64).transpose(1,2)
    seq_len_final = orig_shape1
    out_shape = (1, 16, seq_len_final, 64)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Adaptive block size selection based on input size
    if total_elements <= 2048:
        BLOCK_SIZE = 512  # Smaller block for small inputs
    elif total_elements <= 16384:
        BLOCK_SIZE = 1024  # Medium block for medium inputs  
    else:
        BLOCK_SIZE = 2048  # Larger block for large inputs
    
    # Launch kernel
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_view_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        orig_shape0=orig_shape0,
        orig_shape1=orig_shape1,
        orig_shape2=orig_shape2,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_view_transpose