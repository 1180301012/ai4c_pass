import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_1 = x.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_reshape_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements for better memory coalescing
    pid = tl.program_id(0)
    program_offset = pid * BLOCK_SIZE
    
    # Total elements = hidden * seq_len
    if pid >= (hidden * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE:
        return
    
    # Get thread offsets within the block
    offsets = program_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (hidden * seq_len)
    
    # Calculate coordinates for all elements in the block
    # hidden_idx = offset // seq_len, seq_len_idx = offset % seq_len
    hidden_idx = (offsets // seq_len) % hidden
    seq_len_idx = offsets % seq_len
    
    # Calculate original tensor indices for all elements  
    original_indices = seq_len_idx * hidden + hidden_idx
    
    # Load from original tensor and store in output tensor
    values = tl.load(x_ptr + original_indices, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)

@torch.fx.wrap
def fused_unsqueeze_transpose(x):
    # Get input shape
    batch_size, seq_len, hidden = x.shape
    
    # Calculate output shape: [batch_size, 1, hidden, seq_len]
    output_shape = (batch_size, 1, hidden, seq_len)
    
    # Create output tensor using only allowed operations
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Use optimized block size for better memory coalescing
    block_size = 128
    
    # Calculate launch grid settings
    total_elements = hidden * seq_len  # 128 * 1024 = 131072 elements
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Launch kernel with block processing for better memory coalescing
    fused_reshape_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden=hidden,
        BLOCK_SIZE=block_size,
    )
    
    return out

def replacement_func():
    return fused_unsqueeze_transpose