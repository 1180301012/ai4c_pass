import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern matches simple reshape operation
    tmp_1 = x.reshape(-1, 256, -1)
    return tmp_1

def replacement_args(x):
    # We only need the input tensor
    return (x,)

@triton.jit
def optimized_reshape_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Grid setup - only need to parallelize over batch dimension for simple reshape
    m = tl.program_id(0)
    
    # Calculate offset
    m_offset = m * BLOCK_SIZE_M
    
    # Create mask for bounds checking
    mask_m = m_offset + tl.arange(0, BLOCK_SIZE_M) < batch_size
    
    # Load input tensor slice and store directly to output
    # Input: [batch, 256, seq_len, 1] -> Output: [batch, 256, seq_len]
    base_offset = m_offset * 256 * seq_len
    for i in tl.range(0, 256):
        for j in range(seq_len):
            offset = base_offset + j * 256 + i
            x_val = tl.load(x_ptr + offset, mask=mask_m[j], other=0.0)
            tl.store(out_ptr + offset, x_val, mask=mask_m[j])

@torch.fx.wrap
def optimized_reshape(x):
    # Input shape: [batch, 256, seq_len, 1]
    original_shape = x.shape
    batch_size, dim_256, seq_len, _ = original_shape
    
    # Output shape: [batch, 256, seq_len]
    output_shape = (batch_size, 256, seq_len)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # For simple reshape, use a block size that gives good GPU occupancy
    BLOCK_SIZE_M = 128  # Batch dimension block size
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    optimized_reshape_kernel[grid_m,](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out

def replacement_func():
    return optimized_reshape