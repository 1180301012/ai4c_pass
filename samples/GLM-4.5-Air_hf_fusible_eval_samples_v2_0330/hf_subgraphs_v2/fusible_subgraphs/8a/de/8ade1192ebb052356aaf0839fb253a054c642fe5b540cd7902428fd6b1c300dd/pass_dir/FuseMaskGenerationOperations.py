import torch
import triton
import triton.language as tl

def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = torch.sub(tmp_5, tmp_4)
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = torch.masked_fill(tmp_6, tmp_7, -3.4028234663852886e+38)
    return tmp_8

@triton.jit
def fused_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)  # Single dimension for simplicity
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor values directly
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32 if needed (in_5 might be int64)
    float_vals = in_vals.to(tl.float32)
    
    # Create 1.0 tensor and subtract
    ones = tl.full_like(float_vals, 1.0, tl.float32)
    diff = ones - float_vals
    
    # Create boolean mask where diff != 0  
    bool_mask = (diff != 0).to(tl.int1)
    
    # Apply masked_fill with -inf for boolean True values
    result = tl.where(bool_mask, -3.4028234663852886e+38, diff)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mask_generation(in_5, batch_size, seq_len):
    # Calculate total elements in mask tensor (preserves original shape [1, 1, seq_len, seq_len])
    total_elements = in_5.numel()
    
    # Determine optimal block size and grid
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape as input
    tmp_8 = torch.full_like(in_5, -3.4028234663852886e+38, dtype=torch.float32)
    
    # Launch kernel
    fused_mask_kernel[(num_programs,)](
        in_ptr=in_5,
        out_ptr=tmp_8,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_8

def replacement_args(in_5):
    # Get batch size and sequence length from input tensor shape
    # in_5 has shape [1, 1, seq_len, seq_len]
    seq_len = in_5.shape[-1]
    batch_size = in_5.shape[0] * in_5.shape[1]  # Should be 1 * 1 = 1
    return (in_5, batch_size, seq_len)

def replacement_func():
    return fused_mask_generation