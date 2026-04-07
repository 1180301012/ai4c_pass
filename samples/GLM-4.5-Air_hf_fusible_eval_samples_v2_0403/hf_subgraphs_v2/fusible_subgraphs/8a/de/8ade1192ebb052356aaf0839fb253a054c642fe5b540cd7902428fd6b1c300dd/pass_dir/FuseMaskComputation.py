import torch
import triton
import triton.language as tl

@triton.jit
def fused_mask_kernel(
    in_ptr, 
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load input as int64 and convert to float32
    in_vals = tl.load(in_ptr + offsets, mask=mask, dtype=tl.int64)
    in_float = in_vals.to(tl.float32)
    
    # Compute mask: 1.0 - input, then identify positions where result is <= 0
    sub_result = 1.0 - in_float
    bool_mask = sub_result <= 0.0
    
    # Apply masked fill with -FLT_MAX
    out_vals = tl.where(bool_mask, -3.4028234663852886e+38, sub_result)
    
    # Store result
    tl.store(out_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def fused_mask_computation(in_tensor):
    # Get tensor properties
    n_elements = in_tensor.numel()
    block_size = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Create output tensor with same device/dtype as input but float32
    out_tensor = torch.empty_like(in_tensor, dtype=torch.float32)
    
    # Launch kernel
    fused_mask_kernel[(num_programs,)](
        in_ptr=in_tensor,
        out_ptr=out_tensor,
        n_elements=n_elements,
        block_size=block_size,
    )
    
    return out_tensor

# Pattern matching for mask computation
def pattern(in_5):
    # Match the exact pattern: convert -> subtract -> bool -> masked_fill
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8

def replacement_args(in_5):
    return (in_5,)

def replacement_func():
    return fused_mask_computation