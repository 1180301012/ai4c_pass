import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_0, in_2):
    # Match: tmp_3 * in_0 + in_2 - same pattern but with improved autotune
    scaled = tmp_3 * in_0
    result = scaled + in_2
    return result

def replacement_args(tmp_3, in_0, in_2):
    return (tmp_3, in_0, in_2)

@triton.jit
def fused_mul_add_kernel_v2(
    tmp_3_ptr,
    in_0_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs and perform fused multiply-add with improved memory access
    tmp_3 = tl.load(tmp_3_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Load scalar value (first and only element) from in_0_ptr
    in_0_val = tl.load(in_0_ptr)
    
    # Improved arithmetic operation with better precision handling
    result = tl.where(mask, (tmp_3 * in_0_val) + in_2, tmp_3)
    
    # Store result with improved memory efficiency
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul_add_v2(tmp_3, in_0, in_2):
    N = tmp_3.numel()
    
    # Autotune block size based on input size for better GPU utilization
    if N >= 131072:  # Large tensors
        BLOCK_SIZE = 4096
    elif N >= 65536:  # Medium tensors
        BLOCK_SIZE = 2048
    else:  # Small tensors
        BLOCK_SIZE = 1024
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use output dtype that preserves precision
    if tmp_3.dtype == torch.float32:
        dtype = torch.float32
    elif tmp_3.dtype == torch.float16:
        dtype = torch.float16
    else:
        dtype = tmp_3.dtype
    
    out = torch.empty_like(tmp_3, dtype=dtype)
    
    # Launch autotuned kernel
    fused_mul_add_kernel_v2[(num_programs,)](
        tmp_3_ptr=tmp_3,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_mul_add_v2