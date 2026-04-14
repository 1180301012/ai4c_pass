import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_0, in_2):
    # Match: tmp_3 * in_0 + in_2
    scaled = tmp_3 * in_0
    result = scaled + in_2
    return result

def replacement_args(tmp_3, in_0, in_2):
    return (tmp_3, in_0, in_2)

@triton.jit
def fused_mul_add_kernel(
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
    
    # Load inputs and perform fused multiply-add
    tmp_3 = tl.load(tmp_3_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Load scalar value (first and only element) from in_0_ptr
    in_0_val = tl.load(in_0_ptr)
    
    # Fused operation: result = (tmp_3 * in_0) + in_2
    result = (tmp_3 * in_0_val) + in_2
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul_add(tmp_3, in_0, in_2):
    N = tmp_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(tmp_3, dtype=tmp_3.dtype)
    
    # Pass the scalar tensor directly - kernel will access its first element
    fused_mul_add_kernel[(num_programs,)](
        tmp_3_ptr=tmp_3,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_mul_add