import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Simple arithmetic pattern that should match
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_arithmetic_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements,
    target_dtype: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: in_1 - in_0 * 1000000.0
    converted_in_0 = tl.cast(in_0_val, target_dtype) * scale
    result = in_1_val - converted_in_0
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_arithmetic_compute(in_0, in_1):
    # Handle device and type conversions
    if in_0.device != in_1.device:
        in_0 = in_0.to(in_1.device)
    
    if in_0.dtype != in_1.dtype:
        in_0 = in_0.to(in_1.dtype)
    
    # Determine target dtype and scale
    target_dtype = tl.float16 if in_1.dtype == torch.float16 else tl.bfloat16 if in_1.dtype == torch.bfloat16 else tl.float32
    scale = 1000000.0
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    n_elements = in_1.numel()
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_arithmetic_kernel[(num_programs,)](
        in_0, in_1, out,
        n_elements,
        target_dtype, scale,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_arithmetic_compute