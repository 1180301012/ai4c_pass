import torch
import triton
import triton.language as tl

def pattern(tmp_11, in_0):
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    return tmp_13

@triton.jit
def optimized_addition_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors directly from their current devices
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    out = a + b
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def direct_tensor_addition(tmp_11, in_0):
    # Ensure tensors are on the same device (eliminates redundant transfer)
    if tmp_11.device != in_0.device:
        tmp_11 = tmp_11.to(in_0.device)
    
    # Get total number of elements
    n_elements = tmp_11.numel()
    
    # Use optimal block size for GPU
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(tmp_11)
    
    # Launch optimized kernel (removes redundant device transfer)
    optimized_addition_kernel[(num_programs,)](
        a_ptr=tmp_11,
        b_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(tmp_11, in_0):
    return (tmp_11, in_0)

def replacement_func():
    return direct_tensor_addition