import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    return tmp_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_add_kernel(
    in_0_ptr,
    in_1_ptr, 
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three input tensors
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Fused addition: x + y + z
    out = x + y + z
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_triton_add(in_0, in_1, in_2):
    # Get tensor shape and total elements
    shape = in_0.shape
    n_elements = in_0.numel()
    
    # Determine optimal block size based on tensor size
    if n_elements < 8192:
        BLOCK_SIZE = 128
    elif n_elements < 65536:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch the fused kernel
    fused_add_kernel[(num_programs,)](
        in_0,
        in_1,
        in_2,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_triton_add