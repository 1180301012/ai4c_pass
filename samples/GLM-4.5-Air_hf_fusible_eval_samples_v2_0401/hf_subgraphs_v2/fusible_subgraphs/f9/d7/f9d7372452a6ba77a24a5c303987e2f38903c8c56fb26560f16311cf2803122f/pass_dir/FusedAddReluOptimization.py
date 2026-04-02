import torch
import triton
import triton.language as tl

def pattern(in_3, in_0, in_2):
    # Fused addition operation: in_3 + in_0 + in_2 followed by ReLU
    tmp_0 = in_3 + in_0 + in_2
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2

def replacement_args(in_3, in_0, in_2):
    return (in_3, in_0, in_2)

@triton.jit
def fused_add_relu_kernel(
    in_3_ptr, in_0_ptr, in_2_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors using memory coalescing
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Fused addition operation with reduced memory access
    tmp_0 = in_3 + in_0 + in_2
    
    # ReLU activation using tl.maximum for better performance
    out = tl.maximum(tmp_0, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_relu(in_3, in_0, in_2):
    # Use the largest tensor as reference for shape
    n_elements = in_3.numel()
    out = torch.empty_like(in_3)
    
    # Use optimal block size for better GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_relu_kernel[(num_programs,)](
        in_3_ptr=in_3,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_relu