import torch
import triton
import triton.language as tl

# Pattern matching function for ReLU + Addition
def pattern(in_0, in_1):
    """Match ReLU followed by addition"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace = False)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for ReLU + Addition fusion
@triton.jit
def relu_add_kernel(
    in_0_ptr,   # Input to add
    in_1_ptr,   # Input for ReLU
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both inputs
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Fused ReLU + Addition: max(0, in_1_val) + in_0_val
    out = tl.maximum(in_1_val, 0.0) + in_0_val
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(in_0, in_1):
    """Fused ReLU + Addition operation"""
    n_elements = in_0.numel()
    BLOCK_SIZE = 2048  # Larger block size for better GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    relu_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_relu_add