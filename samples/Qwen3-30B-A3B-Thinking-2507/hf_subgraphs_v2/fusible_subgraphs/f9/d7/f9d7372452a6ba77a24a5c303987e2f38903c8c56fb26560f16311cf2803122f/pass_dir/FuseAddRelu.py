import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_2, in_3):
    # Exactly match the dataflow: in_3 = in_3 + in_0, then in_4 = in_3 + in_2, then relu
    tmp_0 = in_3 + in_0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.relu(tmp_1)
    return tmp_2

# Argument extraction function

def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)

# Optimized kernel
@triton.jit

def fuse_add_relu_kernel(
    in_0_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    in0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: in3 + in0 + in2, then ReLU
    out = in3 + in0 + in2
    out = tl.maximum(out, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap

def fuse_add_relu(in_0, in_2, in_3):
    N = in_0.numel()
    BLOCK_SIZE = 256
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_3)
    
    fuse_add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fuse_add_relu