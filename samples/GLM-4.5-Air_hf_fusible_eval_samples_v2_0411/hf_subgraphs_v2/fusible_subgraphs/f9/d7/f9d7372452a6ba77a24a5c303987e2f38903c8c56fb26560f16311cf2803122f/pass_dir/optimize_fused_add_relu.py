import torch
import triton
import triton.language as tl

def pattern(in_0, in_2, in_3):
    # Pattern matching the fused add + add + relu operations
    in_3 += in_0
    in_4 = in_3
    in_4 += in_2
    tmp_0 = in_4
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2

def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)

@triton.jit
def fused_add_relu_kernel(
    x_ptr,
    y_ptr, 
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse the operations: temp = z + x, out = temp + y, apply relu
    temp = z + x
    out = temp + y
    out = tl.maximum(out, 0.0)  # ReLU operation
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_relu_triton(in_0, in_2, in_3):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    
    # Determine grid size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    fused_add_relu_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_2,
        z_ptr=in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_relu_triton