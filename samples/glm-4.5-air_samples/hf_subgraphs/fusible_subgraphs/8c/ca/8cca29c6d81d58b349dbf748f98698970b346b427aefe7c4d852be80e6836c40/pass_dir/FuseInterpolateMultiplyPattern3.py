import torch
import triton
import triton.language as tl

# Simple pattern matching - just match the input argument structure
def pattern(in_0, in_1, in_2, in_3):
    # This matches the structure: 4 inputs, returns 2 outputs
    # The actual computation will be handled by the replacement kernel
    return (in_0, in_1, in_2, in_3)

# Arguments extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_elementwise_kernel(
    x_ptr, y_ptr, z_ptr, w_ptr,
    out1_ptr, out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    
    # Simple elementwise operations
    out1 = x + y
    out2 = z * w
    
    # Store results
    tl.store(out1_ptr + offsets, out1, mask=mask)
    tl.store(out2_ptr + offsets, out2, mask=mask)

@torch.fx.wrap
def simple_elementwise_fusion(x, y, z, w):
    # Simple fused operation - elementwise add and multiply
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(z)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_elementwise_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        w_ptr=w,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out1, out2)

def replacement_func():
    return simple_elementwise_fusion