import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    # First addition: in_0 += in_1
    in_0 += in_1
    tmp_0 = in_0
    # Second addition: tmp_0 += in_3  
    tmp_0 += in_3
    tmp_1 = tmp_0
    return tmp_1

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def fused_add_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Perform fused addition: (x + y) + z
    out = x + y + z
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_triton_add(x, y, z):
    # Handle the case where we need to broadcast tensors to same shape
    max_shape = torch.broadcast_shapes(x.shape, y.shape, z.shape)
    x_b = x.expand(max_shape) if x.shape != max_shape else x
    y_b = y.expand(max_shape) if y.shape != max_shape else y  
    z_b = z.expand(max_shape) if z.shape != max_shape else z
    
    N = x_b.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x_b)

    fused_add_kernel[(num_programs,)](
        x_ptr=x_b,
        y_ptr=y_b,
        z_ptr=z_b,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return fused_triton_add