import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Match addition followed by contiguous operation
    tmp = a + b
    result = tmp.contiguous()
    return result

def replacement_args(x, y):
    # Extract arguments for the pattern function
    # x and y will be the matched nodes from the computation graph
    return (x, y)

@triton.jit
@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=2),
        triton.Config(num_warps=8, num_stages=2),
        triton.Config(num_warps=16, num_stages=2),
        triton.Config(num_warps=8, num_stages=3),
        triton.Config(num_warps=16, num_stages=3),
    ],
    key=['n_elements'],
)
def fused_add_contiguous_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: x + y
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_contiguous(x, y):
    # Determine output shape (same as input tensors)
    output_shape = x.shape  # Both tensors should have the same shape
    n_elements = x.numel()
    
    # Create output tensor with contiguous memory
    out = torch.empty(output_shape, dtype=torch.float32, device='cuda:0')
    
    # Adaptive block size based on tensor size
    if n_elements < 8192:
        BLOCK_SIZE = 256
    elif n_elements < 65536:
        BLOCK_SIZE = 512
    elif n_elements < 262144:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Calculate number of programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure we have at least some work for each GPU block
    if n_elements > 0:
        fused_add_contiguous_kernel[(num_programs,)](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return fused_add_contiguous