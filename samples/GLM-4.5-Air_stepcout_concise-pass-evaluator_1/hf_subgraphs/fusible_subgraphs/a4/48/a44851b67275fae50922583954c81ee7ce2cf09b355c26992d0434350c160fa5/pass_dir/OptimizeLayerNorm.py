import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # If either tensor is on CPU, fall back to regular addition
    if x.device.type == 'cpu' or y.device.type == 'cpu':
        return x + y
    
    n_elements = x.numel()
    
    # Use Triton only for larger tensors where GPU parallelism provides benefits
    # For small tensors, PyTorch's optimized CPU implementation might be faster
    if n_elements < 8192:  # Threshold for when GPU acceleration is beneficial
        return x + y
    
    # Choose optimal block size based on tensor size
    if n_elements < 65536:
        BLOCK_SIZE = 1024
    elif n_elements < 262144:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure tensors are contiguous for better GPU performance
    x_contiguous = x.contiguous()
    y_contiguous = y.contiguous()
    out = torch.empty_like(x)
    
    # Launch kernel with optimized parameters
    add_kernel[(num_programs,)](
        x_contiguous, y_contiguous, out, n_elements, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    def kernel_wrapper(x, y):
        return triton_add(x, y)
    
    return kernel_wrapper