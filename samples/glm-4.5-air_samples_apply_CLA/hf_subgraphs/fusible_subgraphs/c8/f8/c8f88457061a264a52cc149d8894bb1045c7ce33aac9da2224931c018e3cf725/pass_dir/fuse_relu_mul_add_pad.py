import torch
import triton
import triton.language as tl

def pattern(x, scale, bias):
    """Pattern matches: scale * relu(x) + bias"""
    tmp_2 = torch.nn.functional.relu(x, inplace=False)
    tmp_3 = scale * tmp_2
    tmp_4 = tmp_3 + bias
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    # The pattern expects (x, scale, bias)
    # in_0=bias, in_1=scale, in_2=x
    return (in_2, in_1, in_0)

@triton.jit
def fused_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for scale * relu(x) + bias"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load values
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # Apply operations: scale * relu(x) + bias
    relu_x = tl.maximum(x, 0.0)
    result = relu_x * scale + bias
    
    # Store result
    tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def fused_kernel_wrapper(x, scale, bias):
    """Wrapper for the fused kernel"""
    # Handle x as tensor, scale and bias as scalars
    if isinstance(x, torch.Tensor):
        n_elements = x.numel()
        out = torch.empty_like(x)
    else:
        raise ValueError("Input x must be a tensor")
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_kernel_wrapper