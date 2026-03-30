import torch
import triton
import triton.language as tl

# Pattern matching function for ReLU * in_1 + in_0 fusion
def pattern(in_2, in_1, in_0):
    """Match the sequence: ReLU(in_2) * in_1 + in_0"""
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Optimized fused kernel for ReLU * scale + bias
@triton.jit
def fused_relu_mul_add_kernel(
    x_ptr,      # in_2 - input to ReLU
    scale_ptr,  # in_1 - scale tensor  
    bias_ptr,   # in_0 - bias tensor
    out_ptr,    # output tensor
    n_elements,  # total elements
    scale_val,  # scalar scale value if scale is scalar
    bias_val,   # scalar bias value if bias is scalar
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle scale - use scalar if provided, otherwise load from tensor
    if scale_val is not None:
        scale = x.dtype(scale_val)
    else:
        scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0).to(x.dtype)
    
    # Handle bias - use scalar if provided, otherwise load from tensor  
    if bias_val is not None:
        bias = x.dtype(bias_val)
    else:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(x.dtype)
    
    # Fused operation: ReLU(x) * scale + bias
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_mul_add(x, scale, bias):
    """ fused_relu_mul_add wrapper function using only Triton operations """
    # Extract scalar values if they are scalars, otherwise keep as tensors
    if scale.numel() == 1:
        scale_val = scale.item()
    else:
        scale_val = None
        
    if bias.numel() == 1:
        bias_val = bias.item()
    else:
        bias_val = None
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Pass scalar values as special arguments for the kernel
    fused_relu_mul_add_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=N,
        scale_val=scale_val,
        bias_val=bias_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_relu_mul_add