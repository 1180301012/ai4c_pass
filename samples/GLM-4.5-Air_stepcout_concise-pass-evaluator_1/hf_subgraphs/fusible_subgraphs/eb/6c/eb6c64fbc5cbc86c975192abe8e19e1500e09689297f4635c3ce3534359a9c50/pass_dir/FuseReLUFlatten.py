import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match ReLU followed by flatten operations"""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(in_0):
    """Extract input tensor for fused operation"""
    return (in_0,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Flatten kernel"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    """Fused ReLU + Flatten operation wrapper"""
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Optimized block size for GPU occupancy
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor with same dtype as input
    out = torch.empty_like(x)
    
    # Launch fused kernel
    fused_relu_flatten_kernel[grid_size](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match expected flatten shape
    original_shape = x.shape
    if len(original_shape) >= 2:
        # Flatten all dimensions after the first
        flattened_shape = [original_shape[0], -1]
        out = out.reshape(flattened_shape)
    
    return out

def replacement_func():
    """Return fused function reference"""
    return fused_relu_flatten