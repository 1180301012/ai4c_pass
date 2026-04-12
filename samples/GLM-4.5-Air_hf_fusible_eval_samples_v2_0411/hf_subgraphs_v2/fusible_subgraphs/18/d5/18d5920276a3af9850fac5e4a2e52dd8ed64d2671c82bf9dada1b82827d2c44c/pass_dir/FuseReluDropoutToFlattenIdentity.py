import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the pattern: ReLU -> Flatten (dropout might be optimized away)
    tmp_0 = torch.nn.functional.relu(x, inplace = False)
    tmp_2 = tmp_0.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    # Get flattened dimension (all dimensions after the first)
    original_shape = x.shape
    flattened_dim = 1
    for dim in original_shape[1:]:
        flattened_dim *= dim
    
    N = x.numel()
    BLOCK_SIZE = 1024 if flattened_dim <= 4096 else 2048
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=x.dtype)
    
    relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match expected output: [B, C, 1, 1] -> [B, C]
    expected_shape = (original_shape[0], flattened_dim)
    return out.view(expected_shape)

def replacement_func():
    return fused_relu_flatten

@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    # Get flattened dimension (all dimensions after the first)
    original_shape = x.shape
    flattened_dim = 1
    for dim in original_shape[1:]:
        flattened_dim *= dim
    
    N = x.numel()
    BLOCK_SIZE = 1024 if flattened_dim <= 4096 else 2048
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=x.dtype)
    
    relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match expected output: [B, C, 1, 1] -> [B, C]
    expected_shape = (original_shape[0], flattened_dim)
    return out.view(expected_shape)

def replacement_func():
    return fused_relu_flatten