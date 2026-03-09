import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, weight, bias):
    # Match the specific addition that's followed by layer_norm
    tmp_2 = in_2 + in_3
    return torch.nn.functional.layer_norm(tmp_2, weight.shape, weight, bias, 1e-06)

def replacement_args(in_2, in_3, weight, bias):
    return (in_2, in_3, weight, bias)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    result = x + y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap 
def fused_add_layernorm(in_2, in_3, weight, bias):
    # Check if main tensors are on CUDA for optimization
    if in_2.device.type != 'cuda' or in_3.device.type != 'cuda':
        # Fall back to original computation
        tmp_2 = in_2 + in_3
        return torch.nn.functional.layer_norm(tmp_2, weight.shape, weight, bias, 1e-06)
    
    # Get tensor dimensions
    n_elements = in_2.numel()
    
    # Perform addition using Triton for CUDA tensors
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for the addition
    added = torch.empty_like(in_2, device=in_2.device)
    
    # Launch optimized addition kernel
    add_kernel[(num_programs,)](
        x_ptr=in_2,
        y_ptr=in_3, 
        out_ptr=added,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply layer norm using PyTorch (already optimized)
    return torch.nn.functional.layer_norm(added, weight.shape, weight, bias, 1e-06)

def replacement_func():
    return fused_add_layernorm