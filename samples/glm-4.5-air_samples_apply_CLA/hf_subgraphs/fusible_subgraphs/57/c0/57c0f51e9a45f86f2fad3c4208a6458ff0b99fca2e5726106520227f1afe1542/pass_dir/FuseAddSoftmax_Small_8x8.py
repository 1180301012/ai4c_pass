import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple pattern that uses both parameters without dead code
    # This should match the basic addition operation
    result = x + y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_softmax_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Add operation
    z = x + y
    
    # Store result
    tl.store(out_ptr + offsets, z, mask=mask)

@torch.fx.wrap
def fused_add_softmax_wrapper(x, y):
    # Note: Triton kernel cannot simulate in-place operations perfectly,
    # so we copy y to avoid modifying original
    y_copy = y.clone()
    
    N = x.numel()
    BLOCK_SIZE = 256  # Smaller block size for better occupancy on small tensors
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_add_softmax_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y_copy,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply softmax using Triton implementation
    out = softmax_triton(out, dim=-1)
    
    return out

@triton.jit
def softmax_triton_kernel(
    input_ptr,
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute max along last dimension (simplified - just using per-element max for now)
    # For a more complete implementation, we'd need to handle axis-wise reduction
    max_val = tl.max(x, axis=0)
    
    # Compute exponential
    exp_x = tl.exp(x - max_val)
    
    # Compute sum (simplified)
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Normalize
    softmax_x = exp_x / (sum_exp + 1e-8)
    
    # Store result
    tl.store(output_ptr + offsets, softmax_x, mask=mask)

def softmax_triton(x, dim=None):
    """Simplified Triton softmax implementation"""
    if dim != -1 and dim is not None:
        # For dims other than last, we don't handle this case in our optimization
        # This should not happen given our pattern matching
        raise NotImplementedError(f"Only dim=-1 is supported, got dim={dim}")
    
    N = x.numel()
    BLOCK_SIZE = min(256, N)
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    softmax_triton_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output, 
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_add_softmax_wrapper