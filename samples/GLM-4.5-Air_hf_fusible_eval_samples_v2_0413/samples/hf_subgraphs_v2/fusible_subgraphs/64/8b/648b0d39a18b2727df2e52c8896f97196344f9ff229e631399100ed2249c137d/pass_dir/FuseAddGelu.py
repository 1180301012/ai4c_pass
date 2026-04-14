import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_gelu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for addition + GELU operation"""
    # Compute program indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse addition and GELU: gelu(x + y)
    # GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2.0/pi) * (x + 0.044715 * x^3)))
    # Using the polynomial approximation for performance
    z = x + y
    gelu_z = z * 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0/tl.pi) * z * (1.0 + 0.044715 * z * z)))
    
    # Store result
    tl.store(out_ptr + offsets, gelu_z, mask=mask)

@torch.fx.wrap
def fused_add_gelu(x, y):
    """Wrapper for fused addition + GELU operation"""
    # Handle different tensor shapes by flattening
    original_shape = x.shape
    
    # Flatten tensors if necessary
    if len(original_shape) > 1:
        x_flat = x.flatten()
        y_flat = y.flatten()
    else:
        x_flat = x
        y_flat = y
    
    N = x_flat.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_flat = torch.empty_like(x_flat)
    
    fused_add_gelu_kernel[(num_programs, 1, 1)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=out_flat,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Restore original shape
    return out_flat.reshape(original_shape)

def pattern(x, y):
    """Pattern to match: element-wise addition followed by GELU activation"""
    # Use regular addition instead of in-place to avoid pattern matching issues
    tmp_add = torch.add(x, y)
    tmp_gelu = torch.nn.functional.gelu(tmp_add, approximate='none')
    return tmp_gelu

def replacement_args(x, y):
    """Extract arguments for fused operation"""
    return (x, y)

def replacement_func():
    """Return the fused kernel function"""
    return fused_add_gelu