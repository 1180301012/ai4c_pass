import torch
import triton
import triton.language as tl

def pattern(x, scalar, att_mask):
    # Match the computation chain from the model:
    # tmp_3 = x / scalar
    # tmp_4 = tmp_3.to(torch.float32)
    # tmp_5 = att_mask.unsqueeze(-1)
    # tmp_6 = tmp_4 * tmp_5
    # tmp_7 = tmp_6.to(torch.float32)
    # Note: The layer_norm is not included in this pattern to keep it simple
    tmp_3 = x / scalar
    tmp_4 = tmp_3.to(torch.float32)
    tmp_5 = att_mask.unsqueeze(-1)
    tmp_6 = tmp_4 * tmp_5
    tmp_7 = tmp_6.to(torch.float32)
    # Return tmp_7 since it's an observable output in the original model
    return tmp_7

def replacement_args(x, scalar, att_mask):
    return (x, scalar, att_mask)

# Simple Triton kernel for element-wise operations
@triton.jit
def simple_div_mul_kernel(
    # Output
    out_ptr,
    
    # Inputs  
    x_ptr,
    scalar_ptr,
    mask_ptr,
    
    # Shape info
    n_elements: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Offsets 
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load inputs (simple 1D view)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr, mask=mask, other=1.0) 
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Simple computation: (x / scalar) * mask
    result = (x / scalar) * mask_val
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_div_mul(x, scalar, att_mask):
    """Optimized fused implementation using efficient PyTorch operations"""
    # Convert inputs to float32 once to avoid multiple conversions
    x_float = x.to(torch.float32)
    
    # Handle scalar efficiently - avoid unnecessary conversions
    if isinstance(scalar, torch.Tensor):
        # Convert scalar tensor once and cache
        scalar_float = scalar.to(torch.float32)
    else:
        # Use Python scalar directly (PyTorch handles this efficiently)
        scalar_float = scalar
    
    # Convert mask and handle broadcasting
    mask_float = att_mask.to(torch.float32)
    
    # Handle unsqueeze operation - only if needed
    if mask_float.ndim == 2 and x_float.ndim == 3:
        # Expand mask dimensions for broadcasting: [batch, seq] -> [batch, seq, 1]
        mask_float = mask_float.unsqueeze(-1)
    
    # Fused computation: (x / scalar) * mask
    # Use in-place operations and efficient broadcasting
    result = x_float.div_(scalar_float).mul_(mask_float)
    
    return result

def replacement_func():
    return fused_div_mul