import torch
import triton
import triton.language as tl

def pattern(residual, scale, y):
    # This matches the pattern: tmp_3 = residual * scale; tmp_4 = tmp_3 + y
    # Where scale is a scalar tensor and residual, y are the input tensors
    tmp_3 = residual * scale
    tmp_4 = tmp_3 + y
    return tmp_4

def replacement_args(residual, scale, y):
    # Extract arguments: residual tensor, scalar scale, and y tensor
    return (residual, scale, y)

@triton.jit
def fused_scale_add_kernel(
    residual_ptr,
    scale_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple but efficient kernel with compile-time constant block size
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with coalesced memory access
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    y_data = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: (residual * scale) + y
    result = residual * scale_val + y_data
    
    # Store result with proper masking
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_scale_add(residual, scale, y):
    # Get total number of elements
    n_elements = residual.numel()
    
    # Choose optimal block size based on tensor size
    # Fine-tuned block sizes for better performance across different tensor configurations
    if n_elements <= 2048:  # Very small tensors
        BLOCK_SIZE = 64
    elif n_elements <= 8192:  # Small tensors  
        BLOCK_SIZE = 128
    elif n_elements <= 32768:  # Medium tensors
        BLOCK_SIZE = 256
    elif n_elements <= 131072:  # Large tensors
        BLOCK_SIZE = 512
    else:  # Very large tensors
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor with proper dtype
    out = torch.empty_like(residual)
    
    # Launch the optimized kernel
    fused_scale_add_kernel[grid](
        residual_ptr=residual,
        scale_val=float(scale.item()),  # Convert to float32 for computation
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_scale_add