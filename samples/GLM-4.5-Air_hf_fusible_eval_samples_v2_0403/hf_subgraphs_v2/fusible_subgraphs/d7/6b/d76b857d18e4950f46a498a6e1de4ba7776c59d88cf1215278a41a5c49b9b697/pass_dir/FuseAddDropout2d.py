import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern for addition followed by dropout2d with training=False"""
    # Simple addition pattern - matches the core structure of addition + dropout
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_dropout2d_kernel(
    x_ptr, y_ptr, output_ptr,
    N, C, H, W,
    dropout_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused addition and dropout2d kernel using Triton"""
    # Program id for each output element
    pid = tl.program_id(0)
    
    # Calculate starting index for this program
    start_idx = pid * BLOCK_SIZE
    
    # Create fixed-size offsets for this program
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total number of elements and create mask
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    # Load input tensors with masking
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: addition followed by dropout scaling
    # dropout2d with p=0.1, training=False = multiplication by (1 - 0.1) = 0.9
    # Since we're doing element-wise addition, dropout applies same scaling to all elements
    tmp_vals = x_vals + y_vals
    out_vals = tmp_vals * dropout_scale
    
    # Store results
    tl.store(output_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def optimized_fused_add_dropout2d(x, y):
    """Wrapper function for fused addition and dropout2d operation"""
    # Check input shapes and ensure they're compatible
    assert x.shape == y.shape, f"Input shapes must match: {x.shape} vs {y.shape}"
    
    N, C, H, W = x.shape
    total_elements = N * C * H * W
    
    # Create output tensor
    output = torch.empty((N, C, H, W), dtype=x.dtype, device=x.device)
    
    # Block size configuration
    BLOCK_SIZE = 1024  # Number of elements per thread
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Dropout scaling factor (1 - p) where p=0.1
    # Use exact float32 precision for scaling
    dropout_scale = 0.9
    
    # Launch kernel - grid must be a tuple even for 1D
    fused_add_dropout2d_kernel[(grid_size,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        dropout_scale=dropout_scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the fused addition and dropout2d function"""
    return optimized_fused_add_dropout2d