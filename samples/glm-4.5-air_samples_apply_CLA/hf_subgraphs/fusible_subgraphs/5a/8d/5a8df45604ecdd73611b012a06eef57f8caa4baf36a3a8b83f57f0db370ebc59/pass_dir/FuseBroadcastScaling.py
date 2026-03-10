import torch
import triton
import triton.language as tl

def pattern(scaling_factor, conv_out):
    """
    Pattern to match the broadcast sequence:
    scaling_factor.unsqueeze(-1).unsqueeze(-1) * conv_out
    """
    # First unsqueeze
    tmp_7 = scaling_factor.unsqueeze(-1)
    # Second unsqueeze  
    tmp_8 = tmp_7.unsqueeze(-1)
    # Broadcasting multiplication
    out = tmp_8 * conv_out
    return out

def replacement_args(scaling_factor, conv_out):
    """Return arguments needed for replacement"""
    return (scaling_factor, conv_out)

@triton.jit
def fused_broadcast_kernel(
    scaling_ptr,
    conv_out_ptr,
    out_ptr,
    batch_size,
    feature_dim,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies broadcasting multiplication efficiently
    scaling_factor: [feature_dim] -> broadcast to [batch_size, feature_dim, height, width]
    """
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    total_elements = batch_size * feature_dim * height * width
    if pid >= total_elements:
        return
        
    # Calculate coordinates
    offset = pid
    w = offset % width
    offset = offset // width
    h = offset % height
    offset = offset // height
    c = offset % feature_dim
    b = offset // feature_dim
    
    # Load scaling factor (broadcasted)
    scaling_val = tl.load(scaling_ptr + c)
    
    # Load conv_out value
    conv_out_val = tl.load(conv_out_ptr + pid)
    
    # Apply scaling
    out_val = scaling_val * conv_out_val
    
    # Store result
    tl.store(out_ptr + pid, out_val)

@torch.fx.wrap
def fused_broadcast_scaling(scaling_factor, conv_out):
    """
    Fused operation that combines unsqueeze + unsqueeze + multiplication
    into a single efficient kernel
    """
    batch_size, feature_dim, height, width = conv_out.shape
    
    # Prepare output tensor
    out = torch.empty_like(conv_out)
    
    # Calculate grid size
    total_elements = batch_size * feature_dim * height * width
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_broadcast_kernel[grid_size](
        scaling_ptr=scaling_factor,
        conv_out_ptr=conv_out,
        out_ptr=out,
        batch_size=batch_size,
        feature_dim=feature_dim,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused broadcast scaling function"""
    return fused_broadcast_scaling