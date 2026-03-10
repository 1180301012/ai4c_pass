import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching: ReLU followed by Flatten operation"""
    # Match ReLU -> Flatten sequence
    relu_out = torch.nn.functional.relu(x, inplace=False)
    final_out = relu_out.flatten(1, -1)
    return final_out

def replacement_args(x):
    """Extract arguments needed for replacement"""
    return (x,)

@triton.jit
def relu_flatten_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: ReLU + Flatten operations"""
    # Each program handles one element in the flattened output
    pid = tl.program_id(0)
    
    # Calculate element index in flattened space
    elem_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = elem_id < (batch_size * feature_size)
    
    # Calculate original 4D coordinates for 4D input [N, C, H, W]
    total_elements = batch_size * feature_size
    
    # For 4D input with flatten(1, -1): [N, C, H, W] -> [N, C*H*W]
    # We need to map 1D index back to original 4D coordinates
    n = elem_id // feature_size
    remaining = elem_id % feature_size
    
    # Load input element directly from flattened layout
    x = tl.load(x_ptr + elem_id, mask=mask, other=0.0)
    
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    
    # Store result directly in flattened layout
    tl.store(out_ptr + elem_id, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    """Optimized fused ReLU + Flatten operation"""
    # Handle dynamic cases for input shape
    if len(x.shape) != 4:
        # For non-4D inputs, fall back to original implementation
        return torch.nn.functional.relu(x).flatten(1, -1)
    
    # Get input shape for 4D tensor [N, C, H, W]
    batch_size, channels, height, width = x.shape
    feature_size = channels * height * width
    
    # Create output tensor in flattened shape [N, C*H*W]
    output_shape = (batch_size, feature_size)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Convert to 1D tensor for kernel processing
    x_flat = x.reshape(-1)  # [N*C*H*W]
    out_flat = out.reshape(-1)  # [N*C*H*W]
    
    # Calculate grid dimensions
    total_elements = batch_size * feature_size
    grid = lambda meta: ( (total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    
    # Launch fused kernel
    relu_flatten_fused_kernel[grid](
        x_ptr=x_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        feature_size=feature_size,
        BLOCK_SIZE=1024,
    )
    
    return out

def replacement_func():
    """Return the fused ReLU + Flatten function"""
    return fused_relu_flatten