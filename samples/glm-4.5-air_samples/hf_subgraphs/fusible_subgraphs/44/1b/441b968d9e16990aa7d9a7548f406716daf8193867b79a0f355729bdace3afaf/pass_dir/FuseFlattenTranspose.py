import torch
import triton
import triton.language as tl

@triton.jit
def fused_flatten_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size, orig_channels, height, width,
    flatten_dim, BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Calculate flattened dimensions
    flattened_h = height * width
    flattened_size = orig_channels * flattened_h
    
    # Thread offset within the batch element 
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < flattened_size
    
    # Calculate original coordinates from flattened index
    total_offset = batch_idx * flattened_size + offsets
    
    # Map flattened index back to original [H, W, C] coordinates
    # We flatten dimension 2 (H*W*C), so:
    # flattened_idx = w + W*h + W*H*c
    # We want to transpose to [H*W, C, 1] -> [flattened_h, orig_channels, 1]
    # Final shape becomes [batch, flattened_h, orig_channels]
    
    # Calculate h, w, c from flattened index (flattening dim=2)
    c = offsets // flattened_h
    remaining = offsets % flattened_h
    h = remaining // width
    w = remaining % width
    
    # Original memory layout: [batch, C, H, W] -> offset = batch*C*H*W + c*H*W + h*W + w
    orig_offset = batch_idx * orig_channels * height * width + c * height * width + h * width + w
    
    # Load data
    x = tl.load(x_ptr + orig_offset, mask=mask, other=0.0)
    
    # Transposed layout: [batch, flattened_h, orig_channels] -> offset = batch*flattened_h*orig_channels + flattened_h*c + h
    # Since we're flattening to [H*W, C], it's offset = batch*flattened_size + c*flattened_h + h
    transposed_offset = batch_idx * flattened_size + c * flattened_h + h
    
    # Store result
    tl.store(out_ptr + transposed_offset, x, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose(x, flatten_dim=2):
    original_shape = x.shape
    batch_size = original_shape[0]
    orig_channels = original_shape[1]
    height = original_shape[2]
    width = original_shape[3]
    
    # Calculate output shape
    flattened_h = height * width
    output_shape = (batch_size, orig_channels, flattened_h) if flatten_dim == 2 else original_shape
    
    # Verify this is a valid flattening operation
    if flatten_dim != 2:
        # Fall back to original implementation if not supported
        return x.flatten(flatten_dim).transpose(1, 2)
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid and block sizes
    total_elements = batch_size * orig_channels * flattened_h
    BLOCK_SIZE = 1024
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_flatten_transpose_kernel[grid](
        x, out,
        batch_size, orig_channels, height, width,
        flatten_dim, BLOCK_SIZE
    )
    
    return out

def pattern(in_0, in_1, in_2, in_3, in_4):
    # Flatten input at dimension 2
    tmp_7 = in_4.flatten(2)
    
    # Transpose dimensions 1 and 2  
    tmp_8 = tmp_7.transpose(1, 2)
    
    # Return just the second output - the pattern matching will handle the tuple
    return (tmp_8,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, 2)

def replacement_func():
    return fused_flatten_transpose