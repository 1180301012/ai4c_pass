import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    """Pattern matching torch.cat with 3 tensors along dim=1"""
    result = torch.cat([x, y, z], dim=1)
    return result

def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def cat_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    x_batch, x_channels, x_height, x_width,
    y_channels, z_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Triton kernel for concatenating 3 tensors along channel dimension"""
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    total_channels = x_channels + y_channels + z_channels
    out_height = x_height
    out_width = x_width
    
    # Each program handles a batch and a portion of channels
    batch_idx = pid // ((total_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    channel_offset = (pid % ((total_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)) * BLOCK_SIZE_N
    
    # Bounds checking
    if batch_idx >= x_batch:
        return
        
    channel_end = min(channel_offset + BLOCK_SIZE_N, total_channels)
    
    for c in range(channel_offset, channel_end):
        for h in range(out_height):
            for w in range(out_width):
                offset = batch_idx * total_channels * out_height * out_width + c * out_height * out_width + h * out_width + w
                
                if c < x_channels:
                    src_offset = batch_idx * x_channels * x_height * x_width + c * x_height * x_width + h * x_width + w
                    val = tl.load(x_ptr + src_offset)
                elif c < x_channels + y_channels:
                    src_offset = batch_idx * y_channels * x_height * x_width + (c - x_channels) * x_height * x_width + h * x_width + w
                    val = tl.load(y_ptr + src_offset)
                else:
                    src_offset = batch_idx * z_channels * x_height * x_width + (c - x_channels - y_channels) * x_height * x_width + h * x_width + w
                    val = tl.load(z_ptr + src_offset)
                
                tl.store(out_ptr + offset, val)

@torch.fx.wrap
def triton_cat(x, y, z):
    """Function that performs tensor concatenation using Triton kernel"""
    batch_size, x_channels, x_height, x_width = x.shape
    y_channels = y.shape[1]
    z_channels = z.shape[1]
    
    total_channels = x_channels + y_channels + z_channels
    output_shape = (batch_size, total_channels, x_height, x_width)
    
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (
        (batch_size * ((total_channels + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']),)
    )
    
    cat_kernel[grid](
        x, y, z, output,
        batch_size, x_channels, x_height, x_width,
        y_channels, z_channels,
        1, 64  # BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return triton_cat