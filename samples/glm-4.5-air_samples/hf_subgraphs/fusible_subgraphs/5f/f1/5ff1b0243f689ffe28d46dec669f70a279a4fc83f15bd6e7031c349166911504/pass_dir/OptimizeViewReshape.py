import torch
import triton
import triton.language as tl

@triton.jit
def view_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    in_h,
    in_w,
    out_h,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Each program handles a 2D tile of the output
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate input and output pointers for this batch/channel
    in_batch_ptr = in_ptr + batch_idx * channels * in_h * in_w + channel_idx * in_h * in_w
    out_batch_ptr = out_ptr + batch_idx * channels * out_h + channel_idx * out_h
    
    # Load 2D tile from input
    x_offsets = tl.arange(0, BLOCK_SIZE_X)
    y_offsets = tl.arange(0, BLOCK_SIZE_Y)
    x_mask = x_offsets < in_w
    y_mask = y_offsets < in_h
    mask = y_mask[:, None] & x_mask[None, :]
    
    # Load values
    in_vals = tl.load(in_batch_ptr + y_offsets[:, None] * in_w + x_offsets[None, :], 
                      mask=mask, other=0.0)
    
    # Store to output (flatten the spatial dimensions)
    out_offsets = tl.arange(0, BLOCK_SIZE_X * BLOCK_SIZE_Y)
    out_vals = in_vals.flatten()
    out_mask = out_offsets < out_h
    
    tl.store(out_batch_ptr + out_offsets, out_vals, mask=out_mask)

@torch.fx.wrap
def optimized_view(x, shape):
    # Handle the common case of flattening spatial dimensions
    if len(shape) == 3 and shape[0] == x.shape[0] and shape[1] == x.shape[1]:
        batch_size, channels, in_h, in_w = x.shape
        out_h = shape[2]
        
        assert out_h == in_h * in_w, "Output height must be product of input spatial dims"
        
        # Set block sizes
        BLOCK_SIZE_X = min(64, in_w)
        BLOCK_SIZE_Y = min(16, in_h)
        
        # Grid size: (batch_size, channels, 1)
        grid = (batch_size, channels, 1)
        
        # Create output tensor
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        
        # Launch kernel only if we have data to process
        if batch_size > 0 and channels > 0:
            view_kernel[grid](
                in_ptr=x,
                out_ptr=out,
                batch_size=batch_size,
                channels=channels,
                in_h=in_h,
                in_w=in_w,
                out_h=out_h,
                BLOCK_SIZE_X=BLOCK_SIZE_X,
                BLOCK_SIZE_Y=BLOCK_SIZE_Y,
            )
        
        return out
    
    # Fallback to standard view for other cases
    return x.view(*shape)

def pattern(x, y):
    # Match the view operation pattern
    tmp_5 = y.view(x.shape[0], x.shape[1], -1) if len(x.shape) == 3 else y.view(1, 512, -1)
    return x, tmp_5

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return optimized_view