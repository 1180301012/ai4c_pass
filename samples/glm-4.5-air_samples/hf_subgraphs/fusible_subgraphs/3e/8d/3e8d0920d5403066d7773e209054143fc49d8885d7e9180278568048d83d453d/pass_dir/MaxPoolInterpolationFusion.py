import torch
import triton
import triton.language as tl

def pattern(x, scale_factor):
    tmp_5 = torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, scale_factor, None, 'bilinear', False)
    return tmp_6

def replacement_args(x, scale_factor):
    return (x, scale_factor)

@triton.jit
def max_pool_interpolate_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a batch*channel
    pid = tl.program_id(0)
    batch_idx = pid // in_channels
    channel_idx = pid % in_channels
    
    # Calculate output dimensions (max_pool2d 2x2 with stride 2)
    pool_height = in_height // 2
    pool_width = in_width // 2
    
    # Initialize pointers
    x_base = x_ptr + batch_idx * in_channels * in_height * in_width
    out_base = out_ptr + batch_idx * in_channels * out_height * out_width
    
    # Process spatial dimensions
    for h in range(0, out_height, BLOCK_SIZE):
        for w in range(0, out_width, BLOCK_SIZE):
            # Calculate block bounds
            h_end = min(h + BLOCK_SIZE, out_height)
            w_end = min(w + BLOCK_SIZE, out_width)
            
            # Process each output pixel
            for out_h in range(h, h_end):
                for out_w in range(w, w_end):
                    # Calculate corresponding input positions for bilinear interpolation
                    pool_h = out_h * 2  # interpolation doubles the size
                    pool_w = out_w * 2
                    
                    # Get the 4 input points for bilinear interpolation
                    h1 = min(pool_h, pool_height - 1)
                    w1 = min(pool_w, pool_width - 1)
                    h2 = min(h1 + 1, pool_height - 1)
                    w2 = min(w1 + 1, pool_width - 1)
                    
                    # Get the 4 corner pixels (after max_pool2d)
                    idx_h1_w1 = channel_idx * pool_height * pool_width + h1 * pool_width + w1
                    idx_h1_w2 = channel_idx * pool_height * pool_width + h1 * pool_width + w2
                    idx_h2_w1 = channel_idx * pool_height * pool_width + h2 * pool_width + w1
                    idx_h2_w2 = channel_idx * pool_height * pool_width + h2 * pool_width + w2
                    
                    # Load the 4 pixels
                    p1 = tl.load(x_base + idx_h1_w1, mask=True)
                    p2 = tl.load(x_base + idx_h1_w2, mask=True)
                    p3 = tl.load(x_base + idx_h2_w1, mask=True)
                    p4 = tl.load(x_base + idx_h2_w2, mask=True)
                    
                    # Calculate interpolation weights
                    if pool_h < pool_height:
                        alpha_h = (pool_h - h1) if pool_h < pool_height else 1.0
                    else:
                        alpha_h = 1.0
                    if pool_w < pool_width:
                        alpha_w = (pool_w - w1) if pool_w < pool_width else 1.0
                    else:
                        alpha_w = 1.0
                    
                    # Bilinear interpolation
                    val = (1-alpha_h) * (1-alpha_w) * p1 + alpha_h * (1-alpha_w) * p2 + (1-alpha_h) * alpha_w * p3 + alpha_h * alpha_w * p4
                    
                    # Store result
                    out_idx = channel_idx * out_height * out_width + out_h * out_width + out_w
                    tl.store(out_base + out_idx, val)

@torch.fx.wrap
def max_pool_interpolate_fusion(x, scale_factor):
    B, C, H, W = x.shape
    out_H, out_W = scale_factor
    
    # Calculate grid size
    BLOCK_SIZE = 16
    total_elements = B * C * out_H * out_W
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((B, C, out_H, out_W), dtype=x.dtype, device=x.device)
    
    max_pool_interpolate_kernel[grid_size](
        x_ptr=x,
        out_ptr=out,
        batch_size=B,
        in_channels=C,
        in_height=H,
        in_width=W,
        out_height=out_H,
        out_width=out_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return max_pool_interpolate_fusion