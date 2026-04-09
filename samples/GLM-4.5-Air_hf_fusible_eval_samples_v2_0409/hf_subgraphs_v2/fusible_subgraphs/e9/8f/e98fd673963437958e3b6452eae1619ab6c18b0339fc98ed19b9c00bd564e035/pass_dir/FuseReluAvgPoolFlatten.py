import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def fused_relu_avgpool_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Calculate program indices
    pid_c = tl.program_id(0)  # channel dimension
    pid_h = tl.program_id(1)  # height dimension
    
    # Calculate offsets
    c_offset = pid_c * BLOCK_SIZE_C
    h_offset = pid_h * BLOCK_SIZE_H
    
    # Create masks
    c_mask = c_offset < channels
    h_mask = h_offset < height
    
    # Initialize accumulator for each position
    sum_val = 0.0
    
    # Process each spatial position for this channel
    for c in range(BLOCK_SIZE_C):
        c_idx = c_offset + c
        if c_idx >= channels:
            continue
            
        for h in range(BLOCK_SIZE_H):
            h_idx = h_offset + h
            if h_idx >= height:
                continue
            
            # Calculate memory offset: batch_size=1, so offset = channel * height * width + height_idx * width + width_idx
            # Since width=1 for adaptive pooling result, this simplifies to: c_idx * height + h_idx
            spatial_offset = c_idx * height + h_idx
            
            # Load input value and apply ReLU
            in_val = tl.load(in_ptr + spatial_offset, mask=True, other=0.0)
            relu_val = tl.maximum(in_val, 0.0)
            
            # Accumulate for averaging (adaptive pooling to 1x1)
            sum_val += relu_val
    
    # Compute mean (adaptive average pooling to 1x1)
    # Since we're pooling over height and width, and we process all positions,
    # we need to count how many positions we processed
    count = BLOCK_SIZE_C * BLOCK_SIZE_H
    avg_val = sum_val / count
    
    # Store the result at channel position (since we flatten to [1, channels])
    if c_mask and h_mask:
        tl.store(out_ptr + c_offset, avg_val)

@torch.fx.wrap
def fused_relu_avgpool_flatten(input_tensor):
    # Get tensor shapes
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensor [1, channels]
    out = torch.empty((1, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - note that for adaptive pooling to 1x1, we process
    # all spatial positions for each channel
    grid = lambda meta: (triton.cdiv(channels, meta['BLOCK_SIZE_C']), 
                        triton.cdiv(height, meta['BLOCK_SIZE_H']))
    
    fused_relu_avgpool_kernel[grid](
        in_ptr=input_tensor,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_C=256,
        BLOCK_SIZE_H=8,
    )
    
    return out

def replacement_func():
    return fused_relu_avgpool_flatten