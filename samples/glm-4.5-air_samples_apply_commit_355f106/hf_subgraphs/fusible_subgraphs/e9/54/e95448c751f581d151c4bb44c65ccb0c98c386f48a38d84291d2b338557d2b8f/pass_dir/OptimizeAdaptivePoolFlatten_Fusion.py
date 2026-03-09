import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_6, tmp_7

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def adaptive_pool_and_flatten_kernel(
    input_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel processing
    pid = tl.program_id(0)
    
    # Only process within valid batch size
    if pid >= batch_size:
        return
    
    # Calculate base pointers
    input_base = input_ptr + pid * in_channels * in_height * in_width
    out_base = out_ptr + pid * in_channels
    
    # Process each channel
    for c in range(in_channels):
        channel_sum = 0.0
        pixel_count = in_height * in_width
        
        # Compute sum across spatial dimensions
        for h in range(in_height):
            for w in range(in_width):
                input_val = tl.load(input_base + c * in_height * in_width + h * in_width + w).to(tl.float32)
                channel_sum += input_val
        
        # Compute mean (adaptive_avg_pool2d with size=1 is equivalent to mean)
        mean_val = channel_sum / pixel_count
        
        # Store result (flattened: single value per channel)
        tl.store(out_base + c, mean_val)

@torch.fx.wrap
def adaptive_pool_and_flatten_fusion(tmp_5):
    batch_size = tmp_5.shape[0] if len(tmp_5.shape) == 4 else 1
    in_channels = tmp_5.shape[1]
    in_height = tmp_5.shape[2] if len(tmp_5.shape) == 4 else 1
    in_width = tmp_5.shape[3] if len(tmp_5.shape) == 4 else 1
    
    # Create output tensors
    pooled_out = torch.empty((batch_size, in_channels, 1, 1), dtype=torch.float32, device=tmp_5.device)
    flat_out = torch.empty((batch_size, in_channels), dtype=torch.float32, device=tmp_5.device)
    
    # Set grid size
    grid = (batch_size,)
    
    # Launch kernel for pooling and flattening fusion
    adaptive_pool_and_flatten_kernel[grid](
        tmp_5,
        flat_out,
        batch_size,
        in_channels,
        in_height,
        in_width,
        BLOCK_SIZE=256,
    )
    
    return flat_out.view(batch_size, in_channels, 1, 1), flat_out

def replacement_func():
    return adaptive_pool_and_flatten_fusion