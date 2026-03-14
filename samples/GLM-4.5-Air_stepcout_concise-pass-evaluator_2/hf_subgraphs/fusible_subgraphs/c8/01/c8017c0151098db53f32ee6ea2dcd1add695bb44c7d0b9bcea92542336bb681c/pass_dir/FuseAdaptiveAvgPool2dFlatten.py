import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    return tmp_6

@triton.jit
def fused_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch_size * channels
    
    if offset < batch_size * channels:
        # Calculate output index (flatten batch and channels for parallel processing)
        output_idx = offset
        
        # Compute mean across spatial dimensions using efficient reduction
        sum_val = 0.0
        count = height * width
        
        # Iterate over all spatial locations to compute mean
        for h in range(height):
            for w in range(width):
                # Calculate input index for current spatial location
                input_idx = offset + (h * width + w) * batch_size * channels
                if input_idx < input_ptr.shape[0]:  # Safety check
                    val = tl.load(input_ptr + input_idx, mask=input_idx < input_ptr.shape[0])
                    sum_val += val
        
        # Store the mean value
        mean_val = sum_val / float(count)
        tl.store(output_ptr + output_idx, mean_val, mask=mask)

@torch.fx.wrap
def fused_adaptive_avg_pool2d_flatten(input):
    batch_size, channels, height, width = input.shape
    
    # For this specific case of adaptive pooling to 1x1, it's equivalent to mean pooling
    # followed by flattening the spatial dimensions
    if height == width == 1:
        # If already 1x1, just flatten the spatial dimensions
        return input.squeeze(-1).squeeze(-1)
    else:
        # Compute mean across spatial dimensions
        return input.flatten(2).mean(dim=2)

def replacement_args(tmp_4):
    return (tmp_4,)

def replacement_func():
    return fused_adaptive_avg_pool2d_flatten