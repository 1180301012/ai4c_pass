import torch
import triton
import triton.language as tl

def pattern(x):
    relu_out = torch.nn.functional.relu(x, inplace=False)
    mean_out = relu_out.mean((2, 3), keepdim=True)
    return relu_out, mean_out

def replacement_args(x):
    return (x,)

@triton.jit
def relu_mean_kernel(
    x_ptr, relu_out_ptr, mean_out_ptr,
    batch_size, num_channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_elements = batch_size * num_channels * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_x = tl.maximum(x, 0.0)
    
    # Store ReLU output
    tl.store(relu_out_ptr + offsets, relu_x, mask=mask)
    
    # For reduction: accumulate mean in shared memory
    if pid == 0:  # Only first thread in each block computes reduction
        block_elements = min(BLOCK_SIZE, total_elements - block_start)
        sum_val = 0.0
        
        # Compute sum for this spatial block
        for i in range(block_elements):
            elem = tl.load(x_ptr + block_start + i, other=0.0)
            sum_val += tl.maximum(elem, 0.0)
        
        # Compute mean and store for corresponding batch/channel
        batch_idx = (block_start // (num_channels * height * width))
        channel_idx = ((block_start // (height * width)) % num_channels)
        mean_idx = batch_idx * num_channels + channel_idx
        
        mean_val = sum_val / (height * width)
        tl.store(mean_out_ptr + mean_idx, mean_val)

@torch.fx.wrap
def optimized_relu_mean(x):
    batch_size, num_channels, height, width = x.shape
    total_elements = batch_size * num_channels * height * width
    
    # Create output tensors
    relu_out = torch.empty_like(x)
    mean_out = torch.empty((batch_size, num_channels, 1, 1), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(total_elements, 1024), )
    relu_mean_kernel[grid](
        x, relu_out, mean_out,
        batch_size, num_channels, height, width,
        BLOCK_SIZE=1024
    )
    
    return relu_out, mean_out

def replacement_func():
    return optimized_relu_mean