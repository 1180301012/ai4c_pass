import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching: ReLU + 3 identical max_pool2d + concatenation"""
    relu_out = torch.nn.functional.relu(x, inplace=True)
    pool1 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    pool2 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    pool3 = torch.nn.functional.max_pool2d(relu_out, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    concat_out = torch.cat([relu_out, pool1, pool2, pool3], 1)
    return concat_out

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

@triton.jit
def fused_relu_3way_maxpool_kernel(
    x_ptr,
    relu_out_ptr,
    pool1_out_ptr, 
    pool2_out_ptr,
    pool3_out_ptr,
    batch_size,
    channels,
    height,
    width,
):
    """Fused kernel: ReLU + 3-way max-pooling with 5x5 kernel, stride 1, padding 2"""
    
    # Each thread handles one pixel location for one batch element
    b = tl.program_id(2)
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    # Process spatially for each channel
    for c in range(channels):
        base_idx = b * channels * height * width + c * height * width + i * width + j
        
        # Output 1: ReLU
        relu_val = tl.load(x_ptr + base_idx)
        relu_out = tl.maximum(relu_val, 0.0)
        tl.store(relu_out_ptr + base_idx, relu_out)
        
        # Output 2-4: Max-pooling over 5x5 neighborhood
        pady, padx = 2, 2  # padding of 2 on each side
        
        # Initialize max values
        max_val1 = -float('inf')
        max_val2 = -float('inf') 
        max_val3 = -float('inf')
        
        # Iterate over 5x5 neighborhood
        for dy in range(-pady, pady + 1):
            for dx in range(-padx, padx + 1):
                ni, nj = i + dy, j + dx
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_idx = b * channels * height * width + c * height * width + ni * width + nj
                    val = tl.load(x_ptr + neighbor_idx)
                    relu_val = tl.maximum(val, 0.0)
                    
                    # Three identical pooling operations
                    if dy == 0 and dx == 0:  # center pixel
                        max_val1 = relu_val
                        max_val2 = relu_val
                        max_val3 = relu_val
                    else:
                        max_val1 = tl.maximum(max_val1, relu_val)
                        max_val2 = tl.maximum(max_val2, relu_val)
                        max_val3 = tl.maximum(max_val3, relu_val)
        
        # Store max-pooling results
        tl.store(pool1_out_ptr + base_idx, max_val1)
        tl.store(pool2_out_ptr + base_idx, max_val2)
        tl.store(pool3_out_ptr + base_idx, max_val3)

@torch.fx.wrap
def fused_relu_3way_maxpool(x):
    """Wrapper function for the fused kernel"""
    batch_size, channels, height, width = x.shape
    
    # Output tensors
    relu_out = torch.empty_like(x)
    pool1_out = torch.empty_like(x)
    pool2_out = torch.empty_like(x)
    pool3_out = torch.empty_like(x)
    
    # Grid configuration: (height, width, batch_size)
    # Each thread handles one (row, col) location for one batch element
    grid = (height, width, batch_size)
    
    # Launch kernel
    fused_relu_3way_maxpool_kernel[grid](
        x_ptr=x,
        relu_out_ptr=relu_out,
        pool1_out_ptr=pool1_out,
        pool2_out_ptr=pool2_out,
        pool3_out_ptr=pool3_out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    
    # Concatenate all outputs along channel dimension
    return torch.cat([relu_out, pool1_out, pool2_out, pool3_out], dim=1)

def replacement_func():
    """Replacement function returns the fused implementation"""
    return fused_relu_3way_maxpool