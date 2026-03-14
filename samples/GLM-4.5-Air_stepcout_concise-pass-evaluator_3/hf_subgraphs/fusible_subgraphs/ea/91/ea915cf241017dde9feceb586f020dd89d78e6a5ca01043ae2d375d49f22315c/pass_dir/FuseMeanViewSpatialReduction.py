import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    x_ptr,           # Input tensor pointer [1, 768, H, W]
    out_relu_ptr,    # Output for ReLU [1, 768, H, W]  
    out_mean_ptr,    # Output for mean [1, 1, 768]
    n_channels,      # Number of channels (768)
    n_height,        # Height dimension (32 or 8)
    n_width,         # Width dimension (32 or 8)
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one channel (simpler approach)
    pid = tl.program_id(0)
    if pid >= n_channels:
        return
    
    # Process all spatial positions for this channel
    spatial_size = n_height * n_width
    relu_sum = 0.0
    
    for h in range(n_height):
        for w in range(n_width):
            # Load input element
            input_val = tl.load(x_ptr + pid * spatial_size + h * n_width + w, 
                              mask=True, other=0.0)
            
            # Apply ReLU and store
            relu_val = tl.maximum(input_val, 0.0)
            tl.store(out_relu_ptr + pid * spatial_size + h * n_width + w, relu_val)
            
            # Accumulate for mean
            relu_sum += relu_val
    
    # Compute and store mean
    mean_val = relu_sum / spatial_size
    tl.store(out_mean_ptr + pid, mean_val)

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr
):
    pid = tl.program_id(0)
    block_size = tl.num_programs(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    block_size = 1024
    n_programs = (n_elements + block_size - 1) // block_size
    
    relu_kernel[(n_programs,)](
        in_0,
        out,
        n_elements
    )
    
    return out

def replacement_func():
    return optimized_relu