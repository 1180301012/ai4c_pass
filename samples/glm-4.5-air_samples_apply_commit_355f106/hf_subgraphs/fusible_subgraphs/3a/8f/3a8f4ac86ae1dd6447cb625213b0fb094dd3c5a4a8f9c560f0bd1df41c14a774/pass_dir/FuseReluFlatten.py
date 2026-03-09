import torch
import triton
import triton.language as tl

def pattern(x):
    # Match: ReLU -> Flatten (for 1x1 spatial dimensions)
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_2 = tmp_0.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Each thread handles multiple elements
    mask = offset < (batch_size * channels)
    
    # Load input data (flattened layout: batch_size * channels)
    x_vals = tl.load(x_ptr + offset, mask=mask, dtype=tl.float32)
    
    # Apply ReLU
    relu_vals = tl.maximum(x_vals, 0.0)
    
    # Store output
    tl.store(out_ptr + offset, relu_vals, mask=mask)

@torch.fx.wrap  
def fused_relu_flatten(x):
    # Input shape: [batch, channels, 1, 1]
    # Output shape: [batch, channels]
    
    batch_size, channels, height, width = x.shape
    
    # Reshape to flattened format for processing
    x_flat = x.reshape(-1)  # [batch_size * channels]
    out_flat = torch.empty_like(x_flat)
    
    # Use autotune for better performance
    BLOCK_SIZE = 1024
    total_elements = batch_size * channels
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_flatten_kernel[(num_programs, 1, 1)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [batch, channels]
    return out_flat.reshape(batch_size, channels)

def replacement_func():
    return fused_relu_flatten