import torch
import triton
import triton.language as tl

def pattern(x):
    # Match: ReLU -> Flatten (drop the problematic dropout for now)
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_2 = tmp_0.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Each thread processes one channel
    if offset < channels:
        # Load the value (same for all spatial positions since they're 1x1)
        val = tl.load(x_ptr + offset, dtype=tl.float32)
        
        # Apply ReLU
        relu_val = tl.maximum(val, 0.0)
        
        # Store the result (flatten operation since we're going from [N, C, 1, 1] to [N, C])
        tl.store(out_ptr + offset, relu_val, dtype=tl.float32)

@torch.fx.wrap
def optimized_relu_flatten(x):
    # Input shape: [batch, channels, 1, 1]
    # Output shape: [batch, channels]
    
    batch_size, channels, height, width = x.shape
    assert height == 1 and width == 1, "Only supported for 1x1 spatial dimensions"
    
    # Reshape input to [batch * channels] for processing
    x_flat = x.reshape(-1)  # [batch * channels]
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * channels
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_relu_flatten_kernel[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [batch, channels]
    return out_flat.reshape(batch_size, channels)

def replacement_func():
    return optimized_relu_flatten