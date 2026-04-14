import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern to match ReLU + Flatten(1, -1) sequence
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that applies ReLU and flattens the output
    Input shape: [batch_size, channels, 1, 1]  
    Output shape: [batch_size, channels]
    """
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    channel_idx = (pid // batch_size) % channels
    
    # Calculate linearized index
    offset = batch_idx * channels + channel_idx
    
    # Load input element (since spatial dims are 1,1, we just load directly)
    x_val = tl.load(x_ptr + offset, mask=offset < (batch_size * channels), other=0.0)
    
    # Apply ReLU
    out_val = tl.maximum(x_val, 0.0)
    
    # Store to output (flattened layout)
    tl.store(out_ptr + offset, out_val, mask=offset < (batch_size * channels))

@torch.fx.wrap
def fused_relu_flatten(x):
    """
    Wrapper function that launches the fused ReLU+flatten kernel
    """
    batch_size, channels, h, w = x.shape
    
    # Since we know the spatial dimensions are 1,1, 
    # we can just work with the batch and channels
    N = batch_size * channels
    
    # Determine optimal block sizes based on tensor size
    if N > 1024 * 64:  # Large tensors
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    elif N > 1024 * 16:  # Medium tensors  
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 16
    else:  # Small tensors
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 8
    
    BLOCK_SIZE = BLOCK_SIZE_M
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((batch_size, channels), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    fused_relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_relu_flatten