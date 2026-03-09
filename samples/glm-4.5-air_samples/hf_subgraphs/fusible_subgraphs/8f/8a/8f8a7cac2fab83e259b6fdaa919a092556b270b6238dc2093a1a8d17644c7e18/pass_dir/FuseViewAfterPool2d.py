import torch
import triton
import triton.language as tl

def pattern(x):
    # Match: view -> flatten sequence after adaptive pooling
    # This pattern captures the essential optimization opportunity
    tmp_2 = x.view(x.shape[0], -1)
    out = torch.flatten(tmp_2, 1)
    return out

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # Calculate total elements per batch (channels * 1 * 1 = channels)
    n_elements = channels
    
    # Load the flattened data (after adaptive pooling, spatial dims are 1x1)
    offsets = pid * n_elements + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * n_elements)
    
    # Load data - adaptive_avg_pool2d already reduces to [batch, channels, 1, 1]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store directly - equivalent to flatten(1) after adaptive pool
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input
    x = tl.load(x_ptr + offsets)
    
    # Apply hardtanh: max(0, min(x, 6.0))
    out = tl.maximum(tl.minimum(x, max_val), min_val)
    
    # Store result
    tl.store(out_ptr + offsets, out)

@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location in output
    pid = tl.program_id(0)
    batch_idx = pid // (channels * output_height * output_width)
    channel_idx = (pid // (output_height * output_width)) % channels
    h_idx = (pid // output_width) % output_height
    w_idx = pid % output_width
    
    if batch_idx >= batch_size:
        return
    
    # For adaptive_avg_pool2d to (1,1), we need to average over entire spatial domain
    total_elements = height * width
    sum_val = 0.0
    
    # Simple averaging loop (could be optimized more)
    for h in range(height):
        for w in range(width):
            # Calculate input offset
            input_offset = (batch_idx * channels * height * width + 
                          channel_idx * height * width + h * width + w)
            
            if input_offset < batch_size * channels * height * width:
                val = tl.load(x_ptr + input_offset)
                sum_val += val
    
    # Average
    avg_val = sum_val / total_elements
    
    # Store output
    output_offset = (batch_idx * channels * output_height * output_width + 
                    channel_idx * output_height * output_width + 
                    h_idx * output_width + w_idx)
    tl.store(out_ptr + output_offset, avg_val)

@torch.fx.wrap
def direct_flatten(x):
    # Input: [batch, channels, 1, 1] - tensor after adaptive pooling
    # Output: [batch, channels] - flattened directly (eliminate view operation)
    batch_size, channels, height, width = x.shape
    
    assert height == 1 and width == 1, "This pass expects input to be already pooled to (1,1)"
    
    BLOCK_SIZE = 1024
    
    # Direct flatten from dimension 1 (eliminate view operation)
    x_contiguous = x.contiguous()
    out_shape = (batch_size, channels)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Copy data directly from [batch, channels, 1, 1] -> [batch, channels]
    flatten_elements = batch_size * channels
    num_flatten_programs = (flatten_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel[(num_flatten_programs,)](
        x_ptr=x_contiguous,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return direct_flatten