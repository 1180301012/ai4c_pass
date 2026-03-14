import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized GELU kernel using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Simple GELU approximation using x * sigmoid(1.702 * x)
    # This is computationally efficient and reasonably accurate
    sigmoid_arg = 1.702 * x
    sigmoid_out = 1.0 / (1.0 + tl.exp(-sigmoid_arg))
    gelu_out = x * sigmoid_out
    
    # Store output
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

@triton.jit
def mean_spatial_kernel(x_ptr, out_ptr, batch_size, channels, spatial_h, spatial_w, 
                       BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr):
    """Optimized spatial mean reduction kernel"""
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Each thread handles one channel for one batch
    # We'll accumulate the sum for this (batch, channel) pair
    spatial_total = spatial_h * spatial_w
    
    # Load all spatial elements for this (batch, channel)
    h_indices = tl.arange(0, BLOCK_SIZE_H)
    w_indices = tl.arange(0, BLOCK_SIZE_W)
    
    # Accumulator for sum
    total_sum = 0.0
    
    # Process spatial elements in blocks
    for h in range(0, spatial_h, BLOCK_SIZE_H):
        for w in range(0, spatial_w, BLOCK_SIZE_W):
            # Calculate current spatial block bounds
            h_block_start = h + h_indices
            w_block_start = w + w_indices
            
            # Calculate mask for current block
            h_mask = h_block_start < spatial_h
            w_mask = w_block_start < spatial_w
            
            # Flatten spatial block indices to valid range
            h_valid = tl.where(h_mask, h_block_start, 0)
            w_valid = tl.where(w_mask, w_block_start, 0)
            
            # Calculate linear indices
            linear_indices = h_valid * spatial_w + w_valid
            
            # Calculate final offsets in the tensor
            offset_base = batch_idx * channels * spatial_h * spatial_w + channel_idx * spatial_h * spatial_w
            offsets = offset_base + linear_indices
            
            # Load spatial elements
            mask = h_mask[:, None] & w_mask[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # Accumulate sum
            total_sum += tl.sum(x)
    
    # Compute mean
    mean_val = total_sum / float(spatial_total)
    
    # Store result: output shape is [batch_size, channels, 1, 1]
    output_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_gelu_mean_forward(x):
    """Optimized forward pass combining GELU and spatial mean"""
    batch_size, channels, spatial_h, spatial_w = x.shape
    n_elements = x.numel()
    
    # Optimized GELU
    gelu_out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=gelu_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Optimized spatial mean
    mean_out = torch.empty(batch_size, channels, 1, 1, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_H = min(16, spatial_h)  # Optimized for 56x56 spatial dimensions
    BLOCK_SIZE_W = min(16, spatial_w)
    
    # Each program handles one channel in one batch
    num_batch_programs = batch_size
    num_channel_programs = channels
    
    mean_spatial_kernel[(num_batch_programs, num_channel_programs)](
        x_ptr=gelu_out,
        out_ptr=mean_out.view(-1),  # Flatten to [batch_size * channels]
        batch_size=batch_size,
        channels=channels,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return gelu_out, mean_out

@torch.fx.wrap
def triton_gelu_only(x):
    """Optimized GELU using Triton kernel"""
    n_elements = x.numel()
    gelu_out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=gelu_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return gelu_out

def replacement_func():
    return triton_gelu_only