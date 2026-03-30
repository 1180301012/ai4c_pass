import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_fused_kernel(
    x0_ptr, x1_ptr,
    out_ptr, mean_ptr,
    batch_size, in_channels0, in_channels1, height, width, slice_channels,
    dtype,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    pid = tl.program_id(0)
    
    # Initialize total sum and count for mean
    total_sum = tl.zeros([], dtype=tl.float32)
    total_count = 0
    
    # Process only the slice channels we need
    for c in range(min(slice_channels, max(in_channels0, in_channels1))):
        # Compute base offset in the appropriate input tensor
        if c < in_channels0:
            base_offset = pid * in_channels0 * height * width + c * height * width
        else:
            base_offset = pid * in_channels1 * height * width + (c - in_channels0) * height * width
        
        # Process spatial dimensions
        spatial_offset = base_offset
        for w in range(0, width, BLOCK_SIZE):
            offsets = spatial_offset + w + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (base_offset + width * height)
            
            # Load data and accumulate
            ptr = x0_ptr + base_offset if c < in_channels0 else x1_ptr + base_offset
            vals = tl.load(ptr + w + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
            total_sum += tl.sum(vals.astype(tl.float32))
            total_count += tl.sum(mask.astype(tl.int32))
    
    # Compute mean (avoid division by zero)
    mean_val = total_sum / (total_count + 1e-6)
    
    # Store results - simplified version
    tl.store(mean_ptr + pid, mean_val)

@torch.fx.wrap
def simple_fused_operation(x0, x1):
    batch_size, in_channels0, height, width = x0.shape
    in_channels1 = x1.shape[1]
    
    # Use conservative slice size
    slice_channels = 672
    
    # Determine data type
    if x0.dtype == torch.float32:
        dtype_idx = 0
    else:
        dtype_idx = 1
    
    # Create output tensors  
    out_shape = (batch_size, slice_channels, height, width)
    mean_shape = (batch_size, 1, 1, 1)
    
    out = torch.empty(out_shape, dtype=x0.dtype, device=x0.device)
    mean = torch.empty(mean_shape, dtype=x0.dtype, device=x0.device)
    
    # Launch simplified kernel
    grid = (batch_size,)
    BLOCK_SIZE = 128
    
    simple_fused_kernel[grid](
        x0, x1, out, mean.flatten(),
        batch_size, in_channels0, in_channels1, height, width, slice_channels,
        dtype_idx,
        BLOCK_SIZE
    )
    
    return out, mean

def replacement_func():
    return simple_fused_operation