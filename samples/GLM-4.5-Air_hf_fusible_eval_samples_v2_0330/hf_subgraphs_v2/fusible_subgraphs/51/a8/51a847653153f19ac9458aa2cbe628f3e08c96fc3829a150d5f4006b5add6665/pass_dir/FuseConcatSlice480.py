import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 480, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel_480(
    x0_ptr, x1_ptr,
    out_ptr, mean_ptr,
    batch_size, in_channels0, in_channels1, height, width,
    dtype,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Slice parameters - always take first 480 channels
    slice_channels = 480
    if dtype == 0:  # float32
        slice_factor = tl.float32(1.0 / (slice_channels * height * width))
    else:  # float16, bfloat16
        slice_factor = tl.float16(1.0 / (slice_channels * height * width))
    
    # Each program handles one batch element
    pid = tl.program_id(0)
    
    # Initialize mean accumulator
    mean_sum = tl.zeros([], dtype=tl.float32) if dtype == 0 else tl.zeros([], dtype=tl.float16)
    
    # Process each channel in slices
    for c in range(slice_channels):
        channel_offset = c * height * width
        for h in range(0, height, BLOCK_SIZE_M):
            for w in range(0, width, BLOCK_SIZE_N):
                offsets = h * width + w + tl.arange(0, BLOCK_SIZE_N)
                w_mask = offsets < width
                
                # Load from appropriate input tensor
                if c < in_channels0:
                    # Load from first input
                    base_offset = pid * in_channels0 * height * width + c * height * width
                    ptr = x0_ptr + base_offset + offsets
                    val = tl.load(ptr, mask=w_mask, other=0.0)
                else:
                    # Load from second input
                    c_in_second = c - in_channels0
                    base_offset = pid * in_channels1 * height * width + c_in_second * height * width
                    ptr = x1_ptr + base_offset + offsets
                    val = tl.load(ptr, mask=w_mask, other=0.0)
                
                # Add to mean sum
                if dtype == 0:
                    mean_sum += val.astype(tl.float32) * slice_factor
                else:
                    mean_sum += val.astype(tl.float16) * slice_factor
    
    # Write outputs (simplified - just copy needed data)
    out_offset = pid * slice_channels * height * width
    for c in range(slice_channels):
        for h in range(height):
            for w in range(width):
                idx = out_offset + c * height * width + h * width + w
                if c < max(in_channels0, in_channels1):
                    # Calculate actual data location
                    if c < in_channels0:
                        src_offset = pid * in_channels0 * height * width + c * height * width + h * width + w
                        val = tl.load(x0_ptr + src_offset, mask=True, other=0.0)
                    else:
                        src_offset = pid * in_channels1 * height * width + (c - in_channels0) * height * width + h * width + w
                        val = tl.load(x1_ptr + src_offset, mask=True, other=0.0)
                    tl.store(out_ptr + idx, val)
                else:
                    tl.store(out_ptr + idx, tl.zeros([], dtype=dtype))
    
    # Store mean
    tl.store(mean_ptr + pid, mean_sum, mask=True)

@torch.fx.wrap
def fused_concat_slice_mean_480(x0, x1):
    # Get input shapes
    batch_size, in_channels0, height, width = x0.shape
    in_channels1 = x1.shape[1]
    
    # Determine data type
    if x0.dtype == torch.float32:
        dtype_idx = 0
    else:
        dtype_idx = 1  # float16 or bfloat16
    
    # Create output tensors (always slice to 480 channels)
    out_shape = (batch_size, 480, height, width)
    shape = (batch_size, 1, 1, 1)
    
    out = torch.empty(out_shape, dtype=x0.dtype, device=x0.device)
    mean = torch.empty(shape, dtype=x0.dtype, device=x0.device)
    
    # Launch kernel
    grid = (batch_size,)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    
    fused_kernel_480[grid](
        x0, x1, out, mean.flatten(),
        batch_size, in_channels0, in_channels1, height, width,
        dtype_idx,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out, mean

def replacement_func():
    return fused_concat_slice_mean_480