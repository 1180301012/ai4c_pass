import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Note: The slice(None, X, None) is specific to each target graph and X varies
    # This pattern will match the concatenation + slice operations, but we need 
    # to handle the specific slice count in the kernel
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_concat_slice_mean_kernel(
    x0_ptr, x1_ptr,
    out_ptr, mean_ptr,
    batch_size, in_channels0, in_channels1, height, width,
    slice_channels,
    dtype,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Load slice parameters based on data type
    if dtype == 0:  # float32
        slice_factor = tl.float32(1.0 / (slice_channels * height * width))
    elif dtype == 1:  # float16  
        slice_factor = tl.float16(1.0 / (slice_channels * height * width))
    else:  # bfloat16
        slice_factor = tl.float16(1.0 / (slice_channels * height * width))
    
    # Each program handles one batch element
    pid = tl.program_id(0)
    batch_offset = pid * in_channels0 * height * width
    
    # Calculate the channels to process
    channels_per_program = (slice_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N * BLOCK_SIZE_N
    channel_offset = 0
    
    # Initialize mean accumulator
    mean_sum = tl.zeros([], dtype=tl.float32) if dtype == 0 else tl.zeros([], dtype=tl.float16)
    
    # Process the slice in blocks
    for block_start in range(0, slice_channels, BLOCK_SIZE_N):
        block_end = min(block_start + BLOCK_SIZE_N, slice_channels)
        offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < slice_channels
        
        # Load data with bounds checking
        if block_start < in_channels0:
            # Load from first input tensor
            spatial_offset = (height // BLOCK_SIZE_M) * BLOCK_SIZE_M
            for h in range(0, height, BLOCK_SIZE_M):
                for w in range(0, width, BLOCK_SIZE_N):
                    offset = batch_offset + block_start * height * width + h * width + w
                    ptr = x0_ptr + offset
                    
                    # Load element with bounds checking
                    val = tl.load(ptr + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0)
                    mean_sum += val.to(tl.float32 if dtype == 0 else tl.float16) * slice_factor
        else:
            # Load from second input tensor
            block_idx_in_second = block_start - in_channels0
            spatial_offset = (height // BLOCK_SIZE_M) * BLOCK_SIZE_M
            for h in range(0, height, BLOCK_SIZE_M):
                for w in range(0, width, BLOCK_SIZE_N):
                    offset = batch_offset + block_start * height * width + h * width + w
                    ptr = x1_ptr + offset
                    
                    # Load element with bounds checking
                    val = tl.load(ptr + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0)
                    mean_sum += val.to(tl.float32 if dtype == 0 else tl.float16) * slice_factor

    # Write output
    pid = tl.program_id(0)
    out_offset = pid * slice_channels * height * width
    mean_offset = pid
    
    # Store mean
    tl.store(mean_ptr + mean_offset, mean_sum, mask=True)
    return

@torch.fx.wrap
def fused_concat_slice_mean(x0, x1):
    # Get input shapes and determine slice size
    batch_size, in_channels0, height, width = x0.shape
    in_channels1 = x1.shape[1]
    
    # Set slice size based on the larger input's channels
    slice_channels = max(in_channels0, in_channels1)
    
    # Create output tensors
    out_shape = (batch_size, slice_channels, height, width)
    mean_shape = (batch_size, 1, 1, 1)
    
    out = torch.empty(out_shape, dtype=x0.dtype, device=x0.device)
    mean = torch.empty(mean_shape, dtype=x0.dtype, device=x0.device)
    
    # Map dtype to index for kernel
    if x0.dtype == torch.float32:
        dtype_idx = 0
    elif x0.dtype == torch.float16:
        dtype_idx = 1
    else:  # bfloat16
        dtype_idx = 2
    
    # Set grid and launch kernel
    grid = (batch_size,)
    BLOCK_SIZE_M = 8  # Height blocking factor
    BLOCK_SIZE_N = 32  # Width blocking factor
    
    fused_concat_slice_mean_kernel[grid](
        x0, x1, out, mean.flatten(),
        batch_size, in_channels0, in_channels1, height, width,
        slice_channels,
        dtype_idx,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out, mean

def replacement_func():
    return fused_concat_slice_mean