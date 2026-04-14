import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern: concatenate operation that can be fused with slice and mean"""
    tmp_0 = torch.cat([in_0, in_1], dim = 1)
    return tmp_0

def replacement_args(in_0, in_1):
    """Extract necessary arguments for replacement"""
    return (in_0, in_1)

@triton.jit
def fused_cat_slice_mean_kernel(
    in0_ptr, in1_ptr,
    out_ptr, sum_ptr, count_ptr,
    batch_size, channels0, channels1, height, width,
    slice_end,
    do_accumulate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    """Fused kernel for cat+slice+mean operations"""
    
    # Calculate grid position - 3D grid: batch, height, width
    m = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Check bounds for height and width
    if h >= height or w >= width:
        return
    
    # Process channels from both inputs without intermediate allocation
    c0_total = min(channels0, slice_end)  # How many channels to take from in_0
    c1_total = min(max(0, slice_end - channels0), channels1)  # How many from in_1
    
    # Process first input channels
    for c0 in range(0, c0_total):
        in0_val = tl.load(in0_ptr + m * channels0 * height * width + 
                         c0 * height * width + h * width + w,
                         mask=True, other=0.0)
        
        tl.store(out_ptr + m * slice_end * height * width + 
                c0 * height * width + h * width + w, in0_val)
        
        # Only accumulate for mean calculation if flag is set
        if do_accumulate:
            tl.atomic_add(sum_ptr + m * slice_end * height + c0 * height + h,
                         in0_val.to(tl.float32))
            tl.atomic_add(count_ptr + m * slice_end * height + c0 * height + h, 1)
    
    # Process second input channels
    for c1 in range(0, c1_total):
        in1_val = tl.load(in1_ptr + m * channels1 * height * width + 
                         c1 * height * width + h * width + w,
                         mask=True, other=0.0)
        
        tl.store(out_ptr + m * slice_end * height * width + 
                (c0_total + c1) * height * width + h * width + w, in1_val)
        
        # Only accumulate for mean calculation if flag is set
        if do_accumulate:
            tl.atomic_add(sum_ptr + m * slice_end * height + (c0_total + c1) * height + h,
                         in1_val.to(tl.float32))
            tl.atomic_add(count_ptr + m * slice_end * height + (c0_total + c1) * height + h, 1)

@torch.fx.wrap
def fused_cat_slice_mean(in_0, in_1):
    """Wrapper function for the fused kernel"""
    batch_size, channels0, height, width = in_0.shape
    channels1 = in_1.shape[1]
    total_channels = channels0 + channels1
    slice_end = total_channels  # Take all channels as per pattern
    
    # This function should return what the original concatenation would return
    # which is a tensor of shape [batch, total_channels, height, width]
    
    # Allocate output tensor using only allowed torch operations
    result = torch.empty((batch_size, total_channels, height, width), 
                        dtype=in_0.dtype, device=in_0.device)
    
    # Use the Triton kernel to perform concatenation
    # We'll reuse the kernel but just do concatenation without the slice/mean part
    # Pass empty tensors for accumulation (they won't be used when sum_ptr is None)
    dummy_sum = torch.zeros(0, device=in_0.device)
    dummy_count = torch.zeros(0, device=in_0.device)
    
    fused_cat_slice_mean_kernel[(batch_size, height, width)](
        in_0, in_1,
        result, dummy_sum, dummy_count,
        batch_size, channels0, channels1, height, width,
        total_channels,  # Take all channels
        False,  # Do not accumulate for mean calculation
        1, 1, 1,  # Block sizes
    )
    
    return result

def full_fused_cat_slice_mean(in_0, in_1):
    """Full fused function that handles both concat+slice+mean and returns final result"""
    batch_size, channels0, height, width = in_0.shape
    channels1 = in_1.shape[1]
    total_channels = channels0 + channels1
    slice_end = total_channels  # Take all channels as per pattern
    
    # Output tensors for full fusion
    out = torch.empty((batch_size, slice_end, height, width), dtype=in_0.dtype, device=in_0.device)
    sum_accum = torch.zeros((batch_size, slice_end, height), dtype=torch.float32, device=in_0.device)
    count_accum = torch.zeros((batch_size, slice_end, height), dtype=torch.float32, device=in_0.device)
    
    # Block size configuration
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1  
    BLOCK_SIZE_W = 1
    
    # Launch kernel for full fusion
    fused_cat_slice_mean_kernel[(batch_size, height, width)](
        in_0, in_1,
        out, sum_accum, count_accum,
        batch_size, channels0, channels1, height, width,
        slice_end,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_W,
    )
    
    # Compute mean spatially
    mean_spatial = sum_accum / (count_accum + 1e-6)
    final_mean = mean_spatial.reshape(batch_size, slice_end, height, 1)
    
    return out, final_mean

def replacement_func():
    """Return the fused optimization function"""
    return fused_cat_slice_mean