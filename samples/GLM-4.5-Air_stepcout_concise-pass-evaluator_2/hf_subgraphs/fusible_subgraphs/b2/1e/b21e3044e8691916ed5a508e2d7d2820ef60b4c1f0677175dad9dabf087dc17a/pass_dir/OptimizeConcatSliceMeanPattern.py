import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching: Concat + Slice + Mean computation
    The slice operation takes all channels after concatenation, making it redundant
    """
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement"""
    # Note: we capture the slice end parameter by using the fact that it equals
    # the sum of channel dimensions of the two input tensors
    return (in_0, in_1)



@torch.fx.wrap  
def optimized_concat_slice_mean(in_0, in_1):
    """
    Optimized implementation that eliminates redundant slice operation.
    Since slice takes all channels, we can directly concatenate and compute mean.
    """
    # Directly concatenate tensors (slice is redundant)
    concat_result = torch.cat([in_0, in_1], dim=1)
    
    # Compute mean over spatial dimensions using efficient Triton kernel
    batch_size, channels, height, width = concat_result.shape
    spatial_size = height * width
    
    # Output for mean computation
    mean_result = torch.empty((batch_size, channels, 1, 1), dtype=concat_result.dtype, device=concat_result.device)
    
    # Optimized Triton kernel for spatial mean computation
    @triton.jit
    def spatial_mean_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # Each program handles one channel across all batch elements
        program_id = tl.program_id(0)
        block_start = program_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data and compute spatial mean
        x_data = tl.load(x_ptr + offsets * spatial_size, mask=mask, other=0.0)
        spatial_sum = tl.sum(x_data, axis=0)
        spatial_mean = spatial_sum / spatial_size
        
        # Store result
        tl.store(out_ptr + offsets, spatial_mean)
    
    # Launch kernel for each batch*channel combination
    total_elements = batch_size * channels
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    spatial_mean_kernel[grid](
        concat_result,
        mean_result,
        total_elements,
        BLOCK_SIZE
    )
    
    return concat_result, mean_result

def replacement_func():
    """Return the optimized function"""
    return optimized_concat_slice_mean