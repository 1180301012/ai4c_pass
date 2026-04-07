import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    """Match adaptive_avg_pool2d + flatten sequence"""
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(tmp_0):
    """Extract the input tensor to the sequence"""
    return (tmp_0,)

@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=1, kwargs={}),
        triton.Config(num_warps=8, num_stages=1, kwargs={}),
        triton.Config(num_warps=16, num_stages=1, kwargs={}),
        triton.Config(num_warps=4, num_stages=2, kwargs={}),
        triton.Config(num_warps=8, num_stages=2, kwargs={}),
        triton.Config(num_warps=16, num_stages=2, kwargs={}),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    SPATIAL_SIZE: tl.constexpr,
):
    """Global average pooling kernel that computes mean across spatial dimensions"""
    # Each program handles one or more elements with 1D grid
    pid = tl.program_id(0)
    output_size = N * C
    
    if pid >= output_size:
        return
    
    # Load entire feature map for this channel using a power-of-2 range
    spatial_offsets = tl.arange(0, 32)  # Power of 2
    channel_offset = pid * H * W
    
    # Load all spatial values for this channel with proper masking
    input_base_ptr = input_ptr + channel_offset
    mask = spatial_offsets < SPATIAL_SIZE  # SPATIAL_SIZE = H * W
    spatial_values = tl.load(input_base_ptr + spatial_offsets, mask=mask, other=0.0)
    
    # Compute mean across spatial dimensions
    spatial_sum = tl.sum(spatial_values)
    spatial_mean = spatial_sum / SPATIAL_SIZE
    
    # Store the result
    tl.store(output_ptr + pid, spatial_mean)

@torch.fx.wrap
def global_average_pooling_2d(x):
    """Global average pooling that flattens to [N, C]"""
    N, C, H, W = x.shape
    
    output = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    # Calculate grid size and spatial size
    output_size = N * C
    spatial_size = H * W
    
    # Use larger block size to reduce overhead
    BLOCK_SIZE = 512  # Larger block size
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with 1D grid but larger blocks
    global_avg_pool_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        SPATIAL_SIZE=spatial_size,
    )
    
    return output

def replacement_func():
    """Return global average pooling function"""
    return global_average_pooling_2d