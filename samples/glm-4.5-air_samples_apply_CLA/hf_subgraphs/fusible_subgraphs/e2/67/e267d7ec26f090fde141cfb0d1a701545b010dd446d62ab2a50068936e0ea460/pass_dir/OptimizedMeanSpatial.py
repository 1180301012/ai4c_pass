import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_2

def replacement_args(tmp_1):
    return (tmp_1,)

@triton.jit
def mean_kernel_2d(
    input_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Create program IDs for batch and channel dimensions
    n = tl.program_id(0)
    c = tl.program_id(1)
    
    # Calculate spatial block size based on H,W dimensions
    spatial_elements = H * W
    BLOCK_SIZE_HW = min(256, spatial_elements)  # Optimize based on spatial size
    spatial_programs = (spatial_elements + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Initialize accumulator for channel c in batch n
    sum_val = tl.zeros(1, dtype=tl.float32)
    
    # Reduce across spatial dimensions
    for hw_pid in range(spatial_programs):
        hw_start = hw_pid * BLOCK_SIZE_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < spatial_elements
        
        # Calculate spatial indices
        w = hw_offsets % W
        h = hw_offsets // W
        
        # Calculate global memory offset
        offset = n * (C * H * W) + c * (H * W) + h * W + w
        
        # Load element and accumulate
        val = tl.load(input_ptr + offset, mask=hw_mask, other=0.0)
        sum_val += val
    
    # Calculate mean
    mean_val = sum_val / float(spatial_elements)
    
    # Store result at output location [n, c, 0, 0]
    output_offset = n * C + c
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_spatial_mean(input_tensor):
    # Get tensor dimensions
    N, C, H, W = input_tensor.shape
    
    # Check if we need to optimize (small spatial dimensions might benefit more)
    if H * W < 1024:
        # For very small spatial dimensions, use optimized grid
        spatial_elements = H * W
        BLOCK_SIZE_HW = min(256, spatial_elements)
        num_hw_programs = (spatial_elements + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
        
        # Output shape is [N, C, 1, 1]
        output = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Use optimized grid based on batch and channel dimensions
        mean_kernel_2d[(N, C)](
            input_tensor,
            output.view(N, C),  # Flatten spatial dimensions for output
            N, C, H, W,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW
        )
        
    else:
        # For larger spatial dimensions, use more aggressive grid
        BLOCK_SIZE_HW = 512
        num_hw_programs = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
        
        output = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
        
        mean_kernel_2d[(N, C)](
            input_tensor,
            output.view(N, C),  # Flatten spatial dimensions for output
            N, C, H, W,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW
        )
    
    return output

def replacement_func():
    return optimized_spatial_mean