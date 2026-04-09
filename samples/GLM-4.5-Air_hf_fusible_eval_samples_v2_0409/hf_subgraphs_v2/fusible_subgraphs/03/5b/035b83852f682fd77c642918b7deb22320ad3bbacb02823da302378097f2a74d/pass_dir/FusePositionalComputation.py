import torch
import triton
import triton.language as tl

def pattern():
    # Pattern that matches the core coordinate computation operations from the original model
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    return tmp_14

def replacement_args():
    return ()

@triton.jit
def coordinate_computation_kernel(
    output_ptr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for coordinate computation and squared distance"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, spatial_size * spatial_size)
    
    for idx in range(start_idx, end_idx):
        # Convert flat index to 2D coordinates
        y_pos = idx // spatial_size
        x_pos = idx % spatial_size
        
        # Compute coordinate differences
        dx = x_pos - y_pos
        dy = y_pos - x_pos
        
        # Store squared values (this is what the original computation produces)
        # The original model computes two different views of the coordinate differences
        # We'll compute both optimized ways
        radial_dist = dx * dx + dy * dy
        
        # Store to output tensor [spatial_size, spatial_size] equivalent
        output_idx = y_pos * spatial_size + x_pos
        tl.store(output_ptr + output_idx, radial_dist)

@torch.fx.wrap
def optimized_coordinate_computation():
    # Optimized computation of the coordinate operations
    spatial_size = 14
    total_elements = spatial_size * spatial_size
    
    # Use optimal block size
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(spatial_size, spatial_size, dtype=torch.float32, device='cuda')
    
    # Launch optimized kernel
    coordinate_computation_kernel[(num_programs,)](
        output_ptr=output,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_coordinate_computation