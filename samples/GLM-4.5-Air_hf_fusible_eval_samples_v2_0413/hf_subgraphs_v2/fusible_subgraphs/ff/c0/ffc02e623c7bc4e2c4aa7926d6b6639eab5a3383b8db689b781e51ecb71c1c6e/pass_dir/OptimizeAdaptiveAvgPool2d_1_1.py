import torch
import triton
import triton.language as tl

def pattern(input_tensor, output_size=(1, 1)):
    """Pattern matching for adaptive_avg_pool2d with output (1, 1)"""
    result = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)
    return result

def replacement_args(input_tensor, output_size=(1, 1)):
    """Extract arguments for optimized adaptive avg pool2d kernel"""
    return (input_tensor,)

@triton.jit
def adaptive_avg_pool_1_1_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for adaptive avg pool2d with output (1, 1)"""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, N * C)
    
    for idx in range(start_idx, end_idx):
        n = idx // C
        c = idx % C
        
        # Compute mean over spatial dimensions H x W for each n, c
        sum_val = 0.0
        spatial_elements = H * W
        
        for h in range(H):
            for w in range(W):
                input_offset = n * (C * H * W) + c * (H * W) + h * W + w
                val = tl.load(input_ptr + input_offset)
                sum_val += val
        
        # Store the mean value at output position [n, c, 0, 0]
        output_offset = n * (C * 1 * 1) + c * (1 * 1) + 0 * 1 + 0
        mean_val = sum_val / spatial_elements
        tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_adaptive_avg_pool2d_1_1(input_tensor):
    """Optimized adaptive avg pool2d for output size (1, 1)"""
    N, C, H, W = input_tensor.shape
    
    # Output shape is [N, C, 1, 1]
    output_shape = (N, C, 1, 1)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Custom kernel for computing mean over spatial dimensions
    BLOCK_SIZE = 256
    
    # Calculate grid size
    total_elements = N * C
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    adaptive_avg_pool_1_1_kernel[grid_size, 1](
        input_tensor,
        output_tensor,
        N, C, H, W,
        BLOCK_SIZE
    )
    
    return output_tensor

def replacement_func():
    return optimized_adaptive_avg_pool2d_1_1