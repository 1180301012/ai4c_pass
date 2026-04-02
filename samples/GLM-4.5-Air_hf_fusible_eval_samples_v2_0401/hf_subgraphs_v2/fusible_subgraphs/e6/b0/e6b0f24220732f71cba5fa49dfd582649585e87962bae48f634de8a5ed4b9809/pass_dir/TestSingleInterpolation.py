import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Simple pattern to match a single interpolation operation"""
    return torch.nn.functional.interpolate(input_tensor, size=(40, 40), mode='nearest')

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def simple_interpolation_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    target_height, target_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple nearest neighbor interpolation kernel"""
    pid = tl.program_id(0)
    
    total_elements = batch_size * channels * target_height * target_width
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        batch_idx = idx // (channels * target_height * target_width)
        remainder = idx % (channels * target_height * target_width)
        channel_idx = remainder // (target_height * target_width)
        h_idx = remainder % target_height // target_width
        w_idx = remainder % target_width
        
        # Calculate source coordinates for nearest neighbor
        src_h = h_idx * height // target_height
        src_w = w_idx * width // target_width
        
        src_offset = batch_idx * channels * height * width + channel_idx * height * width + src_h * width + src_w
        dst_offset = batch_idx * channels * target_height * target_width + channel_idx * target_height * target_width + h_idx * target_width + w_idx
        
        val = tl.load(input_ptr + src_offset)
        tl.store(output_ptr + dst_offset, val)

@torch.fx.wrap
def optimized_interpolation(input_tensor):
    """Optimized single interpolation function"""
    batch_size, channels, height, width = input_tensor.shape
    target_height, target_width = 40, 40
    
    output = torch.empty((batch_size, channels, target_height, target_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * channels * target_height * target_width
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_interpolation_kernel[grid_size](
        input_tensor, output,
        batch_size, channels, height, width,
        target_height, target_width,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_interpolation