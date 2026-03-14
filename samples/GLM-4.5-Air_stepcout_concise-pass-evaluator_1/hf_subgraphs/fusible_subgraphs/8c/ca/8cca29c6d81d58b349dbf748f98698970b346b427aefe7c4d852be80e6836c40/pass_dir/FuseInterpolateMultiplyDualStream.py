import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the exact computation pattern from model.py with flexible sizes
    # Note: The sizes will be captured by the graph compiler during matching
    tmp_0 = torch.nn.functional.interpolate(in_0, size=(64, 48), mode='nearest')
    tmp_1 = in_2 * tmp_0
    tmp_0 = None
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_3 = in_3 * tmp_2
    tmp_2 = None
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def interpolate_multiply_kernel_single(
    # Inputs
    input_ptr, scale_ptr, output_ptr,
    batch_size, channels, 
    in_height, in_width, out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one spatial position
    pid = tl.program_id(0)
    
    # Calculate linear index for output tensor
    total_elements = batch_size * channels * out_height * out_width
    if pid >= total_elements:
        return
    
    # Extract dimensions from linear index
    batch = pid // (channels * out_height * out_width)
    remaining = pid % (channels * out_height * out_width)
    ch = remaining // (out_height * out_width)
    h = remaining // out_width
    w = remaining % out_width
    
    if batch < batch_size and ch < channels and h < out_height and w < out_width:
        # Calculate nearest neighbor indices in input
        src_h = (h * in_height) // out_height
        src_w = (w * in_width) // out_width
        
        # Load input value (interpolated via nearest neighbor)
        input_idx = ((batch * channels + ch) * in_height + src_h) * in_width + src_w
        input_val = tl.load(input_ptr + input_idx, eviction_policy='evict_last')
        
        # Load scale value  
        scale_idx = pid  # scale has same shape as output
        scale_val = tl.load(scale_ptr + scale_idx, eviction_policy='evict_last')
        
        # Multiply and store
        result = input_val * scale_val
        tl.store(output_ptr + pid, result, eviction_policy='evict_last')

@torch.fx.wrap
def compute_stream1(in_0, in_2, out_size_0):
    """Process stream 1: interpolate in_0 and multiply by in_2"""
    out_shape0 = in_0.shape[:2] + out_size_0
    out1 = torch.empty(out_shape0, dtype=in_0.dtype, device=in_0.device)
    
    batch_size, channels, in_height, in_width = in_0.shape
    out_height, out_width = out_size_0
    
    # Total number of elements in output
    total_elements = batch_size * channels * out_height * out_width
    
    # Block size for GPU execution
    BLOCK_SIZE = 1024
    
    # Number of programs/threads needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    interpolate_multiply_kernel_single[(num_programs,)](
        input_ptr=in_0,
        scale_ptr=in_2,
        output_ptr=out1,
        batch_size=batch_size, channels=channels,
        in_height=in_height, in_width=in_width,
        out_height=out_height, out_width=out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out1

@torch.fx.wrap  
def compute_stream2(in_1, in_3, out_size_1):
    """Process stream 2: interpolate in_1 and multiply by in_3"""
    out_shape1 = in_1.shape[:2] + out_size_1
    out2 = torch.empty(out_shape1, dtype=in_1.dtype, device=in_1.device)
    
    batch_size, channels, in_height, in_width = in_1.shape
    out_height, out_width = out_size_1
    
    # Total number of elements in output
    total_elements = batch_size * channels * out_height * out_width
    
    # Block size for GPU execution
    BLOCK_SIZE = 1024
    
    # Number of programs/threads needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    interpolate_multiply_kernel_single[(num_programs,)](
        input_ptr=in_1,
        scale_ptr=in_3,
        output_ptr=out2,
        batch_size=batch_size, channels=channels,
        in_height=in_height, in_width=in_width,
        out_height=out_height, out_width=out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out2

def fused_interpolate_multiply(in_0, in_1, in_2, in_3, out_size_0, out_size_1):
    """Main function to process both streams"""
    # Compute each stream independently
    out1 = compute_stream1(in_0, in_2, out_size_0)
    out2 = compute_stream2(in_1, in_3, out_size_1)
    
    return (out1, out2)

def replacement_func():
    # Return a closure that can handle different interpolation sizes
    # The sizes will be determined by the matched graph's pattern
    return lambda in_0, in_1, in_2, in_3: fused_interpolate_multiply(in_0, in_1, in_2, in_3, (64, 48), (32, 24))