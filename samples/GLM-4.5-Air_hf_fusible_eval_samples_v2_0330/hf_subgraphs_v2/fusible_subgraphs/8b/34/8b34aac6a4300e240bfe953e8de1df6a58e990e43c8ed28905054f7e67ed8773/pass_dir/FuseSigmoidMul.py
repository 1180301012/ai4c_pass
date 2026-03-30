import torch
import triton
import triton.language as tl

def pattern(input_tensor, multiplier_tensor):
    """
    Pattern: Sigmoid followed by element-wise multiplication
    Optimization: Fuse both operations in a single Triton kernel
    """
    tmp_sigmoid = torch.sigmoid(input_tensor)
    tmp_result = multiplier_tensor * tmp_sigmoid
    return tmp_result

def replacement_args(input_tensor, multiplier_tensor):
    return (input_tensor, multiplier_tensor)

@triton.jit
def sigmoid_mul_fused_kernel(
    input_ptr, multiplier_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Load input and multiplier data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    multiplier_data = tl.load(multiplier_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse sigmoid and multiplication: result = multiplier * sigmoid(input)
    # Sigmoid formula: 1 / (1 + exp(-x))
    sigmoid_result = 1.0 / (1.0 + tl.exp(-input_data))
    fused_result = multiplier_data * sigmoid_result
    
    # Store result
    tl.store(output_ptr + offsets, fused_result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiply(input_tensor, multiplier_tensor):
    # Ensure tensors have the same shape
    if input_tensor.shape != multiplier_tensor.shape:
        raise ValueError(f"Input tensors must have the same shape: {input_tensor.shape} vs {multiplier_tensor.shape}")
    
    batch_size, channels, height, width = input_tensor.shape
    total_elements = batch_size * channels * height * width
    
    out = torch.empty_like(input_tensor)
    
    # Set up grid
    BLOCK_SIZE = 1024  # Optimal block size for GPU execution
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    sigmoid_mul_fused_kernel[grid_size](
        input_tensor,
        multiplier_tensor,
        out,
        batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_sigmoid_multiply