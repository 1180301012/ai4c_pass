import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, sigmoid_broadcasted):
    # The pattern is: in_1 * sigmoid_broadcasted + in_0
    result = in_1 * sigmoid_broadcasted
    result += in_0
    return result

def replacement_args(in_0, in_1, sigmoid_broadcasted):
    return (in_0, in_1, sigmoid_broadcasted)

@triton.jit
def fused_add_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    sigmoid_ptr,
    out_ptr,
    num_channels,
    height,
    width,
    BLOCK_SIZE_channels: tl.constexpr,
    BLOCK_SIZE_height: tl.constexpr,
    BLOCK_SIZE_width: tl.constexpr,
):
    # Calculate block offsets  
    offsets = tl.arange(0, BLOCK_SIZE_channels * BLOCK_SIZE_height * BLOCK_SIZE_width)
    # Multi-dimensional indexing into the 3D spatial tensor
    c = offsets // (height * width) % num_channels
    h = offsets // width % height  
    w = offsets % width
    
    # Create mask for bounds checking
    mask = (c < num_channels) & (h < height) & (w < width)
    
    # Convert to flattened indices for memory access
    flat_indices = c * height * width + h * width + w
    
    # Load inputs
    in_0_val = tl.load(in_0_ptr + flat_indices, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + flat_indices, mask=mask, other=0.0)
    sigmoid_val = tl.load(sigmoid_ptr + flat_indices, mask=mask, other=0.0)
    
    # Fused operation: result = in_1 * sigmoid + in_0
    result = in_1_val * sigmoid_val + in_0_val
    
    # Store result
    tl.store(out_ptr + flat_indices, result, mask=mask)

@torch.fx.wrap
def optimized_fused_add_mul(in_0, in_1, sigmoid_broadcasted):
    # Get tensor dimensions
    num_channels, height, width = in_0.shape[1], in_0.shape[2], in_0.shape[3]
    total_elements = in_0.numel()
    
    # Optimal block size - should divide the tensor size efficiently
    BLOCK_SIZE = 1024  # Standard Triton block size for good occupancy
    
    # Calculate grid dimensions
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    fused_add_mul_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        sigmoid_ptr=sigmoid_broadcasted,
        out_ptr=out,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE_channels=BLOCK_SIZE,
        BLOCK_SIZE_height=1,
        BLOCK_SIZE_width=1,
    )
    
    return out

def replacement_func():
    return optimized_fused_add_mul