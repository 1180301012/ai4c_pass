import torch
import triton
import triton.language as tl

def pattern(expanded_tensor):
    # Pattern: expand((1, -1, H, W)) -> contiguous()
    tmp_5 = expanded_tensor.contiguous()
    return tmp_5

def replacement_args(expanded_tensor):
    return (expanded_tensor,)

@triton.jit
def direct_copy_kernel(
    input_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    batch_size,
    embed_dim,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (batch_size * embed_dim * height * width)
    
    # Convert linear offset to 4D coordinates
    offset_3 = offset % width
    offset_2 = (offset // width) % height
    offset_1 = (offset // (width * height)) % embed_dim
    offset_0 = offset // (width * height * embed_dim)
    
    # Calculate input addresses (expanded tensor has stride 0 for first dim)
    input_offsets = offset_0 * input_stride_0 + offset_1 * input_stride_1 + offset_2 * input_stride_2 + offset_3 * input_stride_3
    
    # Load and store directly
    values = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + offset, values, mask=mask)

@torch.fx.wrap
def optimized_contiguous(expanded_tensor):
    # Get expanded tensor shape: (1, embed_dim, H, W)
    batch_size, embed_dim, height, width = expanded_tensor.shape
    
    # Create output with proper stride for contiguous layout
    stride_0 = embed_dim * height * width
    stride_1 = height * width
    stride_2 = width
    stride_3 = 1
    
    output = torch.empty((batch_size, embed_dim, height, width), 
                        dtype=expanded_tensor.dtype, 
                        device=expanded_tensor.device)
    
    # Copy strides
    output.stride_(0, stride_0)
    output.stride_(1, stride_1) 
    output.stride_(2, stride_2)
    output.stride_(3, stride_3)
    
    # For small tensors, just use PyTorch's built-in contiguous
    total_elements = batch_size * embed_dim * height * width
    if total_elements <= 8192:
        return expanded_tensor.contiguous()
    
    # Otherwise use optimized kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Note: This is a simplified implementation. In practice, the expand operation
    # often creates a view that already has efficient memory layout.
    # This pass mainly handles cases where contiguous() is actually needed.
    direct_copy_kernel[(num_programs,)](
        input_ptr=expanded_tensor,
        output_ptr=output,
        input_stride_0=0,  # First dim has stride 0 due to expand
        input_stride_1=expanded_tensor.stride(1),
        input_stride_2=expanded_tensor.stride(2), 
        input_stride_3=expanded_tensor.stride(3),
        batch_size=batch_size,
        embed_dim=embed_dim,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_contiguous