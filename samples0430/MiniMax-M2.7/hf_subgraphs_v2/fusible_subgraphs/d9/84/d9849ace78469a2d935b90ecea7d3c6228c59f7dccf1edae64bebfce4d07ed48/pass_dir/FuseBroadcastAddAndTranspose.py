import torch
import triton
import triton.language as tl


@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    src_dim: tl.constexpr,
    dst_dim: tl.constexpr,
    input_stride_0: tl.constexpr,
    input_stride_1: tl.constexpr,
    input_stride_2: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    output_stride_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized transpose kernel for [batch, src, dst] -> [batch, dst, src]
    Uses block-based tiling for better memory coalescing.
    """
    # Calculate global position
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Each block handles BLOCK_SIZE elements
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < src_dim * dst_dim
    
    # Compute input indices
    src_idx = offsets % src_dim
    dst_idx = offsets // src_dim
    
    # Compute input offset for [batch, src, dst]
    input_offset = batch_idx * input_stride_0 + src_idx * input_stride_1 + dst_idx * input_stride_2
    
    # Load from input
    vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Compute output offset for [batch, dst, src]
    # After transpose: index [b, d, s] in output comes from [b, s, d] in input
    # output_idx = batch * output_stride_0 + dst * output_stride_1 + src * output_stride_2
    # where dst = original dst_idx, src = original src_idx
    output_offset = batch_idx * output_stride_0 + dst_idx * output_stride_1 + src_idx * output_stride_2
    
    # Store to output
    tl.store(output_ptr + output_offset, vals, mask=mask)


def pattern(in_1):
    """
    Match only the transpose operation.
    The add will remain in the graph as-is.
    """
    result = in_1.transpose(1, 2)
    return result


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def optimized_transpose(in_1):
    """
    Optimized transpose implementation using Triton.
    Input shape: [batch, src, dst] -> Output shape: [batch, dst, src]
    Note: Only transposes in_1, the add with in_0 happens separately in the graph
    """
    # Input: in_1 with shape [batch, src, dst]
    batch_size, src_dim, dst_dim = in_1.shape
    
    # Allocate output with transposed shape [batch, dst, src]
    output = torch.empty(batch_size, dst_dim, src_dim, dtype=in_1.dtype, device=in_1.device)
    
    # Calculate strides
    input_stride_0 = src_dim * dst_dim  # stride for batch dimension
    input_stride_1 = dst_dim  # stride for src dimension
    input_stride_2 = 1  # stride for dst dimension
    
    output_stride_0 = dst_dim * src_dim  # stride for batch dimension in output
    output_stride_1 = src_dim  # stride for dst dimension in output
    output_stride_2 = 1  # stride for src dimension in output
    
    # Grid configuration
    BLOCK_SIZE = 128
    num_blocks = (batch_size * src_dim * dst_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (batch_size, num_blocks)
    
    optimized_transpose_kernel[grid](
        input_ptr=in_1,
        output_ptr=output,
        batch_size=batch_size,
        src_dim=src_dim,
        dst_dim=dst_dim,
        input_stride_0=input_stride_0,
        input_stride_1=input_stride_1,
        input_stride_2=input_stride_2,
        output_stride_0=output_stride_0,
        output_stride_1=output_stride_1,
        output_stride_2=output_stride_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_transpose