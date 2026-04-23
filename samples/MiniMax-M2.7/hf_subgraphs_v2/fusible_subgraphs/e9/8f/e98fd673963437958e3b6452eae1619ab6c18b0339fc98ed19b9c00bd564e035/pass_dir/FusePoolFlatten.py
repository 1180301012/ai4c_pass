import torch
import triton
import triton.language as tl


@triton.jit
def pool_flatten_kernel(
    in_ptr, out_ptr,
    stride_in_0, stride_in_1, stride_in_2, stride_in_3,
    stride_out_0, stride_out_1,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused AdaptiveAvgPool2d(1) + Flatten kernel
    
    This kernel performs global average pooling and then flattens to [1, C]
    """
    pid = tl.program_id(0)
    
    # Each program processes a subset of channels
    channels_per_block = BLOCK_SIZE
    start_ch = pid * channels_per_block
    
    # Create block offsets for channels
    ch_offsets = tl.arange(0, BLOCK_SIZE)
    mask_ch = (start_ch + ch_offsets) < C
    
    # Compute base pointers as block types
    in_block_ptr = in_ptr + start_ch * stride_in_1
    
    # Initialize accumulators for adaptive_avg_pool2d (summing over H*W spatial positions)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over spatial positions
    for h in range(H):
        for w in range(W):
            # Compute linear offset for this spatial position
            offset = h * stride_in_2 + w * stride_in_3
            
            # Create block pointer for this spatial position
            ptr = in_block_ptr + offset
            ch_ptr = ptr + ch_offsets * stride_in_1
            
            # Load values for all channels in this spatial position
            val = tl.load(ch_ptr, mask=mask_ch, other=0.0)
            
            # Accumulate for average pooling
            acc = acc + val
    
    # Compute average: divide by H * W
    avg_pool = acc / (H * W)
    
    # Store output: [1, C] - each program stores its channels
    out_block_ptr = out_ptr + start_ch * stride_out_1
    out_ptr_with_offsets = out_block_ptr + ch_offsets * stride_out_1
    tl.store(out_ptr_with_offsets, avg_pool, mask=mask_ch)


@torch.fx.wrap
def pool_flatten_wrapper(in_tensor):
    """
    Wrapper for the fused pool + flatten kernel.
    
    Args:
        in_tensor: input tensor [1, C, H, W]
    
    Returns:
        Pooled and flattened output [1, C]
    """
    N, C, H, W = in_tensor.shape
    
    # Allocate output tensor
    out = torch.empty((N, C), dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Get strides
    stride_in = in_tensor.stride()
    stride_out = out.stride()
    
    # Determine block size based on number of channels
    BLOCK_SIZE = min(1024, triton.next_power_of_2(C))
    
    # Calculate grid size
    num_programs = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    pool_flatten_kernel[(num_programs,)](
        in_tensor, out,
        stride_in[0], stride_in[1], stride_in[2], stride_in[3],
        stride_out[0], stride_out[1],
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(x):
    """
    Match the pattern:
    adaptive_avg_pool2d(1) -> flatten(1, -1)
    """
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(x):
    return (x,)


def replacement_func():
    return pool_flatten_wrapper