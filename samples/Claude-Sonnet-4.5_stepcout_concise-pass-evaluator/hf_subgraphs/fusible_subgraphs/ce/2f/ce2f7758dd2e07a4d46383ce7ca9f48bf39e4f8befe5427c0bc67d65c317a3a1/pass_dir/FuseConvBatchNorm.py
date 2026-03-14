import torch
import triton
import triton.language as tl


def pattern(pool_input):
    """
    Match avg_pool2d pattern
    """
    pool_out = torch.nn.functional.avg_pool2d(pool_input, 2, 2, 0, True, False, None)
    return pool_out


def replacement_args(pool_input):
    return (pool_input,)


@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    batch, channels, in_height, in_width,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized 2x2 avg pooling with stride 2, count_include_pad=True
    """
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE output elements
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total output elements
    total_out = batch * channels * out_height * out_width
    mask = idx < total_out
    
    # Decode flat index to (b, c, oh, ow)
    ow = idx % out_width
    temp = idx // out_width
    oh = temp % out_height
    temp = temp // out_height
    c = temp % channels
    b = temp // channels
    
    # Input positions for 2x2 window
    ih = oh * 2
    iw = ow * 2
    
    # Accumulate the 2x2 window
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Unroll the 2x2 loop
    for dh in tl.static_range(2):
        for dw in tl.static_range(2):
            ih_idx = ih + dh
            iw_idx = iw + dw
            
            # Check bounds
            valid = (ih_idx < in_height) & (iw_idx < in_width) & mask
            
            # Calculate input index
            input_idx = b * (channels * in_height * in_width) + \
                       c * (in_height * in_width) + \
                       ih_idx * in_width + iw_idx
            
            # Load with mask
            val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
            sum_val += val
    
    # Average (count_include_pad=True means always divide by 4)
    avg_val = sum_val * 0.25
    
    # Store
    tl.store(output_ptr + idx, avg_val, mask=mask)


@torch.fx.wrap
def optimized_avgpool(pool_input):
    """
    Optimized avg_pool2d using Triton
    """
    batch, channels, in_height, in_width = pool_input.shape
    out_height = (in_height + 2 * 0 - 2) // 2 + 1
    out_width = (in_width + 2 * 0 - 2) // 2 + 1
    
    output = torch.empty(batch, channels, out_height, out_width,
                        device=pool_input.device, dtype=pool_input.dtype)
    
    total_elements = batch * channels * out_height * out_width
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    avg_pool2d_kernel[grid](
        pool_input, output,
        batch, channels, in_height, in_width,
        out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_avgpool