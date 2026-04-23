import torch
import triton
import triton.language as tl

# Pattern matching function - matches reshape + avg_pool2d
def pattern(in_4):
    """
    Match the pattern: reshape -> avg_pool2d
    """
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    return tmp_5

# Argument extraction function
def replacement_args(in_4):
    """
    Extract arguments needed for the fused kernel:
    - input tensor (in_4)
    """
    return (in_4,)

@triton.jit
def fused_reshape_avgpool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    # Output shape info
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    # Pooling params
    stride: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Reshape(input) + AvgPool2d
    
    Input: (4, 128, 256) - treated as (4, 128, 16, 16) then merged to (1, 512, 16, 16)
    Output: (1, 512, 8, 8) after avg_pool2d
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output spatial coordinates
    out_c = offsets // (output_height * output_width)
    out_remainder = offsets % (output_height * output_width)
    out_h = out_remainder // output_width
    out_w = out_remainder % output_width
    
    # For avg pool with kernel=2, stride=2:
    in_h_base = out_h * stride
    in_w_base = out_w * stride
    
    # Total elements to accumulate
    total = tl.zeros([BLOCK_SIZE], tl.float32)
    
    # Channel offset in memory (stride of 256 for the merged tensor)
    channel_offset = out_c * 256
    
    # Accumulate values for pooling window (2x2)
    for dh in range(2):
        for dw in range(2):
            in_h = in_h_base + dh
            in_w = in_w_base + dw
            
            # Calculate input linear index in the merged (1, 512, 16, 16) view
            # Linear offset: channel_offset + h * 16 + w
            input_idx = channel_offset + in_h * 16 + in_w
            
            # Load value - each thread loads its own value
            val = tl.load(in_ptr + input_idx, mask=mask, other=0.0)
            total = total + val
    
    # Average the pooled values
    out = total / 4.0
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_4):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Input:
    - in_4: input (4, 128, 256)
    
    Output: (1, 512, 8, 8)
    """
    # Output after avg_pool: (1, 512, 8, 8)
    total_channels = 512
    output_height = 8
    output_width = 8
    output_elements = total_channels * output_height * output_width  # 32768
    
    stride = 2
    
    BLOCK_SIZE = 1024
    num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out = torch.empty(1, total_channels, output_height, output_width, dtype=in_4.dtype, device=in_4.device)
    
    # Launch fused kernel
    fused_reshape_avgpool_kernel[(num_programs,)](
        in_ptr=in_4,
        out_ptr=out,
        n_elements=output_elements,
        output_height=output_height,
        output_width=output_width,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_kernel_wrapper