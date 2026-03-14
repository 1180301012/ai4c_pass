import torch
import triton
import triton.language as tl

# Pattern for conv2d(stride=1, pad=1) + max_pool2d(kernel=3, stride=2, pad=1)
def pattern(weight, x):
    conv_out = torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)
    pool_out = torch.nn.functional.max_pool2d(conv_out, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return pool_out

def replacement_args(weight, x):
    return (weight, x)

@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W, OH, OW,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = N * C * OH * OW
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Calculate output positions
    n = offsets // (C * OH * OW)
    remaining = offsets % (C * OH * OW)
    c = remaining // (OH * OW)
    remaining = remaining % (OH * OW)
    oh = remaining // OW
    ow = remaining % OW
    
    # Calculate input starting positions (stride=2, padding=1)
    ih_start = oh * 2 - 1
    iw_start = ow * 2 - 1
    
    # Initialize max values
    max_vals = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    # Loop over 3x3 kernel
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = ih_start + kh
            iw = iw_start + kw
            
            # Check bounds
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            
            # Calculate input index
            in_idx = n * input_stride_n + c * input_stride_c + ih * input_stride_h + iw * input_stride_w
            
            # Load and update max
            vals = tl.load(input_ptr + in_idx, mask=valid, other=float('-inf'))
            max_vals = tl.maximum(max_vals, vals)
    
    # Store results
    out_idx = n * output_stride_n + c * output_stride_c + oh * output_stride_h + ow * output_stride_w
    tl.store(output_ptr + out_idx, max_vals, mask=mask)

@torch.fx.wrap
def conv_maxpool_fused_s1p1(weight, x):
    # Perform convolution using torch (cuDNN optimized)
    conv_out = torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)
    
    # Get dimensions
    N, C, H, W = conv_out.shape
    # Output size formula: (H + 2*pad - dil*(k-1) - 1) // stride + 1
    OH = (H + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1
    OW = (W + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1
    
    # Allocate output
    output = torch.empty((N, C, OH, OW), dtype=conv_out.dtype, device=conv_out.device)
    
    # Calculate strides
    input_stride_n, input_stride_c, input_stride_h, input_stride_w = conv_out.stride()
    output_stride_n, output_stride_c, output_stride_h, output_stride_w = output.stride()
    
    # Launch kernel
    num_elements = N * C * OH * OW
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    max_pool2d_kernel[(num_programs,)](
        conv_out, output,
        N, C, H, W, OH, OW,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return conv_maxpool_fused_s1p1