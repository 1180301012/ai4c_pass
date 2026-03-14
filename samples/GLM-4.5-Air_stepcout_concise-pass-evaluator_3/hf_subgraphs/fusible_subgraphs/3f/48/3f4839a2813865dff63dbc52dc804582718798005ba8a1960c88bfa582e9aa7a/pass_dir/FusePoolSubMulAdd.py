import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    tmp_3 = tmp_2 - in_2
    tmp_2 = None
    tmp_4 = tmp_0.unsqueeze(-1)
    tmp_0 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    tmp_6 = tmp_5 * tmp_3
    tmp_5 = tmp_3 = None
    tmp_7 = in_2 + tmp_6
    tmp_6 = None
    tmp_8 = tmp_1.unsqueeze(-1)
    tmp_1 = None
    tmp_9 = tmp_8.unsqueeze(-1)
    tmp_8 = None
    return (tmp_7, tmp_9)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def avg_pool2d_3x1_kernel(
    x_ptr,
    out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    total_elements = batch_size * channels * height * width
    if pid >= total_elements:
        return
    
    # Convert linear index to coordinates
    offset = pid
    b = offset // (channels * height * width)
    offset = offset % (channels * height * width)
    c = offset // (height * width)
    offset = offset % (height * width)
    h = offset // width
    w = offset % width
    
    # Apply 3x1 average pooling with stride 1 and padding 1
    # Pooling window: [h-1, h, h+1] x [w]
    sum_val = 0.0
    count = 0
    
    for kh in range(max(0, h-1), min(height, h+2)):
        sum_val += tl.load(x_ptr + b * channels * height * width + c * height * width + kh * width + w)
        count += 1
    
    out_val = sum_val / count if count > 0 else 0.0
    tl.store(out_ptr + b * channels * height * width + c * height * width + h * width + w, out_val)

@triton.jit
def fuse_pool_sub_mul_add_kernel(
    in_2_ptr,    # Input tensor [batch, channels, height, width]
    scale_ptr,   # Scale tensor [channels] -> expanded to [channels, 1, 1]
    out_ptr,     # Output tensor [batch, channels, height, width]
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elements = batch_size * channels * height * width
    if pid >= total_elements:
        return
    
    # Convert linear index to coordinates
    offset = pid
    b = offset // (channels * height * width)
    offset = offset % (channels * height * width)
    c = offset // (height * width)
    offset = offset % (height * width)
    h = offset // width
    w = offset % width
    
    # Load original value with improved memory access patterns
    orig_val = tl.load(in_2_ptr + b * channels * height * width + c * height * width + h * width + w)
    
    # Load scale value (will be broadcasted from [channels] to [channels, 1, 1])
    scale_val = tl.load(scale_ptr + c)
    
    # Optimized 3x1 average pooling with better memory locality
    if height >= 3:
        # Fetch all 3 values in the neighborhood for full pooling
        h1, h2, h3 = max(0, h-1), h, min(height-1, h+1)
        offset_base = b * channels * height * width + c * height * width + w
        # Use base offset + row increments for better memory access
        val1 = tl.load(in_2_ptr + offset_base + h1 * width)
        val2 = tl.load(in_2_ptr + offset_base + h2 * width)
        val3 = tl.load(in_2_ptr + offset_base + h3 * width)
        pooled_val = (val1 + val2 + val3) / 3.0
    elif height == 2:
        # Handle 2-row case with boundary checking
        h1, h2 = max(0, h-1), min(1, h+1)
        offset_base = b * channels * height * width + c * height * width + w
        val1 = tl.load(in_2_ptr + offset_base + h1 * width)
        val2 = tl.load(in_2_ptr + offset_base + h2 * width)
        pooled_val = (val1 + val2) / 2.0
    else:
        # Single row case: pooling is just the original value
        pooled_val = orig_val
    
    # Optimized computation: result = original + (pooled - original) * scale
    # Rearranged for better arithmetic: result = pooled * scale + original * (1 - scale)
    result = tl.math.fma(pooled_val, scale_val, orig_val * (1.0 - scale_val))
    
    tl.store(out_ptr + b * channels * height * width + c * height * width + h * width + w, result)

@torch.fx.wrap
def fused_pooling_subtraction_multiplication_addition(in_2, in_0_expanded):
    # in_2: [batch, channels, height, width]
    # in_0_expanded: [channels, 1, 1] (squeezed back to [channels])
    batch_size, channels, height, width = in_2.shape
    n_elements = batch_size * channels * height * width
    
    # Choose optimal block size based on input characteristics
    if n_elements < 10000:
        BLOCK_SIZE = 128  # Smaller block for medium workloads
    elif n_elements < 100000:
        BLOCK_SIZE = 256  # Medium block size
    else:
        BLOCK_SIZE = 512  # Larger block size for big workloads
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    # Use the scale from expanded_0 (squeeze to get [channels])
    scale_tensor = in_0_expanded.squeeze()  # [channels]
    
    # Always use optimized Triton kernel for all input sizes
    fuse_pool_sub_mul_add_kernel[(num_programs,)](
        in_2_ptr=in_2,
        scale_ptr=scale_tensor,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def replacement_func():
    def optimized_forward(in_0, in_1, in_2):
        # Expand in_0 for scaling operation
        expanded_0 = in_0.unsqueeze(-1).unsqueeze(-1)  # [channels, 1, 1]
        
        # Expand in_1 for the second output
        expanded_1 = in_1.unsqueeze(-1).unsqueeze(-1)  # [channels, 1, 1]
        
        # Apply fused kernel: result = in_2 + (avg_pool2d(in_2) - in_2) * expanded_0
        result = fused_pooling_subtraction_multiplication_addition(in_2, expanded_0)
        
        return result, expanded_1
    
    return optimized_forward