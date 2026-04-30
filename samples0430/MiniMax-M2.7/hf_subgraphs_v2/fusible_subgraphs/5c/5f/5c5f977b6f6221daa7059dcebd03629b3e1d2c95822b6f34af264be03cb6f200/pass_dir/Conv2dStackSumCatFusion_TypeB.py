import torch
import triton
import triton.language as tl

# Pattern: conv2d(in_3, ...) → stack([...], 0) → sum(0) → cat([..., in_2], 1)
# This pattern is equivalent to: conv2d → cat([..., other], 1)
# Handles the variant where conv2d takes in_3 and cat uses in_2

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: conv2d(in_3, ...) → stack → sum → cat(in_2)
    This pattern eliminates the unnecessary stack/sum by directly using conv2d output.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_2 = None
    tmp_4 = tmp_3.sum(dim=0)
    tmp_3 = None
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    tmp_4 = None
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the optimized conv2d + cat fusion.
    For this pattern: conv(in_3) + cat(in_2) -> returns (bias, weight, in_3, in_2)
    """
    return (in_0, in_1, in_3, in_2)


# Triton kernel for 1x1 conv2d with bias (stores result to first C_out channels)
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, K, H, W,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_co, weight_stride_ci, weight_stride_kh, weight_stride_kw,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate batch, output channel, height, width from flat index
    n = pid // (C * H * W)
    remaining = pid % (C * H * W)
    c = remaining // (H * W)
    remaining = remaining % (H * W)
    h = remaining // W
    w = remaining % W
    
    if n < N and c < C and h < H and w < W:
        # Accumulate convolution result
        result = tl.load(bias_ptr + c).to(tl.float32)
        
        for k in range(K):
            inp_offset = n * input_stride_n + k * input_stride_c + h * input_stride_h + w * input_stride_w
            wt_offset = c * weight_stride_co + k * weight_stride_ci
            
            inp_val = tl.load(input_ptr + inp_offset).to(tl.float32)
            wt_val = tl.load(weight_ptr + wt_offset).to(tl.float32)
            result += inp_val * wt_val
        
        out_offset = n * output_stride_n + c * output_stride_c + h * output_stride_h + w * output_stride_w
        tl.store(output_ptr + out_offset, result)


# Triton kernel to copy cat_other to second half of output (channels C_out onwards)
@triton.jit
def copy_second_half_kernel(
    input_ptr, output_ptr,
    N, C1, C2, H, W,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate position in the second half (C2 channels)
    n = pid // (C2 * H * W)
    remaining = pid % (C2 * H * W)
    c = remaining // (H * W)
    remaining = remaining % (H * W)
    h = remaining // W
    w = remaining % W
    
    if n < N and c < C2 and h < H and w < W:
        # Load from input (cat_other)
        in_offset = n * input_stride_n + c * input_stride_c + h * input_stride_h + w * input_stride_w
        val = tl.load(input_ptr + in_offset)
        
        # Store to output at position C1 + c (second half)
        out_offset = n * output_stride_n + (C1 + c) * output_stride_c + h * output_stride_h + w * output_stride_w
        tl.store(output_ptr + out_offset, val)


# Module-level function for the optimized implementation
def optimized_impl(bias, weight, conv_input, cat_other):
    """
    Optimized implementation using Triton kernels.
    Performs conv2d followed by concatenation along channel dimension.
    """
    # Get shapes
    N, C_in, H, W = conv_input.shape
    C_out = weight.shape[0]
    C_other = cat_other.shape[1]
    C_total = C_out + C_other
    
    # Allocate output
    output = torch.empty((N, C_total, H, W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Launch conv2d kernel to compute first C_out channels
    num_elements = N * C_out * H * W
    BLOCK_SIZE = 256
    num_programs = max(1, (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    conv2d_1x1_kernel[(num_programs,)](
        conv_input, weight, bias, output,
        N, C_out, C_in, H, W,
        conv_input.stride(0), conv_input.stride(1), conv_input.stride(2), conv_input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE
    )
    
    # Launch copy kernel to copy cat_other to second half of output
    num_elements = N * C_other * H * W
    num_programs = max(1, (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    copy_second_half_kernel[(num_programs,)](
        cat_other, output,
        N, C_out, C_other, H, W,
        cat_other.stride(0), cat_other.stride(1), cat_other.stride(2), cat_other.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE
    )
    
    return (output,)


def replacement_func():
    """
    Return the module-level optimized function.
    """
    return optimized_impl