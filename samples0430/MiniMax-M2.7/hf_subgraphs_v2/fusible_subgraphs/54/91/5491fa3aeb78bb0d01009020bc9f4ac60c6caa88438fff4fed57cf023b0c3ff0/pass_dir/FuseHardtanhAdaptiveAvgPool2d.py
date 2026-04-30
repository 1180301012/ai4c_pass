import torch
import triton
import triton.language as tl


def pattern(arg0):
    """
    Match the pattern: hardtanh -> adaptive_avg_pool2d
    Returns the pooled result which will then go through view + flatten
    """
    tmp_0 = torch.nn.functional.hardtanh(arg0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(arg0):
    return (arg0,)


@triton.jit
def fused_hardtanh_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    HW,
):
    """
    Fused kernel that performs hardtanh clipping and adaptive average pooling in one pass.
    
    Grid: (N*C, ) - each program handles one (n, c) output
    Sequential processing of HW elements
    """
    # Get position in the grid
    pid = tl.program_id(0)
    pid_n = pid // C
    pid_c = pid % C
    
    # Calculate base offset for this (n, c) in input tensor
    base_offset = pid_n * in_stride_n + pid_c * in_stride_c
    
    # Accumulator for sum
    sum_val = 0.0
    
    # Process all HW elements sequentially
    for h in range(H):
        for w in range(W):
            offset = base_offset + h * in_stride_h + w * in_stride_w
            val = tl.load(input_ptr + offset)
            # Apply hardtanh: clamp to [0, 6]
            val = tl.minimum(val, 6.0)
            val = tl.maximum(val, 0.0)
            sum_val = sum_val + val
    
    # Calculate mean and store
    # Output tensor is [N, C, 1, 1] with contiguous strides [C, 1, 1, 1]
    # So output offset = pid_n * C + pid_c
    output_offset = pid_n * C + pid_c
    mean_val = sum_val / HW
    tl.store(output_ptr + output_offset, mean_val)


def fused_hardtanh_adaptive_avg_pool2d(x):
    """
    Fused hardtanh + adaptive_avg_pool2d operation.
    Input: [N, C, H, W]
    Output: [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    HW = H * W
    
    # Allocate output tensor
    output = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Grid: (N*C, ) - 1D grid, each program handles one (n, c) output
    grid = (N * C,)
    
    fused_hardtanh_adaptive_avg_pool2d_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        in_stride_n=x.stride(0),
        in_stride_c=x.stride(1),
        in_stride_h=x.stride(2),
        in_stride_w=x.stride(3),
        HW=HW,
        num_warps=4,
    )
    
    return output


@torch.fx.wrap
def fused_hardtanh_adaptive_avg_pool2d_wrapper(x):
    return fused_hardtanh_adaptive_avg_pool2d(x)


def replacement_func():
    return fused_hardtanh_adaptive_avg_pool2d_wrapper