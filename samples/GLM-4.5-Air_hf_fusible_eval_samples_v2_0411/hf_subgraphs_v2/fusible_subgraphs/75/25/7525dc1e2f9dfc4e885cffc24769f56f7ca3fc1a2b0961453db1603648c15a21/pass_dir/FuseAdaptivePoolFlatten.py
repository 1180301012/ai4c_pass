import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_4):
    """
    Match AdaptiveAvgPool2d + Flatten pattern
    """
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    return tmp_6

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

# Triton kernel for fused AdaptiveAvgPool2d + Flatten
@triton.jit
def fused_adaptive_pool_flatten_kernel(
    # Input pointer
    input_ptr,  # [N, C, H, W]
    # Output pointer  
    output_ptr,  # [N, C]
    # Tensor shapes
    N, C, H, W,
    # Strides
    input_stride_N, input_stride_C, input_stride_H, input_stride_W,
    output_stride_N, output_stride_C,
    # Parameters
    BLOCK_SIZE_N: tl.constexpr,
):
    # Determine program position
    pid = tl.program_id(0)
    
    # Guard against out-of-bound accesses  
    if pid >= N * C:
        return
    
    # Calculate which sample and channel this program handles
    sample_idx = pid // C
    channel_idx = pid % C
    
    # Initialize sum for average pooling
    sum_val = 0.0
    
    # Loop over spatial dimensions to compute average
    for h_idx in range(H):
        for w_idx in range(W):
            # Load input element
            input_ptr_base = input_ptr + sample_idx * input_stride_N + channel_idx * input_stride_C + h_idx * input_stride_H + w_idx * input_stride_W
            val = tl.load(input_ptr_base)
            sum_val += val
    
    # Compute average: sum / (H * W)
    avg_val = sum_val / (H * W)
    
    # Store result directly to flattened output
    output_ptr_base = output_ptr + sample_idx * output_stride_N + channel_idx * output_stride_C
    tl.store(output_ptr_base, avg_val)

# Kernel wrapper
@torch.fx.wrap
def fused_adaptive_pool_flatten(input_tensor):
    # Get tensor shapes
    N, C, H, W = input_tensor.shape
    
    # Get tensor strides
    input_stride_N, input_stride_C, input_stride_H, input_stride_W = input_tensor.stride()
    
    # Create output tensor with shape [N, C] (flattened)
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    output_stride_N, output_stride_C = output.stride()
    
    # Calculate grid size
    total_elements = N * C
    BLOCK_SIZE_N = 32  # Number of elements to process per program
    
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE_N),)
    
    # Ensure we launch enough programs
    if grid_size[0] == 0:
        grid_size = (1,)
    
    # Launch kernel
    fused_adaptive_pool_flatten_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        input_stride_N=input_stride_N, input_stride_C=input_stride_C, input_stride_H=input_stride_H, input_stride_W=input_stride_W,
        output_stride_N=output_stride_N, output_stride_C=output_stride_C,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_adaptive_pool_flatten