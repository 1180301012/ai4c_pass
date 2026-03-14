import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Match the pattern: cat + batch_norm + prelu + adaptive_avg_pool2d + view(1, 128)
    
    This pattern matches the graph with batch_size=1 (graph 0).
    """
    # Store inputs
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    
    # Concatenate along channel dimension
    tmp_5 = torch.cat([in_5, in_6], 1)
    
    # Batch normalization: input, running_mean, running_var, weight, bias, training, momentum, eps
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    
    # PReLU activation
    tmp_7 = torch.prelu(tmp_6, tmp_0)
    
    # Adaptive average pooling to 1x1
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
    
    # View to 2D - hardcoded for batch_size=1
    tmp_9 = tmp_8.view(1, 128)
    
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Extract arguments for replacement kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def elementwise_kernel(
    # Input pointers
    in_5_ptr, in_6_ptr,
    # Parameter pointers
    prelu_weight_ptr, mean_ptr, var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Output pointer
    out_ptr,
    # Dimensions
    batch_size, half_channels, height, width,
    # Strides for in_5
    stride_in_5_b, stride_in_5_c, stride_in_5_hw,
    # Strides for in_6
    stride_in_6_b, stride_in_6_c, stride_in_6_hw,
    # Strides for output
    stride_out_b, stride_out_c, stride_out_hw,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for element-wise ops: cat + bn + prelu
    
    Process each element in parallel.
    """
    # Get position
    pid = tl.program_id(0)
    num_elements = batch_size * half_channels * 2 * height * width
    
    if pid >= num_elements:
        return
    
    # Calculate indices
    remaining = pid
    b = remaining // (half_channels * 2 * height * width)
    remaining = remaining % (half_channels * 2 * height * width)
    c_full = remaining // (height * width)  # 0-127 (full channel index)
    remaining = remaining % (height * width)
    hw = remaining
    
    # Determine which half (in_5 or in_6) and actual channel
    if c_full < half_channels:
        c = c_full
        # Load from in_5 [B, 64, H, W]
        in_offset = b * stride_in_5_b + c * stride_in_5_c + hw * 1
        x = tl.load(in_5_ptr + in_offset, mask=True)
    else:
        c = c_full - half_channels
        # Load from in_6 [B, 64, H, W]
        in_offset = b * stride_in_6_b + c * stride_in_6_c + hw * 1
        x = tl.load(in_6_ptr + in_offset, mask=True)
    
    # Full channel index for loading parameters
    c_idx = c_full
    
    # Batch normalization
    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    weight = tl.load(bn_weight_ptr + c_idx)
    bias = tl.load(bn_bias_ptr + c_idx)
    
    eps = 0.001
    normalized = (x - mean) / tl.sqrt(var + eps)
    x = normalized * weight + bias
    
    # PReLU
    prelu_w = tl.load(prelu_weight_ptr + c_idx)
    x = tl.where(x > 0, x, prelu_w * x)
    
    # Store output [B, 128, H, W]
    out_offset = b * stride_out_b + c_full * stride_out_c + hw * 1
    tl.store(out_ptr + out_offset, x)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Wrapper function for the fused kernel
    
    Input shapes:
    - in_0: [128] - PReLU weight
    - in_1: [128] - running mean
    - in_2: [128] - running var
    - in_3: [128] - BN bias
    - in_4: [128] - BN weight
    - in_5: [B, 64, H, W] - first tensor
    - in_6: [B, 64, H, W] - second tensor
    
    Output:
    - tmp_7: [B, 128, H, W] - activated features
    - tmp_9: [B, 128] - pooled and reshaped
    """
    # Get dimensions
    batch_size = in_5.shape[0]
    half_channels = in_5.shape[1]  # 64
    channels = half_channels * 2  # 128
    height = in_5.shape[2]
    width = in_5.shape[3]
    
    # Allocate output for activated features [B, 128, H, W]
    tmp_7 = torch.empty((batch_size, channels, height, width), dtype=torch.float32, device=in_5.device)
    
    # Get strides
    # in_5: [B, 64, H, W] - stride is (64*H*W, H*W, W, 1)
    stride_in_5_b = in_5.stride(0)
    stride_in_5_c = in_5.stride(1)
    stride_in_5_hw = in_5.stride(2)
    
    # in_6: [B, 64, H, W]
    stride_in_6_b = in_6.stride(0)
    stride_in_6_c = in_6.stride(1)
    stride_in_6_hw = in_6.stride(2)
    
    # output: [B, 128, H, W]
    stride_out_b = tmp_7.stride(0)
    stride_out_c = tmp_7.stride(1)
    stride_out_hw = tmp_7.stride(2)
    
    # Launch kernel for element-wise operations
    num_elements = batch_size * channels * height * width
    grid = (num_elements,)
    
    elementwise_kernel[(num_elements,)](
        in_5, in_6,
        in_0, in_1, in_2, in_4, in_3,  # prelu_weight, mean, var, bn_weight, bn_bias
        tmp_7,
        batch_size, half_channels, height, width,
        # Strides for in_5
        stride_in_5_b, stride_in_5_c, stride_in_5_hw,
        # Strides for in_6
        stride_in_6_b, stride_in_6_c, stride_in_6_hw,
        # Strides for output
        stride_out_b, stride_out_c, stride_out_hw,
        1,  # BLOCK_SIZE - each thread processes 1 element
    )
    
    # Adaptive average pooling using PyTorch
    tmp_8 = tmp_7.mean(dim=(2, 3))  # [B, 128]
    
    # View reshape to [1, 128] - batch_size=1
    tmp_9 = tmp_8.view(1, channels)
    
    return tmp_7, tmp_9


def replacement_func():
    return fused_kernel_wrapper