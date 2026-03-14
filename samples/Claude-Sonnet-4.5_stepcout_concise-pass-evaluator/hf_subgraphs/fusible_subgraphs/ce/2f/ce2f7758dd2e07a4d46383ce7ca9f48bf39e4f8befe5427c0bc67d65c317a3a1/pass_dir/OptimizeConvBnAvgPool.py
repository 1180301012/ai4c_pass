import torch
import triton
import triton.language as tl


def pattern(bn_running_mean, bn_running_var, bn_bias, bn_weight, conv_weight, pool_input, conv_input):
    """
    Match the pattern:
    - conv2d + batch_norm on conv_input
    - avg_pool2d on pool_input
    Returns both outputs
    """
    # Conv + BN path
    conv_out = torch.conv2d(conv_input, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(conv_out, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Pool path
    pool_out = torch.nn.functional.avg_pool2d(pool_input, 2, 2, 0, True, False, None)
    
    return (pool_out, bn_out)


def replacement_args(bn_running_mean, bn_running_var, bn_bias, bn_weight, conv_weight, pool_input, conv_input):
    return (bn_running_mean, bn_running_var, bn_bias, bn_weight, conv_weight, pool_input, conv_input)


@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch, in_channels, in_height, in_width,
    out_height, out_width,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized 2x2 avg pooling with stride 2
    """
    pid = tl.program_id(0)
    total_elements = batch * in_channels * out_height * out_width
    
    if pid < total_elements:
        # Decode flat index to (b, c, oh, ow)
        ow = pid % out_width
        temp = pid // out_width
        oh = temp % out_height
        temp = temp // out_height
        c = temp % in_channels
        b = temp // in_channels
        
        # Calculate input positions (2x2 window)
        ih = oh * 2
        iw = ow * 2
        
        # Load 2x2 window and compute average
        sum_val = 0.0
        
        for dh in range(2):
            for dw in range(2):
                ih_idx = ih + dh
                iw_idx = iw + dw
                
                if ih_idx < in_height and iw_idx < in_width:
                    input_idx = b * stride_b + c * stride_c + ih_idx * stride_h + iw_idx * stride_w
                    val = tl.load(input_ptr + input_idx)
                    sum_val += val
        
        # Average (count_include_pad=True means always divide by 4)
        avg_val = sum_val * 0.25
        
        output_idx = pid
        tl.store(output_ptr + output_idx, avg_val)


@triton.jit
def conv2d_3x3_bn_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_height, in_width, out_height, out_width,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused Conv2D (3x3, stride=1, padding=1) + BatchNorm (folded into weights)
    """
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Calculate output position
    oh = (pid_hw // ((out_width + BLOCK_W - 1) // BLOCK_W)) * BLOCK_H
    ow = (pid_hw % ((out_width + BLOCK_W - 1) // BLOCK_W)) * BLOCK_W
    
    oh_offsets = oh + tl.arange(0, BLOCK_H)
    ow_offsets = ow + tl.arange(0, BLOCK_W)
    
    oh_mask = oh_offsets < out_height
    ow_mask = ow_offsets < out_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution: accumulate over input channels and kernel
    for ic in range(in_channels):
        for kh in range(3):
            for kw in range(3):
                # Input indices with padding
                ih_offsets = oh_offsets + kh - 1
                iw_offsets = ow_offsets + kw - 1
                
                ih_mask = (ih_offsets >= 0) & (ih_offsets < in_height)
                iw_mask = (iw_offsets >= 0) & (iw_offsets < in_width)
                
                # Load input
                input_idx = (pid_b * stride_in_b + ic * stride_in_c + 
                           ih_offsets[:, None] * stride_in_h + iw_offsets[None, :] * stride_in_w)
                mask = oh_mask[:, None] & ow_mask[None, :] & ih_mask[:, None] & iw_mask[None, :]
                input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                
                # Load weight
                weight_idx = pid_oc * (in_channels * 9) + ic * 9 + kh * 3 + kw
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + pid_oc)
    acc += bias_val
    
    # Store output
    output_idx = (pid_b * stride_out_b + pid_oc * stride_out_c + 
                 oh_offsets[:, None] * stride_out_h + ow_offsets[None, :] * stride_out_w)
    mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(output_ptr + output_idx, acc, mask=mask)


@torch.fx.wrap
def optimized_conv_bn_avgpool(bn_running_mean, bn_running_var, bn_bias, bn_weight, conv_weight, pool_input, conv_input):
    """
    Optimized implementation:
    1. Fold BatchNorm into Conv2D weights (using allowed operations)
    2. Use Triton kernel for fused Conv+BN
    3. Use Triton kernel for AvgPool2D
    """
    eps = 1e-05
    
    # Fold BatchNorm into Conv2D weights using power instead of sqrt
    scale = bn_weight / ((bn_running_var + eps) ** 0.5)
    new_weight = conv_weight * scale.view(-1, 1, 1, 1)
    new_bias = bn_bias - scale * bn_running_mean
    
    # Get dimensions
    batch, in_channels, in_height, in_width = conv_input.shape
    out_channels = conv_weight.shape[0]
    out_height = in_height  # stride=1, padding=1, kernel=3
    out_width = in_width
    
    # Allocate output for conv+bn
    bn_out = torch.empty(batch, out_channels, out_height, out_width,
                        device=conv_input.device, dtype=conv_input.dtype)
    
    # Launch Conv+BN kernel
    BLOCK_H = 8
    BLOCK_W = 8
    grid_h = (out_height + BLOCK_H - 1) // BLOCK_H
    grid_w = (out_width + BLOCK_W - 1) // BLOCK_W
    grid = (batch, out_channels, grid_h * grid_w)
    
    stride_in_b = in_channels * in_height * in_width
    stride_in_c = in_height * in_width
    stride_in_h = in_width
    stride_in_w = 1
    stride_out_b = out_channels * out_height * out_width
    stride_out_c = out_height * out_width
    stride_out_h = out_width
    stride_out_w = 1
    
    conv2d_3x3_bn_kernel[grid](
        conv_input, new_weight, new_bias, bn_out,
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        stride_in_b, stride_in_c, stride_in_h, stride_in_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    
    # Perform optimized AvgPool2D
    pool_batch, pool_channels, pool_height, pool_width = pool_input.shape
    pool_out_height = (pool_height + 2 * 0 - 2) // 2 + 1
    pool_out_width = (pool_width + 2 * 0 - 2) // 2 + 1
    
    pool_out = torch.empty(pool_batch, pool_channels, pool_out_height, pool_out_width,
                          device=pool_input.device, dtype=pool_input.dtype)
    
    total_elements = pool_batch * pool_channels * pool_out_height * pool_out_width
    grid_pool = lambda meta: (total_elements,)
    
    stride_pool_b = pool_channels * pool_height * pool_width
    stride_pool_c = pool_height * pool_width
    stride_pool_h = pool_width
    stride_pool_w = 1
    
    avg_pool2d_kernel[grid_pool](
        pool_input, pool_out,
        pool_batch, pool_channels, pool_height, pool_width,
        pool_out_height, pool_out_width,
        stride_pool_b, stride_pool_c, stride_pool_h, stride_pool_w,
        BLOCK_SIZE=1024,
    )
    
    return (pool_out, bn_out)


def replacement_func():
    return optimized_conv_bn_avgpool