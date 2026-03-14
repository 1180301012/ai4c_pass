import torch
import triton
import triton.language as tl


def pattern(in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    """
    Match the computation pattern for starnet_s2 (groups=256).
    """
    tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (3, 3), (1, 1), 256)
    tmp_7 = in_7 + tmp_6
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    tmp_10 = tmp_9.flatten(1, -1)
    return tmp_10


def replacement_args(in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2)


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4)],
    key=['total_elements'],
)
@triton.jit
def fused_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr,
    running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    output_ptr,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c,
    batch_size, num_channels, height, width,
    kernel_size, padding,
    total_elements: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < total_elements
    
    linear_idx = offset + tl.arange(0, BLOCK_SIZE)
    batch_idx = linear_idx // num_channels
    channel_idx = linear_idx % num_channels
    
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=mask, other=1.0)
    bn_weight = tl.load(weight_bn_ptr + channel_idx, mask=mask, other=1.0)
    bn_bias = tl.load(bias_bn_ptr + channel_idx, mask=mask, other=0.0)
    
    inv_std = bn_weight / tl.sqrt(var + eps)
    bn_scale = bn_bias - mean * inv_std
    conv_bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    spatial_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    spatial_count = height * width
    
    for h in range(height):
        for w in range(width):
            pixel_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
            
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_h = h + kh - padding
                    in_w = w + kw - padding
                    
                    in_h_valid = (in_h >= 0) and (in_h < height)
                    in_w_valid = (in_w >= 0) and (in_w < width)
                    valid = in_h_valid and in_w_valid
                    
                    input_offset = (batch_idx * input_stride_b + 
                                  channel_idx * input_stride_c + 
                                  in_h * input_stride_h + 
                                  in_w * input_stride_w)
                    
                    inp_val = tl.load(input_ptr + input_offset, mask=mask and valid, other=0.0).to(tl.float32)
                    
                    weight_offset = channel_idx * (kernel_size * kernel_size) + kh * kernel_size + kw
                    w_val = tl.load(weight_ptr + weight_offset, mask=mask and valid, other=0.0).to(tl.float32)
                    
                    pixel_sum = pixel_sum + tl.where(valid, inp_val * w_val, 0.0)
            
            pixel_sum = pixel_sum + conv_bias
            
            residual_offset = (batch_idx * input_stride_b + 
                             channel_idx * input_stride_c + 
                             h * input_stride_h + 
                             w * input_stride_w)
            residual_val = tl.load(residual_ptr + residual_offset, mask=mask, other=0.0).to(tl.float32)
            pixel_sum = pixel_sum + residual_val
            
            normalized = pixel_sum * inv_std + bn_scale
            spatial_sum = spatial_sum + normalized
    
    avg_value = spatial_sum / tl.cast(spatial_count, tl.float32)
    
    output_offset = batch_idx * num_channels + channel_idx
    tl.store(output_ptr + output_offset, avg_value, mask=mask)


@torch.fx.wrap
def fused_wrapper(in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    batch_size, num_channels, height, width = in_6.shape
    kernel_size = 7
    padding = 3
    
    output = torch.empty((batch_size, num_channels), device=in_6.device, dtype=torch.float32)
    
    in_6 = in_6.contiguous()
    in_7 = in_7.contiguous()
    tmp_5 = tmp_5.contiguous()
    tmp_4 = tmp_4.contiguous()
    tmp_0 = tmp_0.contiguous()
    tmp_1 = tmp_1.contiguous()
    tmp_3 = tmp_3.contiguous()
    tmp_2 = tmp_2.contiguous()
    
    total_elements = batch_size * num_channels
    grid = ((total_elements + 255) // 256,)
    
    fused_kernel[grid](
        in_6, tmp_5, tmp_4, in_7,
        tmp_0, tmp_1, tmp_3, tmp_2,
        output,
        in_6.stride(0), in_6.stride(1), in_6.stride(2), in_6.stride(3),
        tmp_5.stride(0),
        batch_size, num_channels, height, width,
        kernel_size, padding,
        total_elements,
        1e-05,
    )
    
    return output


def replacement_func():
    return fused_wrapper