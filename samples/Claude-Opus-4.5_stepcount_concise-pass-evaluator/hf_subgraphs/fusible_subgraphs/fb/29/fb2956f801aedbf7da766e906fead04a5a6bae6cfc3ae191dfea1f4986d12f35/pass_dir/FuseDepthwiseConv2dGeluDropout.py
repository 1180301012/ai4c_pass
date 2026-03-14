import torch
import triton
import triton.language as tl

# Pattern: depthwise conv2d -> gelu -> dropout(p=0.0)
# The dropout with p=0.0 is a no-op, so we fuse conv2d + gelu
# This pass handles groups=1024 case

def pattern(bias, weight, input_tensor):
    """
    Match depthwise conv2d + gelu + dropout pattern.
    conv2d with stride=(1,1), padding=(1,1), dilation=(1,1), groups=1024 (depthwise)
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (1, 1), (1, 1), 1024)
    gelu_out = torch.nn.functional.gelu(conv_out)
    dropout_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return dropout_out


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.jit
def depthwise_conv3x3_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C, H, W,
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Total number of output elements
    total_elements = N * C * H * W
    
    # Process BLOCK_SIZE elements per program
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Calculate n, c, h, w indices from linear offset
    # output layout is [N, C, H, W]
    hw = H * W
    chw = C * hw
    
    n = offs // chw
    remainder = offs % chw
    c = remainder // hw
    remainder2 = remainder % hw
    h = remainder2 // W
    w = remainder2 % W
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # Load weights for this channel (9 values for 3x3 kernel)
    # Weight layout is [C, 1, 3, 3], so weight for channel c starts at c*9
    w00 = tl.load(weight_ptr + c * 9 + 0, mask=mask, other=0.0)
    w01 = tl.load(weight_ptr + c * 9 + 1, mask=mask, other=0.0)
    w02 = tl.load(weight_ptr + c * 9 + 2, mask=mask, other=0.0)
    w10 = tl.load(weight_ptr + c * 9 + 3, mask=mask, other=0.0)
    w11 = tl.load(weight_ptr + c * 9 + 4, mask=mask, other=0.0)
    w12 = tl.load(weight_ptr + c * 9 + 5, mask=mask, other=0.0)
    w20 = tl.load(weight_ptr + c * 9 + 6, mask=mask, other=0.0)
    w21 = tl.load(weight_ptr + c * 9 + 7, mask=mask, other=0.0)
    w22 = tl.load(weight_ptr + c * 9 + 8, mask=mask, other=0.0)
    
    # Compute convolution with padding=1
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # For each position in 3x3 kernel
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h + kh - 1  # -1 for padding
            iw = w + kw - 1
            
            # Check if within bounds
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            
            # Calculate input offset
            input_offset = n * input_batch_stride + c * input_channel_stride + ih * input_h_stride + iw * input_w_stride
            
            # Load input value
            inp_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
            
            # Get appropriate weight
            if kh == 0 and kw == 0:
                wgt = w00
            elif kh == 0 and kw == 1:
                wgt = w01
            elif kh == 0 and kw == 2:
                wgt = w02
            elif kh == 1 and kw == 0:
                wgt = w10
            elif kh == 1 and kw == 1:
                wgt = w11
            elif kh == 1 and kw == 2:
                wgt = w12
            elif kh == 2 and kw == 0:
                wgt = w20
            elif kh == 2 and kw == 1:
                wgt = w21
            else:  # kh == 2 and kw == 2
                wgt = w22
            
            acc += inp_val * wgt
    
    # Add bias
    acc = acc + bias_val
    
    # Apply GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    result = acc * 0.5 * (1.0 + tl.math.erf(acc / sqrt2))
    
    # Store output
    output_offset = n * output_batch_stride + c * output_channel_stride + h * output_h_stride + w * output_w_stride
    tl.store(output_ptr + output_offset, result, mask=mask)


@torch.fx.wrap
def fused_depthwise_conv_gelu_1024(bias, weight, input_tensor):
    N, C, H, W = input_tensor.shape
    
    # Output tensor
    output = torch.empty_like(input_tensor)
    
    # Total elements
    total_elements = N * C * H * W
    
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    depthwise_conv3x3_gelu_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        N, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_depthwise_conv_gelu_1024