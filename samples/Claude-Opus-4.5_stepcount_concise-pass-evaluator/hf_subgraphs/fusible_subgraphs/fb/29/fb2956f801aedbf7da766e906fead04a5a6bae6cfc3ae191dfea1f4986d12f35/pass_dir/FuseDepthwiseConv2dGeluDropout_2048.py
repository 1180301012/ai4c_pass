import torch
import triton
import triton.language as tl

# Pattern: depthwise conv2d -> gelu -> dropout(p=0.0)
# This pass handles groups=2048 case

def pattern(bias, weight, input_tensor):
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (1, 1), (1, 1), 2048)
    gelu_out = torch.nn.functional.gelu(conv_out)
    dropout_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return dropout_out


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.jit
def depthwise_conv3x3_gelu_kernel_2048(
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
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    hw = H * W
    chw = C * hw
    
    n = offs // chw
    remainder = offs % chw
    c = remainder // hw
    remainder2 = remainder % hw
    h = remainder2 // W
    w = remainder2 % W
    
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    w00 = tl.load(weight_ptr + c * 9 + 0, mask=mask, other=0.0)
    w01 = tl.load(weight_ptr + c * 9 + 1, mask=mask, other=0.0)
    w02 = tl.load(weight_ptr + c * 9 + 2, mask=mask, other=0.0)
    w10 = tl.load(weight_ptr + c * 9 + 3, mask=mask, other=0.0)
    w11 = tl.load(weight_ptr + c * 9 + 4, mask=mask, other=0.0)
    w12 = tl.load(weight_ptr + c * 9 + 5, mask=mask, other=0.0)
    w20 = tl.load(weight_ptr + c * 9 + 6, mask=mask, other=0.0)
    w21 = tl.load(weight_ptr + c * 9 + 7, mask=mask, other=0.0)
    w22 = tl.load(weight_ptr + c * 9 + 8, mask=mask, other=0.0)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h + kh - 1
            iw = w + kw - 1
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            input_offset = n * input_batch_stride + c * input_channel_stride + ih * input_h_stride + iw * input_w_stride
            inp_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
            
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
            else:
                wgt = w22
            
            acc += inp_val * wgt
    
    acc = acc + bias_val
    sqrt2 = 1.4142135623730951
    result = acc * 0.5 * (1.0 + tl.math.erf(acc / sqrt2))
    
    output_offset = n * output_batch_stride + c * output_channel_stride + h * output_h_stride + w * output_w_stride
    tl.store(output_ptr + output_offset, result, mask=mask)


@torch.fx.wrap
def fused_depthwise_conv_gelu_2048(bias, weight, input_tensor):
    N, C, H, W = input_tensor.shape
    output = torch.empty_like(input_tensor)
    total_elements = N * C * H * W
    
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    depthwise_conv3x3_gelu_kernel_2048[grid](
        input_tensor, weight, bias, output,
        N, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_depthwise_conv_gelu_2048