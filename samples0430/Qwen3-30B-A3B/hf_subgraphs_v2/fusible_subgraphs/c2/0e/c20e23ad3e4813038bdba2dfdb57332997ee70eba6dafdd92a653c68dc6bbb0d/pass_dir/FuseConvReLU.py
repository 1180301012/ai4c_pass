import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    relu_out = torch.nn.functional.relu(conv2d, inplace = True)
    return relu_out

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,
    bias_stride0,
    output_stride0, output_stride1, output_stride2, output_stride3,
    batch_size, in_channels, out_channels, input_height, input_width,
    kernel_size, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_OUT_CH: tl.constexpr,
):
    # Compute output tile position
    block_start_h = tl.program_id(0) * BLOCK_H
    block_start_w = tl.program_id(1) * BLOCK_W
    block_out_ch = tl.program_id(2) * BLOCK_OUT_CH

    # Initialize output tile
    output = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_OUT_CH), dtype=tl.float32)

    # Iterate over input and weight
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            for ic in range(in_channels):
                # Compute input position
                ih = block_start_h + kh * dilation_h - padding_h
                iw = block_start_w + kw * dilation_w - padding_w
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    input_val = tl.load(
                        input_ptr + (
                            0 * input_stride0 + 
                            ic * input_stride1 + 
                            ih * input_stride2 + 
                            iw * input_stride3
                        )
                    )
                    # Compute weight position
                    weight_val = tl.load(
                        weight_ptr + (
                            block_out_ch * weight_stride0 + 
                            ic * weight_stride1 + 
                            kh * weight_stride2 + 
                            kw * weight_stride3
                        )
                    )
                    output += input_val * weight_val

    # Add bias and apply ReLU
    for oc in range(BLOCK_OUT_CH):
        bias_val = tl.load(
            bias_ptr + (block_out_ch + oc) * bias_stride0
        )
        output_val = output[0, 0, oc] + bias_val
        output_val = tl.maximum(output_val, 0.0)
        tl.store(
            output_ptr + (
                0 * output_stride0 + 
                (block_out_ch + oc) * output_stride1 + 
                block_start_h * output_stride2 + 
                block_start_w * output_stride3
            ),
            output_val
        )

@torch.fx.wrap
def fused_conv2d_relu(in_3, in_1, in_0):
    batch_size, in_channels, input_height, input_width = in_3.shape
    out_channels, _, kernel_size, _ = in_1.shape
    
    # Output spatial dimensions
    output_height = (input_height + 2 * padding_h - dilation_h * (kernel_size - 1) - 1) // stride_h + 1
    output_width = (input_width + 2 * padding_w - dilation_w * (kernel_size - 1) - 1) // stride_w + 1

    # Allocate output tensor
    out = torch.empty((batch_size, out_channels, output_height, output_width), dtype=in_3.dtype, device=in_3.device)

    # Grid dimensions
    grid = (
        (output_height + BLOCK_H - 1) // BLOCK_H,
        (output_width + BLOCK_W - 1) // BLOCK_W,
        (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH
    )

    # Kernel launch
    conv2d_relu_kernel[grid](
        in_3, in_1, in_0,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_size, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_H=32, BLOCK_W=32, BLOCK_OUT_CH=32,
    )
    return out

def replacement_func():
    return fused_conv2d_relu