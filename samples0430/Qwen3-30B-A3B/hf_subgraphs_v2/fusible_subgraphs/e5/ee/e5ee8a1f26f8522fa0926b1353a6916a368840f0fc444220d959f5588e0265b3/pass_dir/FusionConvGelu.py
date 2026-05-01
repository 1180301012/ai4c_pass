import torch
import triton
import triton.language as tl


def pattern(input, weight, bias):
    conv = torch.conv2d(input, weight, bias, (1, 1), (1, 1), (1, 1), 128)
    return torch.nn.functional.gelu(conv)

def replacement_args(input, weight, bias):
    return (input, weight, bias)


@triton.jit
def conv_gelu_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_height, kernel_width,
                    output_height, output_width,
                    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr):
    # Each program handles a block of output (H, W)
    block_start_y = tl.program_id(0) * BLOCK_SIZE_H
    block_start_x = tl.program_id(1) * BLOCK_SIZE_W
    out_channel = tl.program_id(2)
    batch_idx = tl.program_id(3)

    # Calculate output indices
    y = block_start_y + tl.arange(0, BLOCK_SIZE_H)
    x = block_start_x + tl.arange(0, BLOCK_SIZE_W)
    mask = (y < output_height) & (x < output_width)

    # Calculate input offsets (with padding)
    input_y = y + 1  # for padding=1
    input_x = x + 1

    # Get input data for the current channel
    input_offsets = (
        batch_idx * in_channels * in_height * in_width +
        out_channel * in_height * in_width +
        input_y * in_width +
        input_x
    )
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Load weight and bias for this channel
    weight_offset = out_channel * kernel_height * kernel_width
    weight = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, kernel_height * kernel_width), other=0.0)
    bias_val = tl.load(bias_ptr + out_channel, other=0.0)

    # Perform convolution
    conv_result = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for k in range(kernel_height * kernel_width):
        ky = k // kernel_width
        kx = k % kernel_width
        conv_result += input_vals * weight[k]
    conv_result += bias_val

    # Apply gelu activation
    gelu_result = conv_result * 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (conv_result + 0.044715 * (conv_result ** 3))))

    # Store output
    output_offsets = (
        batch_idx * out_channels * output_height * output_width +
        out_channel * output_height * output_width +
        y * output_width +
        x
    )
    tl.store(output_ptr + output_offsets, gelu_result, mask=mask)


@torch.fx.wrap
def conv_gelu(input, weight, bias):
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = weight.shape[0]
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    output_height = in_height - kernel_height + 2  # (1 + padding = 1, so -1 + 1 + 2*1?)
    output_width = in_width - kernel_width + 2

    # Calculate grid dimensions
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (grid_h, grid_w, out_channels, batch_size)

    # Create output tensor
    output = torch.empty_like(input)

    # Launch kernel
    conv_gelu_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        output_height, output_width,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return output

def replacement_func():
    return conv_gelu