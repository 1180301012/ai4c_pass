import torch
import triton
import triton.language as tl


def pattern(tmp_2):
    return torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)

def replacement_args(tmp_2):
    return (tmp_2,)


@triton.jit
def interpolate_kernel(
    input_ptr,
    output_ptr,
    channels,
    output_height,
    output_width,
    input_width,
    BLOCK_WIDTH: tl.constexpr
):
    # Compute width index and mask
    width_id = tl.program_id(0) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    mask = width_id < output_width

    # Channel ID for this block
    channel_id = tl.program_id(1)

    # Calculate input coordinates
    input_w = (width_id * input_width) / output_width
    w_floor = tl.floor(input_w).to(tl.int32)
    w_ceil = tl.minimum(w_floor + 1, input_width - 1)

    # Compute interpolation weights
    weight_left = input_w - w_floor
    weight_right = 1.0 - weight_left

    # Load input values
    input_base = input_ptr + channel_id * input_width
    in1 = tl.load(input_base + w_floor, mask=(w_floor < input_width), other=0.0)
    in2 = tl.load(input_base + w_ceil, mask=(w_ceil < input_width), other=0.0)

    # Interpolate
    out_val = in1 * weight_right + in2 * weight_left

    # Store for all height dimensions
    output_base = output_ptr + channel_id * output_height * output_width
    for h in range(output_height):
        offset = h * output_width + width_id
        tl.store(output_base + offset, out_val, mask=mask)


@torch.fx.wrap
def triton_interpolate(input_tensor):
    batch, channels, _, width = input_tensor.shape
    output_height = 64
    output_width = 128

    # Allocate output tensor
    output = torch.empty((batch, channels, output_height, output_width), 
                         dtype=input_tensor.dtype, 
                         device=input_tensor.device)

    BLOCK_WIDTH = 128
    num_width_blocks = (output_width + BLOCK_WIDTH - 1) // BLOCK_WIDTH

    # Launch kernel
    interpolate_kernel[(num_width_blocks, channels)](
        input_tensor,
        output,
        channels,
        output_height,
        output_width,
        width,
        BLOCK_WIDTH=BLOCK_WIDTH
    )

    return output

def replacement_func():
    return triton_interpolate