import torch
import triton
import triton.language as tl

def pattern(in_0, in_2, in_1):
    # Exactly match the convolution call with positional arguments
    result = torch.conv2d(in_0, in_2, in_1, (2, 2), (0, 0), (1, 1), 1)
    return result

def replacement_args(in_0, in_2, in_1):
    # Extract tensors needed for replacement
    return (in_0, in_2, in_1)

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, input_h, input_w,
    output_h, output_w,
    BLOCK_H: tl.constexpr = 16,
    BLOCK_W: tl.constexpr = 16,
    TILE_C: tl.constexpr = 8,
):
    # Each block processes a [BLOCK_H x BLOCK_W] tile of output
    pid = tl.program_id(0)
    block_h = pid // output_w
    block_w = pid % output_w

    # Calculate start position in output for this block
    out_h_start = block_h * BLOCK_H
    out_w_start = block_w * BLOCK_W

    # Load weight data to shared memory (32x3x2x2)
    weight_shm = tl.zeros((out_channels, in_channels, 2, 2), dtype=tl.float32)
    for c_out in tl.range(0, out_channels, TILE_C):
        for c_in in tl.range(0, in_channels):
            for k_h in tl.range(0, 2):
                for k_w in tl.range(0, 2):
                    weight_idx = c_out * in_channels * 2 * 2 + c_in * 2 * 2 + k_h * 2 + k_w
                    weight_shm[c_out, c_in, k_h, k_w] = tl.load(weight_ptr + weight_idx)

    # Process each pixel in the block
    for out_h in tl.range(0, BLOCK_H):
        for out_w in tl.range(0, BLOCK_W):
            h = out_h_start + out_h
            w = out_w_start + out_w
            if h >= output_h or w >= output_w:
                continue

            # Accumulate result
            acc = tl.zeros((TILE_C,), dtype=tl.float32)
            for c_in in tl.range(0, in_channels):
                for k_h in tl.range(0, 2):
                    for k_w in tl.range(0, 2):
                        # Calculate input position
                        input_h_idx = h * 2 + k_h
                        input_w_idx = w * 2 + k_w
                        input_idx = batch_size * in_channels * input_h * input_w + \
                                   c_in * input_h * input_w + input_h_idx * input_w + input_w_idx
                        input_val = tl.load(input_ptr + input_idx)
                        
                        # Multiply with weight
                        for c_out in tl.range(0, TILE_C):
                            acc[c_out] += input_val * weight_shm[c_out, c_in, k_h, k_w]

            # Store results for this tile
            for c_out in tl.range(0, TILE_C):
                output_idx = batch_size * out_channels * output_h * output_w + \
                            (c_out) * output_h * output_w + h * output_w + w
                tl.store(output_ptr + output_idx, acc[c_out] + tl.load(bias_ptr + c_out))

@torch.fx.wrap
def conv2d_wrapper(in_0, in_2, in_1):
    # Get tensor shapes
    batch_size, in_channels, input_h, input_w = in_0.shape
    out_channels = in_2.shape[0]
    output_h = (input_h - 2) // 2 + 1  # 15
    output_w = (input_w - 2) // 2 + 1  # 15

    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, output_h, output_w), dtype=in_0.dtype, device=in_0.device)

    # Set block dimensions
    BLOCK_H = 16
    BLOCK_W = 16
    grid = (output_h * output_w + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W)

    # Launch kernel
    conv2d_kernel[grid](
        in_0, in_2, in_1, output,
        batch_size, in_channels, out_channels, input_h, input_w,
        output_h, output_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, TILE_C=8
    )
    return output

def replacement_func():
    return conv2d_wrapper