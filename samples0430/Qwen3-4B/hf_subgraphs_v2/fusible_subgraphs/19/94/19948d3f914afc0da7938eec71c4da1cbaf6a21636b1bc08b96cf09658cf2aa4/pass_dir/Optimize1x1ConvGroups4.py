import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    return torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

def replacement_func():
    return kernel_wrapper

@triton.jit
def optimized_conv_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    N_OUTPUT_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    groups: tl.constexpr,
):
    # Compute output indices for current block
    output_idx = tl.arange(0, BLOCK_SIZE)
    # Calculate start position in output
    out_offset = tl.program_id(0) * BLOCK_SIZE + output_idx
    # Load input values (simplified for 1x1)
    input_vals = tl.load(in_3_ptr + out_offset, mask=tl.arange(0, BLOCK_SIZE) < N_OUTPUT_CHANNELS)
    # Load weight values (single value per channel for 1x1)
    weight_vals = tl.load(in_1_ptr + out_offset, mask=tl.arange(0, BLOCK_SIZE) < N_OUTPUT_CHANNELS)
    # Load bias
    bias = tl.load(in_0_ptr + out_offset, mask=tl.arange(0, BLOCK_SIZE) < N_OUTPUT_CHANNELS)
    # Apply convolution operation
    output_vals = input_vals * weight_vals + bias
    # Store results
    tl.store(out_ptr + out_offset, output_vals, mask=tl.arange(0, BLOCK_SIZE) < N_OUTPUT_CHANNELS)

def kernel_wrapper(in_3, in_1, in_0):
    N_OUTPUT_CHANNELS = in_1.shape[0]
    num_blocks = (N_OUTPUT_CHANNELS + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_3)
    optimized_conv_kernel[(num_blocks,)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        N_OUTPUT_CHANNELS=N_OUTPUT_CHANNELS,
        BLOCK_SIZE=256,
        groups=4,
    )
    return out