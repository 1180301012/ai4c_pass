import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input, weight, bias):
    # Match exact convolution operation with positional arguments
    result = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return result

# Argument extraction function
def replacement_args(input, weight, bias):
    return (input, weight, bias)

# Triton kernel for 1x1 convolution
@triton.jit
def conv1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B: tl.int32,
    C_in: tl.int32,
    H: tl.int32,
    W: tl.int32,
    input_stride0: tl.int32,
    input_stride1: tl.int32,
    input_stride2: tl.int32,
    input_stride3: tl.int32,
    weight_stride0: tl.int32,
    weight_stride1: tl.int32,
    weight_stride2: tl.int32,
    weight_stride3: tl.int32,
    output_stride0: tl.int32,
    output_stride1: tl.int32,
    output_stride2: tl.int32,
    output_stride3: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global index for this thread
    idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id(0)
    if idx >= B * H * W:
        return

    # Compute spatial coordinates
    b = idx // (H * W)
    h = (idx % (H * W)) // W
    w = idx % W

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Load bias
    bias_val = tl.load(bias_ptr)

    # Compute dot product across input channels
    for c in range(C_in):
        input_val = tl.load(
            input_ptr + b * input_stride0 + c * input_stride1 + h * input_stride2 + w * input_stride3
        )
        weight_val = tl.load(
            weight_ptr + c * weight_stride1
        )
        acc = acc + input_val * weight_val

    acc = acc + bias_val

    # Store output
    tl.store(
        output_ptr + b * output_stride0 + 0 * output_stride1 + h * output_stride2 + w * output_stride3,
        acc
    )

# Kernel wrapper
@torch.fx.wrap
def custom_conv2d(input, weight, bias):
    B, C_in, H, W = input.shape
    output = torch.empty((B, 1, H, W), dtype=input.dtype, device=input.device)

    # Extract tensor strides
    input_stride0, input_stride1, input_stride2, input_stride3 = input.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = output.stride()

    # Configure kernel grid
    num_elements = B * H * W
    BLOCK_SIZE = 128
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv1x1_kernel[(num_blocks,)](
        input, weight, bias, output,
        B, C_in, H, W,
        input_stride0, input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3,
        BLOCK_SIZE
    )

    return output

# Replacement function
@torch.fx.wrap
def replacement_func():
    return custom_conv2d