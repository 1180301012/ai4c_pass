import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input, normalized_shape, weight, bias, eps):
    """
    Matches: torch.nn.functional.layer_norm(input, (1024,), weight, bias, 1e-05)
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)

# Argument extraction function
def replacement_args(input, normalized_shape, weight, bias, eps):
    return (input, weight, bias, eps)

# Triton kernel for layer norm
@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    n_seq,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Block ID for channel grouping
    block_id = tl.program_id(0)
    # Compute start channel for this block
    start_c = block_id * BLOCK_SIZE

    # Process BLOCK_SIZE channels per block
    end_c = tl.minimum(start_c + BLOCK_SIZE, n_channels)
    for c in range(start_c, end_c):

        # Initialize reduction values
        sum_val = 0.0
        sum_sq = 0.0

        # First pass: compute mean and variance
        for s in range(n_seq):
            # Compute index: [batch=0][seq=s][channel=c]
            input_index = s * n_channels + c
            x = tl.load(input_ptr + input_index)
            sum_val += x
            sum_sq += x * x

        # Compute mean and variance
        mean = sum_val / tl.cast(n_seq, tl.float32)
        variance = (sum_sq / tl.cast(n_seq, tl.float32)) - (mean * mean)
        inv_std = 1.0 / tl.sqrt(variance + eps)

        # Second pass: compute output
        for s in range(n_seq):
            input_index = s * n_channels + c
            x = tl.load(input_ptr + input_index)
            norm_val = (x - mean) * inv_std
            weight_val = tl.load(weight_ptr + c)
            bias_val = tl.load(bias_ptr + c)
            output_val = norm_val * weight_val + bias_val
            output_index = s * n_channels + c
            tl.store(output_ptr + output_index, output_val)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_wrapper(input, weight, bias, eps):
    # Extract dimensions
    batch, seq, channels = input.shape
    n_channels = channels
    n_seq = seq

    # Create output tensor
    output = torch.empty_like(input)

    # Configure block size and grid
    BLOCK_SIZE = 256
    num_blocks = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    layer_norm_kernel[
        (num_blocks,)
    ](
        input,
        weight,
        bias,
        output,
        n_channels,
        n_seq,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return layer_norm_wrapper