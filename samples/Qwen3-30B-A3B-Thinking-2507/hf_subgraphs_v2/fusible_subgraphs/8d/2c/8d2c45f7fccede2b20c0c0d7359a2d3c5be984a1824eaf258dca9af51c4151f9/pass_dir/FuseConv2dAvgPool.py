import torch
import triton
import triton.language as tl

# Pattern matching function
# Note: We need to match exactly the operations in model.py
# (with same number of arguments, positions)
def pattern(in_0, in_1):
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

# Argument extraction function
# Extract inputs needed for the replacement
# This needs to match exactly the pattern
# We're returning (weight, input, output_shape) but simplified for Triton
# Note: We'll handle batch size and spatial dims in kernel

def replacement_args(in_0, in_1):
    # Get shapes
    # We're using the shapes from weight_meta.py for our kernel configuration
    weight_shape = in_0.shape
    input_shape = in_1.shape
    # (C_out, C_in, H_k, W_k)
    C_out, C_in, _, _ = weight_shape
    B, _, H_in, W_in = input_shape
    H_out = H_in // 2
    W_out = W_in // 2
    return (in_0, in_1, B, H_in, W_in, C_in, C_out, H_out, W_out)

# Triton kernel for fused convolution + avg pool
@triton.jit
def fused_conv_pool_kernel(input_ptr, weight_ptr, output_ptr,
                           B, H_in, W_in, C_in, C_out,
                           H_out, W_out,
                           BLOCK_SIZE: tl.constexpr):
    # Block index for spatial position
    block_idx = tl.program_id(0)
    h_prime = block_idx // W_out
    w_prime = block_idx % W_out

    # Compute input positions for the 2x2 block
    h0 = 2 * h_prime
    h1 = h0 + 1
    w0 = 2 * w_prime
    w1 = w0 + 1

    # Process all output channels (i) in this block
    i = tl.thread_id(0)  # Each thread handles one output channel
    if i >= C_out:
        return

    # Accumulate sums for the 4 input positions
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0

    # Loop over input channels (c) with block processing
    c_start = tl.program_id(1) * BLOCK_SIZE
    c = c_start + tl.arange(0, BLOCK_SIZE)
    mask = c < C_in

    # Load input data for the 4 positions for all input channels
    input0 = tl.load(input_ptr + c * H_in * W_in + h0 * W_in + w0, mask=mask)
    input1 = tl.load(input_ptr + c * H_in * W_in + h0 * W_in + w1, mask=mask)
    input2 = tl.load(input_ptr + c * H_in * W_in + h1 * W_in + w0, mask=mask)
    input3 = tl.load(input_ptr + c * H_in * W_in + h1 * W_in + w1, mask=mask)

    # Load weight values for output channel i
    weights = tl.load(weight_ptr + i * C_in + c, mask=mask)

    # Compute contributions
    contrib0 = input0 * weights
    contrib1 = input1 * weights
    contrib2 = input2 * weights
    contrib3 = input3 * weights

    # Accumulate across channels
    sum0 += tl.sum(contrib0)
    sum1 += tl.sum(contrib1)
    sum2 += tl.sum(contrib2)
    sum3 += tl.sum(contrib3)

    # Compute final result
    total = (sum0 + sum1 + sum2 + sum3) / 4.0

    # Store result
    output_idx = i * H_out * W_out + h_prime * W_out + w_prime
    tl.store(output_ptr + output_idx, total)

# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_pool(in_0, in_1, B, H_in, W_in, C_in, C_out, H_out, W_out):
    # Grid: each block processes one spatial position (h', w')
    grid = (H_out * W_out,)
    # Block size for input channels (BLOCK_SIZE = 64)
    BLOCK_SIZE = 64
    num_blocks = (C_in + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((B, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)

    # Launch kernel
    fused_conv_pool_kernel[grid, (num_blocks,)](
        in_1, in_0, output,
        B, H_in, W_in, C_in, C_out,
        H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Replacement function - returns the kernel wrapper function
# Must not call the function, just return the reference

def replacement_func():
    return fused_conv_pool