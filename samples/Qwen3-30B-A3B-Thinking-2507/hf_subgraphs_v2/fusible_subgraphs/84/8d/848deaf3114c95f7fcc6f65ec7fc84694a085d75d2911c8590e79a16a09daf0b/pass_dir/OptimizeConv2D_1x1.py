import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_3, in_1, in_0):
    # Conv2d with 1x1 kernel, stride 1, padding 0
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Triton kernel for 1x1 convolution (matrix multiplication)
@triton.jit
def conv2d_1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B: tl.int32,
    C_in: tl.int32,
    C_out: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Block IDs
    pid_m = tl.program_id(0)  # For output channels (C_out)
    pid_n = tl.program_id(1)  # For spatial positions (H*W)

    # Starting points
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m = m_start + tl.arange(0, BLOCK_M)
    n = n_start + tl.arange(0, BLOCK_N)

    # Masks for bounds
    m_mask = m < C_out
    n_mask = n < H * W

    # Load bias for current output channels
    bias = tl.load(bias_ptr + m, mask=m_mask, other=0.0)

    # Load weight [C_out, C_in]
    weight_ptr_batch = weight_ptr + m[:, None] * C_in
    weight = tl.load(weight_ptr_batch + tl.arange(0, BLOCK_K), mask=m_mask[:, None], other=0.0)

    # Load input [B, C_in, H, W] → [H*W, C_in] for this block
    input_ptr_batch = input_ptr + n[:, None] * C_in
    input = tl.load(input_ptr_batch + tl.arange(0, BLOCK_K), mask=n_mask[:, None], other=0.0)

    # Manual matrix multiplication loop (corrected shape)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(C_in):
        input_val = tl.load(input_ptr + (k + n * C_in), mask=n_mask, other=0.0)
        weight_val = tl.load(weight_ptr + (m * C_in + k), mask=m_mask, other=0.0)
        acc += input_val[:, None] * weight_val[None, :]

    # Add bias
    acc += bias

    # Store output
    output_ptr_batch = output_ptr + n[:, None] * C_out
    output_idx = output_ptr_batch + m
    tl.store(output_idx, acc, mask=n_mask[:, None] & m_mask)

# Wrapper function
@torch.fx.wrap
def optimized_conv2d(input_tensor, weight_tensor, bias_tensor):
    # Get shapes
    B, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]  # [C_out, C_in, 1, 1]

    # Block sizes tuned for this specific pattern
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128  # Matches C_in=256

    # Calculate grid dimensions
    num_blocks_m = (C_out + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (H * W + BLOCK_N - 1) // BLOCK_N

    # Create output tensor
    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # Launch kernel
    conv2d_1x1_kernel[(num_blocks_m, num_blocks_n)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        B, C_in, C_out, H, W,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return output

# Replacement function
def replacement_func():
    return optimized_conv2d