import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: conv2d -> hardsigmoid -> elementwise mul
# Returns the final elementwise multiplication result
# (which becomes input to adaptive_avg_pool2d)
def pattern(in_3, in_1, in_0, in_2):
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hardsigmoid = torch.nn.functional.hardsigmoid(conv, False)
    mul = in_2 * hardsigmoid
    return mul

def replacement_args(in_3, in_1, in_0, in_2):
    return (in_3, in_1, in_0, in_2)


# Triton kernel: 1x1 conv + hardsigmoid + elementwise mul fusion
@triton.jit
def conv_hardsigmoid_mul_kernel(
    A_ptr,  # Input tensor [N, K], flattened: (batch, in_channels, h, w) -> (batch*h*w, in_channels)
    B_ptr,  # Weight tensor [M, K]
    bias_ptr,  # Bias tensor [M]
    C_ptr,  # Output tensor [N, M]
    mul_in_ptr,  # Multiplication input [N, M]
    N: tl.int32, M: tl.int32, K: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    n_id = tl.program_id(0)  # Handles one N element (batch*h*w)
    m_id = tl.program_id(1)  # Handles a block of M elements

    # Calculate start indices
    start_m = m_id * BLOCK_M
    
    # Initialize accumulator for current M block
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Process input channels in blocks
    for start_k in range(0, K, BLOCK_K):
        # Load A [BLOCK_K] for current N
        a = tl.load(
            A_ptr + (n_id * K + start_k),
            shape=(BLOCK_K,),
            mask=(start_k + tl.arange(0, BLOCK_K) < K),
            other=0.0
        )
        
        # Load B [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + (start_m * K + start_k),
            shape=(BLOCK_M, BLOCK_K),
            mask=(start_m + tl.arange(0, BLOCK_M) < M)[:, None] & (start_k + tl.arange(0, BLOCK_K) < K)[None, :],
            other=0.0
        )
        
        # Matrix multiply: (BLOCK_M, BLOCK_K) * (BLOCK_K,) -> (BLOCK_M,)
        acc += tl.sum(b * a[None, :], axis=1)

    # Add bias
    bias_val = tl.load(
        bias_ptr + start_m,
        shape=(BLOCK_M,),
        mask=(start_m + tl.arange(0, BLOCK_M) < M),
        other=0.0
    )
    acc += bias_val

    # Apply hardsigmoid: 0.5 * (x + 1)
    acc = 0.5 * (acc + 1.0)

    # Multiply by input (elementwise)
    mul_in_val = tl.load(
        mul_in_ptr + (n_id * M + start_m),
        shape=(BLOCK_M,),
        mask=(start_m + tl.arange(0, BLOCK_M) < M),
        other=0.0
    )
    acc *= mul_in_val

    # Store result
    tl.store(
        C_ptr + (n_id * M + start_m),
        acc,
        mask=(start_m + tl.arange(0, BLOCK_M) < M)
    )


# Kernel wrapper (must be wrapped for FX)
@torch.fx.wrap
def conv_hardsigmoid_mul(in_3, in_1, in_0, in_2):
    # Extract shapes
    batch, in_channels, h, w = in_3.shape
    out_channels = in_1.shape[0]  # [out_channels, in_channels, 1, 1]
    
    # Flattening: (batch, in_channels, h, w) -> (batch*h*w, in_channels)
    in_3_flat = in_3.permute(0, 2, 3, 1).reshape(batch * h * w, in_channels)
    
    # Multiplication input: (batch, out_channels, h, w) -> (batch*h*w, out_channels)
    in_2_flat = in_2.permute(0, 2, 3, 1).reshape(batch * h * w, out_channels)
    
    # Output tensor (N, M)
    out_flat = torch.empty((batch * h * w, out_channels), dtype=in_3.dtype, device=in_3.device)

    # Configuration
    BLOCK_M = 128
    BLOCK_K = 128
    num_m_blocks = (out_channels + BLOCK_M - 1) // BLOCK_M
    grid = (batch * h * w, num_m_blocks)

    # Launch kernel
    conv_hardsigmoid_mul_kernel[grid](
        in_3_flat, in_1.reshape(out_channels, in_channels), in_0,
        out_flat, in_2_flat,
        batch * h * w, out_channels, in_channels,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )

    # Reshape back to (batch, out_channels, h, w)
    out = out_flat.reshape(batch, h, w, out_channels).permute(0, 3, 1, 2)
    return out


def replacement_func():
    return conv_hardsigmoid_mul