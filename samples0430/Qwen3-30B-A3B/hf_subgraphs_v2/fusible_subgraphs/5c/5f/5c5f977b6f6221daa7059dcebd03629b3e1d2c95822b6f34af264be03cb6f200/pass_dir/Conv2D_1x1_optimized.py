import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches conv2d with kernel size 1x1, padding 0, stride 1, groups 1
# This is a 1x1 convolution which can be optimized as matrix multiplication
@torch.fx.wrap
def pattern(in_2, in_1, in_0):
    # Exact matching of the target pattern in model.py
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

# Argument extraction function
# Returns the arguments needed for the replacement
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)


# Triton kernel for 1x1 convolution optimized as matrix multiplication
@triton.jit
def conv1x1_kernel(
    input_ptr,  
    weight_ptr, 
    bias_ptr,  
    output_ptr, 
    batch,    
    H,        
    W,        
    C_in,     
    C_out,    
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    # Compute starting point for this block
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_M
    offsets = block_start + tl.arange(0, BLOCK_M)
    mask = offsets < batch * H * W

    # Output pointer for the block
    output_row = output_ptr + offsets[:, None] * C_out

    # Load input row
    input_row = tl.load(input_ptr + offsets[:, None] * C_in, mask=mask[:, None], other=0.0)

    # Process output channels in blocks
    block_n_id = tl.program_id(1)
    block_start_n = block_n_id * BLOCK_N
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < C_out

    # Initialize output
    output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Process k dimension in tiles
    for k in range(0, C_in, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        mask_k = k_offset < C_in

        # Load weight block [BLOCK_K, BLOCK_N]
        weight_block = tl.load(
            weight_ptr + k_offset[:, None] * C_in + offsets_n[None, :],
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0
        )

        # Load input block [BLOCK_M, BLOCK_K]
        input_block = tl.load(
            input_ptr + offsets[:, None] + k_offset[None, :] * (H * W),
            mask=(mask[:, None] & mask_k[None, :]),
            other=0.0
        )

        # Accumulate dot product
        output += tl.dot(input_block, weight_block, allow_tf32=True)

    # Add bias
    bias_block = tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)
    output += bias_block

    # Store result
    tl.store(output_row + offsets_n, output, mask=(mask[:, None] & mask_n[None, :]))


# Wrapper function for the kernel
@torch.fx.wrap
def conv1x1_triton(input, weight, bias):
    # Get dimensions from input
    batch, C_in, H, W = input.shape
    C_out = weight.shape[0]

    # Reshape input to [batch*H*W, C_in]
    # No reshaping needed - kernel handles CHW layout directly
    # Input: [batch, C_in, H, W], Output: [batch, C_out, H, W]
    output = torch.empty((batch, C_out, H, W), dtype=input.dtype, device=input.device)

    # Configure block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 16

    # Determine grid size
    grid_m = int((batch * H * W + BLOCK_M - 1) // BLOCK_M)
    grid_n = int((C_out + BLOCK_N - 1) // BLOCK_N)

    # Launch kernel
    conv1x1_kernel[grid_m, grid_n, 1](
        input, weight, bias,
        output, batch, H, W, C_in, C_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return output

# Replacement function - returns the optimized function
# Do not call it directly, just return the reference
def replacement_func():
    return conv1x1_triton