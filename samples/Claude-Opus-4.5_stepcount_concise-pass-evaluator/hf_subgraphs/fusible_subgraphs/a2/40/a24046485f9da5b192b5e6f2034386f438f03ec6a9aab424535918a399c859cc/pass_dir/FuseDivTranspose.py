import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_div_transpose_kernel(
    input_ptr,
    output_ptr,
    B, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused division and transpose kernel.
    Input shape: [B, M, N]
    Output shape: [B, N, M]
    Each program handles a BLOCK_M x BLOCK_N tile.
    """
    pid = tl.program_id(0)
    
    # Compute tile indices
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_tiles_m * num_tiles_n
    
    pid_b = pid // tiles_per_batch
    pid_tile = pid % tiles_per_batch
    pid_m = pid_tile // num_tiles_n
    pid_n = pid_tile % num_tiles_n
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Compute masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load [BLOCK_M, BLOCK_N] tile from input[b, m, n]
    # Input strides: [M*N, N, 1]
    input_offsets = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply division
    DIV_FACTOR = 1.6817928305074292
    data = data / DIV_FACTOR
    
    # Store to output[b, n, m] (transposed)
    # Output strides: [N*M, M, 1]
    # For element at data[i, j], we store to output[b, offs_n[j], offs_m[i]]
    output_offsets = pid_b * N * M + offs_n[None, :] * M + offs_m[:, None]
    tl.store(output_ptr + output_offsets, data, mask=mask)

@torch.fx.wrap
def fused_div_transpose(in_0):
    # Get original shape and compute batch dimension
    orig_shape = in_0.shape
    B = 1
    for i in range(len(orig_shape) - 2):
        B *= orig_shape[i]
    M = orig_shape[-2]
    N = orig_shape[-1]
    
    # Ensure input is contiguous
    in_0_cont = in_0.contiguous()
    
    # Create output with transposed last two dimensions
    out_shape = list(orig_shape[:-2]) + [N, M]
    output = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Block sizes - N is always 8 in this case, so use BLOCK_N=8
    BLOCK_M = 64
    BLOCK_N = 8
    
    # Calculate grid
    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    
    grid = (B * num_tiles_m * num_tiles_n,)
    
    fused_div_transpose_kernel[grid](
        in_0_cont,
        output,
        B, M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return (output,)

def replacement_func():
    return fused_div_transpose