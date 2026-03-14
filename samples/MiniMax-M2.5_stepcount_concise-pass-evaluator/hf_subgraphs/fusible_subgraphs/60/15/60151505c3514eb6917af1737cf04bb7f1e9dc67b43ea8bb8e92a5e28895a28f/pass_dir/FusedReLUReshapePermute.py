import torch
import triton
import triton.language as tl


# Pattern matching function - most minimal version
def pattern(in_0, in_1):
    r = torch.relu(in_1)
    p = r.reshape(in_1.shape[0], in_1.shape[1], -1).permute(0, 2, 1)
    s = in_0.reshape(in_0.shape[0], in_0.shape[1], -1)
    return p, s


# Argument extraction function 
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel for fused ReLU + reshape + permute
@triton.jit
def fused_relu_reshape_permute_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of rows
    pid = tl.program_id(0)
    num_rows = B * N
    
    # Calculate which row this program handles
    row_idx = pid * BLOCK_SIZE
    row_offsets = row_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create masks
    mask = row_offsets < num_rows
    
    # Calculate b and n indices from linear row index
    b = row_offsets // N
    n = row_offsets % N
    
    # Load input data: input is [B, C, N, 1], access at [b, :, n, 0]
    # Flatten: offset = b * C * N + c * N + n
    # Actually, let's compute 3D indices properly
    # Input stride: [C*N, N, 1] for [B, C, N, 1]
    
    # Load all C channels for this b,n position and apply relu, then transpose
    # Output is [B, N, C], so we store at [b, n, c]
    
    for c in range(C):
        # Compute input offset: [b, c, n, 0] -> b*C*N + c*N + n
        input_offset = b * (C * N) + c * N + n
        # Load value
        val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        # Apply ReLU
        val = tl.where(val > 0, val, 0.0)
        # Compute output offset: [b, n, c] -> b*N*C + n*C + c
        output_offset = b * (N * C) + n * C + c
        # Store
        tl.store(output_ptr + output_offset, val, mask=mask)


@torch.fx.wrap
def fused_relu_reshape_permute_wrapper(in_0, in_1):
    """Fused kernel that combines: relu(in_1) -> reshape -> permute"""
    B = in_1.shape[0]
    C = in_1.shape[1]
    N = in_1.shape[2]
    
    # Output shapes
    # tmp_3: [B, N, C] (permuted from [B, C, N])
    # tmp_1: [B, C, N] (reshaped from [B, C, N, 1])
    
    # Create output tensor for permuted result [B, N, C]
    out_permuted = torch.empty((B, N, C), dtype=in_1.dtype, device=in_1.device)
    
    # Also need to reshape in_0 from [B, C, N, 1] to [B, C, N]
    out_reshaped = in_0.reshape(B, C, N)
    
    # Calculate grid
    total_rows = B * N
    BLOCK_SIZE = 1024
    num_programs = (total_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_reshape_permute_kernel[(num_programs,)](
        input_ptr=in_1,
        output_ptr=out_permuted,
        B=B,
        C=C,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_permuted, out_reshaped


def replacement_func():
    return fused_relu_reshape_permute_wrapper