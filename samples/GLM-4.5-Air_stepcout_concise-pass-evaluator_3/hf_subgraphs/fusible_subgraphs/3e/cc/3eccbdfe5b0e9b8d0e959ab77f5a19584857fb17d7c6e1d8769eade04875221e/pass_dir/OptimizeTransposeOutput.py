import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern matching: input tensor and its transpose"""
    tmp_1 = input_tensor
    tmp_2 = tmp_1.T
    return tmp_1, tmp_2

def replacement_args(input_tensor):
    """Extract arguments for replacement"""
    return (input_tensor,)

@triton.jit
def optimized_transpose_output_kernel(
    input_ptr,
    output_ptr,
    transpose_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that efficiently computes both original and transposed outputs"""
    
    # Each program handles a tile of the output
    i, j = tl.program_id(0), tl.program_id(1)
    
    # Calculate indices for input and output tiles
    i_in = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    j_in = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for bounds checking
    mask_i = i_in < M
    mask_j = j_in < N
    
    # Load input tile
    input_ptrs = input_ptr + i_in[:, None] * N + j_in[None, :]
    input_tile = tl.load(input_ptrs, mask=(mask_i[:, None] & mask_j[None, :]), other=0.0)
    
    # Store both original and transposed results
    # Original output
    tl.store(output_ptr + i_in[:, None] * N + j_in[None, :], input_tile, mask=(mask_i[:, None] & mask_j[None, :]))
    
    # Transposed output - swap i and j indices
    transpose_ptrs = transpose_ptr + j_in[:, None] * M + i_in[None, :]
    transpose_mask = (mask_j[:, None] & mask_i[None, :])
    tl.store(transpose_ptrs, input_tile, mask=transpose_mask)

@torch.fx.wrap
def optimized_transpose_output(input_tensor):
    """Optimized function to compute both tensor and its transpose"""
    M, N = input_tensor.shape
    
    # Configuring block sizes for optimal performance
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensors
    output = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    transpose_output = torch.empty((N, M), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch single kernel that computes both outputs
    optimized_transpose_output_kernel[(grid_m, grid_n)](
        input_tensor,
        output,
        transpose_output,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output, transpose_output

def replacement_func():
    """Return the optimized function"""
    return optimized_transpose_output