import torch
import triton
import triton.language as tl


def pattern(in_0):
    return in_0.transpose(-2, -1)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def transpose_last2_kernel(
    input_ptr,
    output_ptr,
    num_matrices,
    M,
    N,
    MN,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid_matrix = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    matrix_base = pid_matrix * MN
    input_offsets = matrix_base + m_offsets[:, None] * N + n_offsets[None, :]
    input_mask = m_mask[:, None] & n_mask[None, :]
    
    tile = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Write transposed tile to output
    output_offsets = matrix_base + n_offsets[:, None] * M + m_offsets[None, :]
    output_mask = n_mask[:, None] & m_mask[None, :]
    
    tl.store(output_ptr + output_offsets, tile.T, mask=output_mask)


@torch.fx.wrap
def transpose_last2_wrapper(in_0):
    shape = in_0.shape
    M = shape[-2]
    N = shape[-1]
    n_elements = in_0.numel()
    num_matrices = n_elements // (M * N)
    MN = M * N
    
    # Output shape: swap last two dimensions
    out_shape = list(shape)
    out_shape[-2] = N
    out_shape[-1] = M
    
    # Tile sizes (must be powers of 2)
    TILE_M = 16
    TILE_N = 16
    
    grid = (num_matrices, triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    transpose_last2_kernel[grid](
        input_ptr=in_0,
        output_ptr=out,
        num_matrices=num_matrices,
        M=M,
        N=N,
        MN=MN,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
    )
    return out


def replacement_func():
    return transpose_last2_wrapper