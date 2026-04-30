import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_unsqueeze_transpose_loop_kernel(
    input_ptr,
    output_ptr,
    M,  # 1024
    N,  # 128
    stride_in1,
    stride_in2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single program loops over all tiles
    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    total_tiles = num_m_tiles * num_n_tiles
    
    for tile_id in range(total_tiles):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N
        
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        
        # Coalesced reads: contiguous within rows
        input_offsets = m_offsets[:, None] * stride_in1 + n_offsets[None, :] * stride_in2
        mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
        
        data = tl.load(input_ptr + input_offsets, mask=mask)
        
        # Transpose and write
        data_t = tl.trans(data)
        output_offsets = n_offsets[:, None] * M + m_offsets[None, :]
        mask_t = (n_offsets[:, None] < N) & (m_offsets[None, :] < M)
        
        tl.store(output_ptr + output_offsets, data_t, mask=mask_t)


@torch.fx.wrap
def fused_unsqueeze_transpose(in_0):
    M = in_0.shape[1]  # 1024
    N = in_0.shape[2]  # 128

    out = torch.empty(1, 1, N, M, dtype=in_0.dtype, device=in_0.device)

    BLOCK_M = 128
    BLOCK_N = 128

    # Single program that loops over all tiles
    grid = (1,)

    fused_unsqueeze_transpose_loop_kernel[grid](
        input_ptr=in_0,
        output_ptr=out,
        M=M,
        N=N,
        stride_in1=in_0.stride(1),
        stride_in2=in_0.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out


def replacement_func():
    return fused_unsqueeze_transpose