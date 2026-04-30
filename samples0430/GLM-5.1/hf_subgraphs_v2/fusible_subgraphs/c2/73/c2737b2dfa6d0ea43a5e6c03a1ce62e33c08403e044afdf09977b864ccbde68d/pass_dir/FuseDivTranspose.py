import torch
import triton
import triton.language as tl


def pattern(in_0, scale):
    tmp_0 = in_0 / scale
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)


def replacement_args(in_0, scale):
    return (in_0, scale)


@triton.jit
def scale_transpose_kernel(
    input_ptr,
    output_ptr,
    inv_scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 3D grid: (B*H, M_tiles, N_tiles)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute strides for contiguous tensors inside the kernel
    # Input [B, H, M, N]: strides (H*M*N, M*N, N, 1)
    # Combined (b*H+h) offset: pid_bh * M*N
    # Output [B, H, N, M]: strides (H*N*M, N*M, M, 1)
    # Combined (b*H+h) offset: pid_bh * N*M
    stride_in_bh = M * N
    stride_in_m = N
    stride_out_bh = N * M
    stride_out_n = M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Input offsets: [pid_bh, offs_m, offs_n] with strides (M*N, N, 1)
    input_offsets = pid_bh * stride_in_bh + offs_m[:, None] * stride_in_m + offs_n[None, :]

    input_tile = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Apply inverse scale (multiplication is faster than division)
    scaled_tile = input_tile * inv_scale

    # Output offsets: [pid_bh, offs_n, offs_m] with strides (N*M, M, 1)
    # Note: offs_m[:, None] uses stride_out_m=1, offs_n[None, :] uses stride_out_n=M
    output_offsets = pid_bh * stride_out_bh + offs_m[:, None] + offs_n[None, :] * stride_out_n

    tl.store(output_ptr + output_offsets, scaled_tile, mask=mask)


@torch.fx.wrap
def scale_transpose(input_tensor, scale):
    inv_scale = 1.0 / scale

    # Handle 4D tensors [B, H, M, N] -> [B, H, N, M]
    B, H, M, N = input_tensor.shape

    output = torch.empty(B, H, N, M, dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_M = 128
    BLOCK_N = 8

    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N

    grid = (B * H, num_m_tiles, num_n_tiles)

    scale_transpose_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        inv_scale=inv_scale,
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


def replacement_func():
    return scale_transpose