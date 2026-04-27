import torch
import triton
import triton.language as tl


@triton.jit
def _av_matmul_reshape_kernel(
    X_ptr, Y_ptr, Out_ptr,
    # X strides: [B, H, N_Q, N_K]
    stride_xb, stride_xh, stride_xm, stride_xk,
    # Y strides: [B, H, N_K, D]
    stride_yb, stride_yh, stride_yk, stride_yd,
    # Out strides: [B, N_Q, H*D]  (last dim stride=1 implicit)
    stride_ob, stride_on,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,     # actual head dim (e.g. 80)
    HEAD_DIM_P1: tl.constexpr,  # largest power-of-2 <= HEAD_DIM (e.g. 64)
    HEAD_DIM_P2: tl.constexpr,  # remainder = HEAD_DIM - HEAD_DIM_P1 (e.g. 16)
    H: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_F16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d1 = tl.arange(0, HEAD_DIM_P1)          # 0..63
    offs_d2 = tl.arange(0, HEAD_DIM_P2)          # 0..15 (offset by HEAD_DIM_P1 below)
    m_mask = offs_m < N_Q

    # Two accumulators — no wasted computation from padding
    acc1 = tl.zeros([BLOCK_M, HEAD_DIM_P1], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, HEAD_DIM_P2], dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(N_K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < N_K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr
                  + pid_b * stride_xb
                  + pid_h * stride_xh
                  + offs_m[:, None] * stride_xm
                  + offs_k[None, :] * stride_xk)
        x_tile = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)

        # Load Y part-1: dims 0..HEAD_DIM_P1-1  [BLOCK_K, HEAD_DIM_P1]
        y1_ptrs = (Y_ptr
                   + pid_b * stride_yb
                   + pid_h * stride_yh
                   + offs_k[:, None] * stride_yk
                   + offs_d1[None, :] * stride_yd)
        y1_tile = tl.load(y1_ptrs, mask=k_mask[:, None], other=0.0)
        acc1 += tl.dot(x_tile, y1_tile)

        # Load Y part-2: dims HEAD_DIM_P1..HEAD_DIM-1  [BLOCK_K, HEAD_DIM_P2]
        y2_ptrs = (Y_ptr
                   + pid_b * stride_yb
                   + pid_h * stride_yh
                   + offs_k[:, None] * stride_yk
                   + (HEAD_DIM_P1 + offs_d2[None, :]) * stride_yd)
        y2_tile = tl.load(y2_ptrs, mask=k_mask[:, None], other=0.0)
        acc2 += tl.dot(x_tile, y2_tile)

    # Convert to output dtype
    if IS_BF16:
        v1 = acc1.to(tl.bfloat16)
        v2 = acc2.to(tl.bfloat16)
    elif IS_F16:
        v1 = acc1.to(tl.float16)
        v2 = acc2.to(tl.float16)
    else:
        v1 = acc1
        v2 = acc2

    # Write part-1 to [B, N_Q, H*D] — head h occupies positions [h*HD .. h*HD+HD)
    out1_ptrs = (Out_ptr
                 + pid_b * stride_ob
                 + offs_m[:, None] * stride_on
                 + (pid_h * HEAD_DIM + offs_d1[None, :]))
    tl.store(out1_ptrs, v1, mask=m_mask[:, None])

    # Write part-2
    out2_ptrs = (Out_ptr
                 + pid_b * stride_ob
                 + offs_m[:, None] * stride_on
                 + (pid_h * HEAD_DIM + HEAD_DIM_P1 + offs_d2[None, :]))
    tl.store(out2_ptrs, v2, mask=m_mask[:, None])


@torch.fx.wrap
def av_matmul_reshape(x, y):
    """
    x: attention weights [B, H, N_Q, N_K]
    y: value states     [B, H, N_K, D]
    Returns: [B, N_Q, H*D]  (fused matmul + transpose(1,2) + reshape)
    """
    B, H, N_Q, N_K = x.shape
    D = y.shape[3]
    dtype = x.dtype

    is_bf16 = (dtype == torch.bfloat16)
    is_f16 = (dtype == torch.float16)

    # Split D into two power-of-2 parts: e.g. D=80 → P1=64, P2=16
    p1 = 1 << (D.bit_length() - 1)   # largest power of 2 <= D
    p2 = D - p1                        # remainder (must be 0 or power of 2)
    if p2 == 0:
        p2 = 1   # dummy; will be unused since no valid elements

    out = torch.empty((B, N_Q, H * D), dtype=dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_K = 64

    grid = (B, H, triton.cdiv(N_Q, BLOCK_M))

    _av_matmul_reshape_kernel[grid](
        x, y, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        out.stride(0), out.stride(1),
        N_Q=N_Q,
        N_K=N_K,
        HEAD_DIM=D,
        HEAD_DIM_P1=p1,
        HEAD_DIM_P2=p2,
        H=H,
        IS_BF16=is_bf16,
        IS_F16=is_f16,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


def pattern(x, y):
    matmul_1 = torch.matmul(x, y)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return av_matmul_reshape