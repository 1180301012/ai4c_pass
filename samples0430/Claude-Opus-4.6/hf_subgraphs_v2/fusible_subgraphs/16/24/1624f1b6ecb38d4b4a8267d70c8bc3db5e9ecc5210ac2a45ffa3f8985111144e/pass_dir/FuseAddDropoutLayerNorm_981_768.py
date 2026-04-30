import torch
import triton
import triton.language as tl


def pattern(input_tensor, weight, bias, cls_token):
    conv = torch.conv3d(input_tensor, weight, bias, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    flat = conv.flatten(2)
    transposed = flat.transpose(1, 2)
    tiled = cls_token.tile([1, 1, 1])
    catted = torch.cat((tiled, transposed), dim=1)
    return catted


def replacement_args(input_tensor, weight, bias, cls_token):
    return (input_tensor, weight, bias, cls_token)


@triton.jit
def patchify_gemm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_ic: tl.constexpr,
    stride_it: tl.constexpr,
    stride_ih: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    nH: tl.constexpr,
    nW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute patch positions for this M tile
    m_vals = m_offsets[:, None]
    t_vals = m_vals // (nH * nW)
    h_vals = (m_vals // nW) % nH
    w_vals = m_vals % nW

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile: [BLOCK_N, BLOCK_K]
        w_ptrs = weight_ptr + n_offsets[:, None] * K + k_offsets[None, :]
        w_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Decompose k for patch addressing
        k_vals = k_offsets[None, :]
        c_vals = k_vals // 512
        dt_vals = (k_vals // 256) % 2
        dh_vals = (k_vals // 16) % 16
        dw_vals = k_vals % 16

        # Compute input addresses: [BLOCK_M, BLOCK_K]
        input_addrs = (c_vals * stride_ic +
                       (t_vals * 2 + dt_vals) * stride_it +
                       (h_vals * 16 + dh_vals) * stride_ih +
                       (w_vals * 16 + dw_vals))

        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(input_ptr + input_addrs, mask=a_mask, other=0.0)

        acc += tl.dot(a_tile, tl.trans(w_tile))

    # Add bias
    bias_vals = tl.load(bias_ptr + n_offsets, mask=n_offsets < N, other=0.0)
    acc = acc + bias_vals[None, :].to(tl.float32)

    # Store to output rows 1..980 (row 0 is cls_token)
    out_m = m_offsets + 1
    out_ptrs = out_ptr + out_m[:, None] * N + n_offsets[None, :]
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def copy_cls_kernel(
    cls_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    data = tl.load(cls_ptr + col_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + col_offsets, data, mask=mask)


@torch.fx.wrap
def fused_conv_transpose_cat(input_tensor, weight, bias, cls_token):
    M = 980
    N = 768
    K = 1536

    out = torch.empty(
        [1, 981, 768],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # Copy cls_token to row 0
    copy_cls_kernel[(1,)](
        cls_token, out,
        N=N, BLOCK_SIZE=1024, num_warps=4,
    )

    # Fused patchify + GEMM
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 16
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # 12

    patchify_gemm_kernel[(grid_m, grid_n)](
        input_tensor, weight, bias, out,
        stride_ic=501760,
        stride_it=50176,
        stride_ih=224,
        M=M, N=N, K=K,
        nH=14, nW=14,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_conv_transpose_cat