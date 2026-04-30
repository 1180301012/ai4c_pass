import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        x_offs = rm[:, None] * stride_xm + rk[None, :] * stride_xk
        w_offs = rn[None, :] * stride_wn + rk[:, None] * stride_wk

        mask_x = (rm[:, None] < M) & (rk[None, :] < K)
        mask_w = (rk[:, None] < K) & (rn[None, :] < N)

        x = tl.load(x_ptr + x_offs, mask=mask_x, other=0.0)
        w = tl.load(w_ptr + w_offs, mask=mask_w, other=0.0)

        acc += tl.dot(x, w)

    # Add bias
    mask_n = rn < N
    bias = tl.load(b_ptr + rn, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Store output
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptr + out_offs, acc, mask=mask_out)


def _run_linear(x, w, b):
    """Run linear: x @ w.T + b, handling any input dimensionality"""
    K = x.shape[-1]
    M = x.numel() // K
    N = w.shape[0]
    dtype = x.dtype
    device = x.device

    out_shape = list(x.shape)
    out_shape[-1] = N
    out = torch.empty(out_shape, dtype=dtype, device=device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_m_blocks * num_n_blocks,)

    matmul_bias_kernel[grid](
        x, w, b, out,
        M, K, N,
        K, 1,
        w.stride(0), w.stride(1),
        N, 1,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return out


def _run_reshape_linear(in_4, w, b):
    """Reshape in_4 from [1,150,1,512] to [300,1,256] then linear to [300,1,512]"""
    K = 256
    M = in_4.numel() // K  # 300
    N = w.shape[0]  # 512
    dtype = in_4.dtype
    device = in_4.device

    out = torch.empty((M, 1, N), dtype=dtype, device=device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_m_blocks * num_n_blocks,)

    matmul_bias_kernel[grid](
        in_4, w, b, out,
        M, K, N,
        K, 1,
        w.stride(0), w.stride(1),
        N, 1,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return out


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "route_linear1":
        return _run_linear(args[0], args[1], args[2])
    else:
        return _run_reshape_linear(args[0], args[1], args[2])