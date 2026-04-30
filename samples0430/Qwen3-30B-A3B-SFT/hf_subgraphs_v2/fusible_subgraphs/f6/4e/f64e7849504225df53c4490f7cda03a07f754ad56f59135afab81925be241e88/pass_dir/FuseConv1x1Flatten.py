import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: computes  out = x_2d @ w.T + bias
#   x_2d  : [M, K]  where M = N*HW, K = C_in  (NCHW viewed as 2-D)
#   w     : [N_OUT, K]
#   bias  : [N_OUT]
#   out   : [N, N_OUT, HW]  — written directly in flattened layout
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def _conv1x1_flatten_kernel_f32(
    x_ptr,    # input  [N, C_in, H, W]  (float32)
    w_ptr,    # weight [C_out, C_in]     (float32)
    b_ptr,    # bias   [C_out]
    out_ptr,  # output [N, C_out, H*W]   (float32)
    M,        # = N * HW
    N_OUT,    # = C_out
    K,        # = C_in
    HW,       # = H * W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        batch_m  = m_offs // HW
        spatial_m = m_offs % HW
        x_addr  = batch_m[:, None] * (K * HW) + k_offs[None, :] * HW + spatial_m[:, None]
        x_mask  = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x_tile  = tl.load(x_ptr + x_addr, mask=x_mask, other=0.0)
        w_addr  = n_offs[:, None] * K + k_offs[None, :]
        w_mask  = (n_offs[:, None] < N_OUT) & (k_offs[None, :] < K)
        w_tile  = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    b_offs = n_start + tl.arange(0, BLOCK_N)
    bias   = tl.load(b_ptr + b_offs, mask=b_offs < N_OUT, other=0.0)
    acc   += bias[None, :].to(tl.float32)
    batch_m  = m_offs // HW
    spatial_m = m_offs % HW
    out_addr  = batch_m[:, None] * (N_OUT * HW) + n_offs[None, :] * HW + spatial_m[:, None]
    out_mask  = (m_offs[:, None] < M) & (n_offs[None, :] < N_OUT)
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def _conv1x1_flatten_kernel_fp16(
    x_ptr,    # input  [N, C_in, H, W]  (float16)
    w_ptr,    # weight [C_out, C_in]     (float16)
    b_ptr,    # bias   [C_out]
    out_ptr,  # output [N, C_out, H*W]   (float16)
    M,        # = N * HW
    N_OUT,    # = C_out
    K,        # = C_in
    HW,       # = H * W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        batch_m  = m_offs // HW
        spatial_m = m_offs % HW
        x_addr  = batch_m[:, None] * (K * HW) + k_offs[None, :] * HW + spatial_m[:, None]
        x_mask  = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x_tile  = tl.load(x_ptr + x_addr, mask=x_mask, other=0.0)
        w_addr  = n_offs[:, None] * K + k_offs[None, :]
        w_mask  = (n_offs[:, None] < N_OUT) & (k_offs[None, :] < K)
        w_tile  = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    b_offs = n_start + tl.arange(0, BLOCK_N)
    bias   = tl.load(b_ptr + b_offs, mask=b_offs < N_OUT, other=0.0)
    acc   += bias[None, :].to(tl.float32)
    batch_m  = m_offs // HW
    spatial_m = m_offs % HW
    out_addr  = batch_m[:, None] * (N_OUT * HW) + n_offs[None, :] * HW + spatial_m[:, None]
    out_mask  = (m_offs[:, None] < M) & (n_offs[None, :] < N_OUT)
    tl.store(out_ptr + out_addr, acc.to(tl.float16), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def _conv1x1_flatten_kernel_bf16(
    x_ptr,    # input  [N, C_in, H, W]  (bfloat16)
    w_ptr,    # weight [C_out, C_in]     (bfloat16)
    b_ptr,    # bias   [C_out]
    out_ptr,  # output [N, C_out, H*W]   (bfloat16)
    M,        # = N * HW
    N_OUT,    # = C_out
    K,        # = C_in
    HW,       # = H * W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        batch_m  = m_offs // HW
        spatial_m = m_offs % HW
        x_addr  = batch_m[:, None] * (K * HW) + k_offs[None, :] * HW + spatial_m[:, None]
        x_mask  = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x_tile  = tl.load(x_ptr + x_addr, mask=x_mask, other=0.0)
        w_addr  = n_offs[:, None] * K + k_offs[None, :]
        w_mask  = (n_offs[:, None] < N_OUT) & (k_offs[None, :] < K)
        w_tile  = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    b_offs = n_start + tl.arange(0, BLOCK_N)
    bias   = tl.load(b_ptr + b_offs, mask=b_offs < N_OUT, other=0.0)
    acc   += bias[None, :].to(tl.float32)
    batch_m  = m_offs // HW
    spatial_m = m_offs % HW
    out_addr  = batch_m[:, None] * (N_OUT * HW) + n_offs[None, :] * HW + spatial_m[:, None]
    out_mask  = (m_offs[:, None] < M) & (n_offs[None, :] < N_OUT)
    tl.store(out_ptr + out_addr, acc.to(tl.bfloat16), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: exactly mirrors the computation in model.py
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    # in_0 = bias [C_out],  in_1 = weight [C_out, C_in, 1, 1],  in_2 = input [N, C_in, H, W]
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrapper (must be @torch.fx.wrap so the tracer treats it as a leaf)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _conv1x1_flatten_wrapper(bias, weight, x):
    """
    bias   : [C_out]
    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    returns: [N, C_out, H*W]   (conv1x1 + flatten fused)

    The kernel's address arithmetic directly handles the NCHW memory layout
    without any .view() / .reshape() calls:
        x[n, k, hw]  →  x_ptr + n*K*HW + k*HW + hw
        out[n, co, hw] → out_ptr + n*C_out*HW + co*HW + hw
    """
    N     = x.shape[0]
    C_in  = x.shape[1]
    H     = x.shape[2]
    W     = x.shape[3]
    C_out = weight.shape[0]
    HW    = H * W
    M     = N * HW

    out = torch.empty((N, C_out, HW), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M,    meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    if x.dtype == torch.float32:
        _conv1x1_flatten_kernel_f32[grid](x, weight, bias, out, M, C_out, C_in, HW)
    elif x.dtype == torch.float16:
        _conv1x1_flatten_kernel_fp16[grid](x, weight, bias, out, M, C_out, C_in, HW)
    else:   # bfloat16
        _conv1x1_flatten_kernel_bf16[grid](x, weight, bias, out, M, C_out, C_in, HW)

    return out


def replacement_func():
    return _conv1x1_flatten_wrapper