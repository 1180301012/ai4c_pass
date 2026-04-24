import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# GEMM kernel: 1×1 conv as matrix multiply
#   input  : [N_batch, C_in, H, W]  viewed as [N_batch*H*W, C_in]
#   weight : [C_out, C_in]           (original [C_out, C_in, 1, 1] stripped)
#   bias   : [C_out]
#   output : [N_batch, C_out, H, W]  viewed as [N_batch*H*W, C_out]
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M, N, K,
    H, W,
    stride_ib,   # input  batch stride  = C_in * H * W
    stride_ic,   # input  channel stride = H * W
    stride_wk,   # weight row stride    = C_in
    stride_ob,   # output batch stride  = C_out * H * W
    stride_on,   # output channel stride = H * W
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

    # Decompose flat row m = batch * H*W + spatial
    HW = H * W
    batch   = m_offs // HW
    spatial = m_offs % HW

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load input as [BLOCK_M, BLOCK_K]
        # (k_mask broadcasts to [BLOCK_M, BLOCK_K] for input)
        inp_ptrs = (
            input_ptr
            + batch[:, None] * stride_ib
            + k_offs[None, :] * stride_ic
            + spatial[:, None]
        )
        inp_mask = (m_offs[:, None] < M) & k_mask[None, :]
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)   # [BLOCK_M, BLOCK_K]

        # weight: w[n, k] = weight_ptr + n*stride_wk + k
        w_ptrs = weight_ptr + n_offs[:, None] * stride_wk + k_offs[None, :]
        w_mask = (n_offs[:, None] < N) & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc += tl.dot(inp, tl.trans(w))

    # Add bias
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output
    out_ptrs = (
        out_ptr
        + batch[:, None] * stride_ob
        + n_offs[None, :] * stride_on
        + spatial[:, None]
    )
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _conv1x1_impl(in_3, in_1, in_0):
    N_batch, C_in, H, W = in_3.shape
    C_out = in_1.shape[0]
    M = N_batch * H * W
    out = torch.empty((N_batch, C_out, H, W), dtype=in_3.dtype, device=in_3.device)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    _conv1x1_gemm_kernel[grid](
        in_3, in_1, in_0, out,
        M, C_out, C_in, H, W,
        in_3.stride(0), in_3.stride(1),
        in_1.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Mean reduction kernel: reduce dim=-2 (M dimension), keepdim=True
#   input  : [N_batch, M, N]
#   output : [N_batch, 1, N]
#
# Grid: (N_batch * N, cdiv(M, TILE_M))
# Each program accumulates a TILE_M×TILE_N tile over the M dimension,
# loading TILE_N consecutive elements (coalesced) per row step.
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 8,  'TILE_N': 256, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'TILE_M': 16, 'TILE_N': 256, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'TILE_M': 4,  'TILE_N': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'TILE_M': 8,  'TILE_N': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'TILE_M': 16, 'TILE_N': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'TILE_M': 4,  'TILE_N': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'TILE_M': 8,  'TILE_N': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'TILE_M': 16, 'TILE_N': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'TILE_M': 8,  'TILE_N': 256, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'TILE_M': 16, 'TILE_N': 256, 'num_stages': 2, 'num_warps': 8}),
    ],
    key=['M', 'N'],
)
@triton.jit
def _mean_reduction_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_xb,   # input  batch stride = M * N
    stride_xm,   # input  row stride   = N
    stride_xn,   # input  col stride   = 1
    stride_ob,   # output batch stride = N   (output [N_batch, 1, N])
    stride_on,   # output col stride   = 1
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid_bm = tl.program_id(0)   # encodes (batch b, output column n)
    pid_m  = tl.program_id(1)   # tile index along M

    n = pid_bm % N
    b = pid_bm // N

    m_start = pid_m * TILE_M
    m_offs  = m_start + tl.arange(0, TILE_M)   # [TILE_M]
    n_offs  = tl.arange(0, TILE_N)             # [TILE_N]

    m_mask = m_offs < M
    # n_offs is always < N when TILE_N == N (use all N channels at once)

    # Load TILE_M rows × TILE_N cols — TILE_N contiguous reads per row → coalesced
    x_ptrs = (x_ptr
              + b * stride_xb
              + n * stride_xn                         # start at column n
              + m_offs[:, None] * stride_xm
              + n_offs[None, :] * stride_xn)

    x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)   # [TILE_M, TILE_N]

    # Reduce over M dimension
    acc = tl.sum(x, axis=0)   # [TILE_N]
    mean_val = acc / M

    # Write output[b, 0, n_offs] — contiguous TILE_N stores
    tl.store(out_ptr + b * stride_ob + n_offs * stride_on,
             mean_val.to(x_ptr.dtype.element_ty),
             mask=n_offs < N)


def _mean_impl(in_2):
    N_batch = in_2.shape[0]
    M = in_2.shape[1]
    N = in_2.shape[2]
    out = torch.empty((N_batch, 1, N), dtype=in_2.dtype, device=in_2.device)
    grid = lambda meta: (
        N_batch * N,
        triton.cdiv(M, meta['TILE_M']),
    )
    _mean_reduction_kernel[grid](
        in_2, out,
        M, N,
        in_2.stride(0),   # stride_xb
        in_2.stride(1),   # stride_xm
        in_2.stride(2),   # stride_xn
        out.stride(0),    # stride_ob
        out.stride(2),    # stride_on
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch (returned by replacement_func in both pass files)
#   route = last element of args (string constant inserted by replacement_args)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _shared_dispatch(*args):
    route = args[-1]
    if route == "conv1x1_bias":
        return _conv1x1_impl(args[0], args[1], args[2])
    elif route == "mean_dim1":
        return _mean_impl(args[0])