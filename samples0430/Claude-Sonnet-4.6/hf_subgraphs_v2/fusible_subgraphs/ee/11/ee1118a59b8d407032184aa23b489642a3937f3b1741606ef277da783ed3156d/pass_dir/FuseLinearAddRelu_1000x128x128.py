import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match: linear(in_3, in_1, in_0) + in_2 + relu_()
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused Triton kernel:  out = ReLU(in_2 + in_3 @ in_1.T + in_0)
#   in_3 : [M, K]   (input activations, CUDA)
#   in_1 : [N, K]   (weight matrix,     CPU → moved to CUDA in wrapper)
#   in_0 : [N]      (bias vector,        CPU → moved to CUDA in wrapper)
#   in_2 : [M, N]   (residual,           CUDA)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_add_relu_kernel(
    x_ptr,        # [M, K] input
    w_ptr,        # [N, K] weight
    bias_ptr,     # [N]    bias
    res_ptr,      # [M, N] residual
    out_ptr,      # [M, N] output
    M, N, K,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    stride_rm: tl.constexpr, stride_rn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base pointers for x and w
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :] * stride_xk
    # w is [N, K]; we want w[n, k] to compute x @ w.T
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + tl.arange(0, BLOCK_K)[:, None] * stride_wk

    # -----------------------------------------------------------------------
    # Main GEMM loop — accumulate in float32 for precision
    # -----------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        # Load x tile — mask on M (K always divisible)
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # Load w tile — mask on N (K always divisible)
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # -----------------------------------------------------------------------
    # Epilogue: bias + residual + ReLU
    # -----------------------------------------------------------------------
    # bias [N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # residual [M, N]
    r_ptrs = res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    residual = tl.load(
        r_ptrs,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc += residual.to(tl.float32)

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store result (cast back to original dtype)
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Kernel wrapper — must be @torch.fx.wrap so the framework can replace the
# matched subgraph with this function call.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    """
    Computes: ReLU(in_2 + in_3 @ in_1.T + in_0)
    in_0 : bias  [N]       (may be on CPU)
    in_1 : weight [N, K]   (may be on CPU)
    in_2 : residual [M, N] (CUDA)
    in_3 : input [M, K]    (CUDA)
    """
    device = in_3.device
    dtype  = in_3.dtype

    # Move weight/bias to GPU with matching dtype
    w    = in_1.to(device=device, dtype=dtype, non_blocking=True)
    bias = in_0.to(device=device, dtype=dtype, non_blocking=True)

    M, K = in_3.shape
    N    = w.shape[0]          # weight is [N, K]

    out = torch.empty((M, N), device=device, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _fused_linear_add_relu_kernel[grid](
        in_3, w, bias, in_2, out,
        M, N, K,
        in_3.stride(0), in_3.stride(1),
        w.stride(0),    w.stride(1),
        in_2.stride(0), in_2.stride(1),
        out.stride(0),  out.stride(1),
    )

    return (out,)


def replacement_func():
    return fused_linear_add_relu