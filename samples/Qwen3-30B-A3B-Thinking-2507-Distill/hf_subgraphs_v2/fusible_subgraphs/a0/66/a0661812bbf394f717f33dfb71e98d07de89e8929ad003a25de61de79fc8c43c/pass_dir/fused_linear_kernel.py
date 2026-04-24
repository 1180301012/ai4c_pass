import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
    ],
    key=['S', 'D_in', 'D_out'],
)
@triton.jit
def _gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    S, D_in, D_out,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B^T + bias
      A:  [S, D_in]   (input viewed as 2-D using strides from 3-D tensor)
      B:  [D_out, D_in] (weight, row-major)
      bias: [D_out]
      C:  [S, D_out]   (same layout as torch.nn.functional.linear output)

    Key design:
      - Load first A tile (_a0) BEFORE the accumulation loop so _a0.dtype is
        accessible in every branch below (Triton JIT cannot always see loop
        variable dtype outside the loop body).
      - Use tl.constexpr if/elif on _a0.dtype for compile-time dtype branching.
      - B loaded as [BLOCK_N, BLOCK_K] and transposed for tl.dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load first K-tile to capture element dtype (before accumulation loop)
    offs_k0 = tl.arange(0, BLOCK_K)
    _a0 = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak,
        mask=(offs_m[:, None] < S) & (offs_k0[None, :] < D_in),
        other=0.0,
    )

    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(D_in, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < S) & (offs_k[None, :] < D_in),
            other=0.0,
        )

        # Load B as [BLOCK_N, BLOCK_K], transpose for A @ B^T
        b = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=(offs_n[:, None] < D_out) & (offs_k[None, :] < D_in),
            other=0.0,
        )

        acc += tl.dot(a, tl.trans(b))

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < D_out, other=0.0)
    acc += bias[None, :].to(tl.float32)

    out_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < S) & (offs_n[None, :] < D_out)

    # Compile-time dtype branching using _a0.dtype (constexpr in Triton JIT)
    if _a0.dtype == tl.float16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask)
    elif _a0.dtype == tl.bfloat16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)


@torch.fx.wrap
def linear_triton_dispatch(in_0, in_1, in_2, route):
    """
    Fused GEMM+bias kernel for linear(in_2, in_1, in_0).
    in_0: bias  [D_out]
    in_1: weight [D_out, D_in]
    in_2: input  [1 or B, S, D_in]
    route: dispatch tag shared across all 4 passes (same replacement_func_limit)
    Returns: linear output [1 or B, S, D_out]
    No .view()/.reshape() — only torch.empty allocation is used.
    Strides are passed directly so PoisonDispatchTensor whitelist is respected.
    """
    B    = in_2.shape[0]
    S    = in_2.shape[1]
    D_in = in_2.shape[2]
    D_out = in_1.shape[0]

    # Allocate output in the same shape/layout as torch.nn.functional.linear
    c = torch.empty((B, S, D_out), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_M']),
                         triton.cdiv(D_out, meta['BLOCK_N']))

    _gemm_bias_kernel[grid](
        in_2, in_1, in_0, c,
        S, D_in, D_out,
        # A strides (dim 1 and 2 of the 3-D input tensor)
        in_2.stride(1), in_2.stride(2),
        # B strides (row-major [D_out, D_in])
        in_1.stride(0), in_1.stride(1),
        # C strides (dim 1 and 2 of the 3-D output tensor)
        c.stride(1), c.stride(2),
    )

    return c