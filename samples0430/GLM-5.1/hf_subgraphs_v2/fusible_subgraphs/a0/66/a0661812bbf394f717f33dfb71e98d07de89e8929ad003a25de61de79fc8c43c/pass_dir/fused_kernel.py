import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_linear_ptr, out_transpose_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_bb,
    stride_olm, stride_oln,
    stride_otn, stride_otm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input: [BLOCK_M, BLOCK_K]
        input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)

        # Load weight in natural order: [BLOCK_N, BLOCK_K] (coalesced)
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Dot product: input @ weight.T
        # tl.trans(weight_tile) gives [BLOCK_K, BLOCK_N]
        acc += tl.dot(input_tile, tl.trans(weight_tile))

    # Load bias: [BLOCK_N]
    bias_ptrs = bias_ptr + offs_n * stride_bb
    bias_mask = offs_n < N
    bias_tile = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias_tile[None, :]

    # Cast accumulator to output dtype
    if IS_FP16:
        acc_out = acc.to(tl.float16)
    elif IS_BF16:
        acc_out = acc.to(tl.bfloat16)
    else:
        acc_out = acc

    # Store to linear output: position [offs_m, offs_n] in [B, M, N]
    linear_ptrs = out_linear_ptr + offs_m[:, None] * stride_olm + offs_n[None, :] * stride_oln
    linear_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(linear_ptrs, acc_out, mask=linear_mask)

    # Store to transpose output: position [offs_n, offs_m] in [B, N, M]
    transpose_ptrs = out_transpose_ptr + offs_n[:, None] * stride_otn + offs_m[None, :] * stride_otm
    transpose_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(transpose_ptrs, tl.trans(acc_out), mask=transpose_mask)


@torch.fx.wrap
def fused_linear_dropout_transpose_dispatch(bias, weight, input, route):
    B, M, K = input.shape
    N = weight.shape[0]

    out_linear = torch.empty((B, M, N), device=input.device, dtype=input.dtype)
    out_transpose = torch.empty((B, N, M), device=input.device, dtype=input.dtype)

    IS_FP16 = input.dtype == torch.float16
    IS_BF16 = input.dtype == torch.bfloat16

    stride_im = input.stride()[1]
    stride_ik = input.stride()[2]
    stride_wn = weight.stride()[0]
    stride_wk = weight.stride()[1]
    stride_bb = bias.stride()[0]
    stride_olm = out_linear.stride()[1]
    stride_oln = out_linear.stride()[2]
    stride_otn = out_transpose.stride()[1]
    stride_otm = out_transpose.stride()[2]

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_transpose_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias,
        out_linear_ptr=out_linear, out_transpose_ptr=out_transpose,
        M=M, N=N, K=K,
        stride_im=stride_im, stride_ik=stride_ik,
        stride_wn=stride_wn, stride_wk=stride_wk,
        stride_bb=stride_bb,
        stride_olm=stride_olm, stride_oln=stride_oln,
        stride_otn=stride_otn, stride_otm=stride_otm,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    if route == "d3_t4":
        return (out_linear, out_transpose)
    elif route == "t4_d3":
        return (out_transpose, out_linear)
    else:
        raise ValueError(f"Unknown route: {route}")