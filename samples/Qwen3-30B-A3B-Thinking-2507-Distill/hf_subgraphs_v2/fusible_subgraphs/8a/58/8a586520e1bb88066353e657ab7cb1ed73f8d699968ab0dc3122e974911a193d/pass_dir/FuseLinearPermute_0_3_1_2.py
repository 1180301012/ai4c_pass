import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024,'BLOCK_N': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N_OUT'],
)
@triton.jit
def fused_linear_permute_kernel(
    a_ptr,       # input  [M, K=3]
    b_ptr,       # weight [N_OUT, K=3]
    bias_ptr,    # bias   [N_OUT]
    c_ptr,       # output [N_OUT, M]
    M,           # total rows = B * H * W
    N_OUT: tl.constexpr,   # 16
    N_IN:  tl.constexpr,   # 3
    K_pad: tl.constexpr,   # 16  (padded to meet tl.dot requirement)
    DTYPE: tl.constexpr,   # output element type
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K_pad)

    k_mask = offs_k < N_IN   # only first 3 are valid
    n_mask = offs_n < N_OUT  # all 16 are valid

    # Load weight b: [K_pad, N_OUT]  (transposed layout, with zero-padding)
    # b_t[k, n] = weight[n, k]  →  shape [K_pad=16, N_OUT=16]
    b_t = tl.load(
        b_ptr + offs_k[:, None] + offs_n[None, :] * N_IN,
        mask=k_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    # Load input a: [BLOCK_M, K_pad]  (transposed in K for tl.dot)
    # a[m, k] = in3[b, i, j, k]  →  linear offset = offs_m * N_IN + offs_k
    a = tl.load(
        a_ptr + offs_m[:, None] * N_IN + offs_k[None, :],
        mask=(offs_m[:, None] < M) & k_mask[None, :],
        other=0.0,
    )

    # Load bias: [BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    # GEMM with zero-padded K: [BLOCK_M, 16] @ [16, N_OUT] = [BLOCK_M, N_OUT]
    c = tl.dot(a, b_t, out_dtype=tl.float32)

    # Add bias (broadcast)
    c = c + bias[None, :].to(tl.float32)

    # Store in original layout [N_OUT, M] which corresponds to permuted output
    # Element [n, m] maps to output[0, n, i, j] after permute(0,3,1,2)
    tl.store(
        c_ptr + offs_n[None, :] * M + offs_m[:, None],
        c.to(DTYPE),
        mask=(offs_m[:, None] < M) & n_mask[None, :],
    )


@torch.fx.wrap
def fused_linear_permute(in_0, in_1, in_3):
    """
    Fused linear + permute(0, 3, 1, 2).

    in_0 : bias   [N_OUT]         (may be on CPU)
    in_1 : weight [N_OUT, N_IN]   (may be on CPU)
    in_3 : input  [B, H, W, N_IN] (on CUDA)  -- contiguous
    returns: [B, N_OUT, H, W]

    Memory layout insight (no reshape needed):
      - in_3[0, i, j, k] lives at ptr + (i*W + j)*N_IN + k = ptr + m*N_IN + k
      - out[0, n, i, j]  lives at ptr + n*M + m   where m = i*W + j, M = H*W
    """
    device = in_3.device
    dtype  = in_3.dtype

    # torch.as_tensor is in the allowed whitelist; moves CPU weights to GPU
    weight = torch.as_tensor(in_1, dtype=dtype, device=device)
    bias   = torch.as_tensor(in_0, dtype=dtype, device=device)

    B, H, W, N_IN = in_3.shape
    N_OUT = weight.shape[0]
    M     = B * H * W

    # Allocate output directly in the permuted layout — no reshape
    out = torch.empty((B, N_OUT, H, W), dtype=dtype, device=device)

    # DTYPE constexpr
    if dtype == torch.float16:
        DTYPE = tl.float16
    elif dtype == torch.bfloat16:
        DTYPE = tl.bfloat16
    else:
        DTYPE = tl.float32

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_OUT, meta['BLOCK_N']))

    # Pass tensors directly — kernel computes strides internally
    fused_linear_permute_kernel[grid](
        in_3, weight, bias, out,
        M, N_OUT, N_IN, 16, DTYPE,
    )

    return out


def replacement_func():
    return fused_linear_permute