import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: conv2d (1×1) → dropout(p=0, identity) → residual add
# Shapes:  in_0=[128], in_1=[128,256,1,1], in_2=[1,128,4,256], in_3=[1,256,4,256]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """Match: conv2d(1x1) + dropoutidentity + residual add."""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    out = torch.nn.functional.dropout(conv2d, 0.0, False, False)
    result = out + in_2
    return result


def replacement_args(in_0, in_1, in_2, in_3):
    # in_0 = bias [128]
    # in_1 = weight [128, 256, 1, 1]
    # in_2 = residual [1, 128, 4, 256]
    # in_3 = input  [1, 256, 4, 256]
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused 1×1-conv + bias + residual-add Triton kernel
#
# NCHW-access strategy (ensures coalesced loads):
#   A_T[k, m]  = x_ptr  + k * HW + m        dim1-stride HW, dim0-stride 1  → M axis coalesced
#   W_T[k, n]  = w_ptr  + n * C_in + k       dim1-stride C_in, dim0-stride 1 → N axis coalesced
#   residual_B[n, m] = r_ptr + n * HW + m    dim1-stride HW,  dim0-stride 1 → M axis coalesced
#
# GEMM: out[m,n] = Σ_k A_T[k,m]*W_T[k,n] + bias[n] + residual[n,m]
#        = tl.dot(tl.trans(A_T_tile), W_T_tile) + bias + residual_B
#
# Output layout: out is [1, C_out, H, W] contiguous → NCHW
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # 16×16 tiles – maximum block count (64 blocks) for this problem
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_dropout_add_kernel(
    x_ptr,    # input  [1, C_in,  H, W]  strides: (C_in*HW, HW, W, 1)
    w_ptr,    # weight [C_out, C_in, 1, 1] strides: (C_in, 1, 1, 1)
    b_ptr,    # bias   [C_out]
    r_ptr,    # residual [1, C_out, H, W] strides: (C_out*HW, HW, W, 1)
    out_ptr,  # output  [1, C_out, H, W] strides: (C_out*HW, HW, W, 1)
    M, N, K,  # HW, C_out, C_in
    # x strides: stride(1)=HW (accessed as k-stride in A_T), stride(0)=C_in*HW (m-stride=1)
    sx_k,   # = x.stride(1) = H*W = HW
    sx_m,   # = x.stride(3) = 1   (fast axis for A_T load)
    # w strides: stride(0)=C_in (n-stride in W_T), stride(1)=1 (k-stride, fast axis for W_T)
    sw_n,   # = w.stride(0) = C_in
    sw_k,   # = w.stride(1) = 1
    # r/out strides: stride(1)=HW (n-stride in res_B), stride(0)=C_out*HW
    sr_n,   # = r.stride(1) = H*W = HW
    sr_m,   # = r.stride(3) = 1   (fast axis for residual_B load)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- K loop ----
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # ---- A_T tile [BLOCK_K, BLOCK_M]: stride along m=1 (fast) → coalesced ----
        # A_T[k, m] = x[0, k, m//W, m%W] = x_ptr + k*HW + m
        a_T_ptrs = x_ptr + offs_k[:, None] * sx_k + offs_m[None, :] * sx_m
        a_T_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
        a_T = tl.load(a_T_ptrs, mask=a_T_mask, other=0.0)  # [BLOCK_K, BLOCK_M]

        # ---- W_T tile [BLOCK_K, BLOCK_N]: stride along n=1 (fast) → coalesced ----
        # W_T[k, n] = weight[n, k, 0, 0] = w_ptr + n*C_in + k
        w_T_ptrs = w_ptr + offs_n[None, :] * sw_n + offs_k[:, None] * sw_k
        w_T_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_T = tl.load(w_T_ptrs, mask=w_T_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # GEMM step: trans(A_T) @ W_T  →  [BLOCK_M, BLOCK_N]
        acc = tl.dot(tl.trans(a_T), w_T, acc)

    # ---- add bias ----
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # ---- add residual_B [BLOCK_N, BLOCK_M]: stride along m=1 (fast) → coalesced ----
    # residual_B[n, m] = r[0, n, m//W, m%W] = r_ptr + n*HW + m
    res_B_ptrs = r_ptr + offs_n[:, None] * sr_n + offs_m[None, :] * sr_m
    res_B_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    res_B = tl.load(res_B_ptrs, mask=res_B_mask, other=0.0)  # [BLOCK_N, BLOCK_M]
    acc = acc + res_B.to(tl.float32).T  # broadcast over BLOCK_M

    # ---- store output [BLOCK_M, BLOCK_N] (NCHW contiguous: stride along n=1) ----
    o_ptrs = out_ptr + offs_m[:, None] * sx_m + offs_n[None, :] * sx_k
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=res_B_mask.T)


@torch.fx.wrap
def fused_conv1x1_add(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : residual [1, C_out, H, W]
    in_3 : input    [1, C_in,  H, W]
    """
    C_out = in_1.shape[0]
    C_in  = in_1.shape[1]
    H     = in_3.shape[2]
    W_dim = in_3.shape[3]
    HW    = H * W_dim

    M = HW     # 1024
    N = C_out  # 128
    K = C_in   # 256

    out = torch.empty_like(in_2)  # [1, C_out, H, W]

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _conv1x1_dropout_add_kernel[grid](
        in_3,   # x_ptr
        in_1,   # w_ptr
        in_0,   # b_ptr  (bias)
        in_2,   # r_ptr  (residual)
        out,    # out_ptr
        M, N, K,
        # A_T layour: x.stride(1) = HW (k-stride=HW), x.stride(3)=1 (m-stride=1)
        in_3.stride(1), in_3.stride(3),
        # W_T layout (m-stride=C_in, k-stride=1): w.stride(0)=C_in, w.stride(1)=1
        in_1.stride(0), in_1.stride(1),
        # res_B/out: r.stride(1)=HW, r.stride(3)=1
        in_2.stride(1), in_2.stride(3),
    )

    return out


def replacement_func():
    return fused_conv1x1_add