import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: linear(in_3, in_1, in_0) -> view(1,1,-1,64) -> transpose(1,2)
#          -> contiguous()
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused GEMV (M=1, N=512, K=512) + bias + reshape
#
# Grid: (8,) — one program per attention head
# Each program computes 64 outputs for its head:
#   out[pid*64+d] = sum_k( x[k] * w[pid*64+d, k] ) + b[pid*64+d]
# Uses element-wise multiply + tl.sum (no tl.dot, no Triton indexing issues).
# DTYPE is hardcoded to bfloat16 (works for float16 with tl.float16 too).
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_reshape_kernel(
    x_ptr,   # [1, 1, K]  — flat K-vector (row 0)
    w_ptr,   # [512, K]
    b_ptr,   # [512]
    out_ptr, # [1, 8, 1, 64]  contiguous (= flat 512-vector)
    K,
    DTYPE: tl.constexpr,    # tl.bfloat16 or tl.float16
    BLOCK_K: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = 64

    pid = tl.program_id(0)
    offs_n = pid * HEAD_DIM + tl.arange(0, HEAD_DIM)   # [64]
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x tile [BLOCK_K]
        x = tl.load(x_ptr + offs_k)

        # Load w tile [HEAD_DIM, BLOCK_K]  (always in bounds: N=512, K=512)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])

        # acc[d] += sum_k( w[d,k] * x[k] )
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)

    # Add bias
    b = tl.load(b_ptr + offs_n)
    acc += b.to(tl.float32)

    # Store using compile-time DTYPE — x is NOT referenced after the loop
    tl.store(out_ptr + offs_n, acc.to(DTYPE))


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_linear_reshape(in_0, in_1, in_3):
    """
    Replaces: F.linear(in_3, in_1, in_0).view(1,1,-1,64).transpose(1,2).contiguous()
    in_0 : bias   [512]
    in_1 : weight [512, K]
    in_3 : input  [1, 1, K]
    Returns: [1, 8, 1, 64]  contiguous
    """
    K = in_1.shape[1]   # 512
    out = torch.empty((1, 8, 1, 64), dtype=in_3.dtype, device=in_3.device)

    dtype = tl.bfloat16 if in_3.dtype == torch.bfloat16 else tl.float16

    fused_linear_reshape_kernel[(8,)](
        in_3, in_1, in_0, out, K,
        DTYPE=dtype, BLOCK_K=64,
    )

    return out


def replacement_func():
    return fused_linear_reshape