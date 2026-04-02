import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match torch.matmul(A, B) — [M,K]×[K,1] → [M,1]
# ---------------------------------------------------------------------------
def pattern(in_2, in_3):
    return torch.matmul(in_2, in_3)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ---------------------------------------------------------------------------
# Per-row Triton kernel: one block per output row, 1-D K-reduction.
# BLOCK_K=128 divides both K=768 (6 iters) and K=1152 (9 iters) exactly —
# no masking overhead at all for either case.
# Stores directly in the target dtype — no extra cast kernel.
# ---------------------------------------------------------------------------
@triton.jit
def matvec_row_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am,
    BLOCK_K: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc  = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k    = k0 + offs
        mask = k < K
        a    = tl.load(a_ptr + row * stride_am + k,
                       mask=mask, other=0.0).to(tl.float32)
        b    = tl.load(b_ptr + k,
                       mask=mask, other=0.0).to(tl.float32)
        acc += a * b

    result = tl.sum(acc, axis=0)

    if IS_FP16:
        tl.store(c_ptr + row, result.to(tl.float16))
    elif IS_BF16:
        tl.store(c_ptr + row, result.to(tl.bfloat16))
    else:
        tl.store(c_ptr + row, result)


# ---------------------------------------------------------------------------
# Lean Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_matvec(in_2, in_3):
    M, K, dt = in_2.shape[0], in_2.shape[1], in_2.dtype
    out = torch.empty((M, 1), dtype=dt, device=in_2.device)

    matvec_row_kernel[(M,)](
        in_2, in_3, out,
        M, K,
        in_2.stride(0),
        BLOCK_K=128,                     # divides K=768 & K=1152 exactly
        IS_FP16=(dt is torch.float16),
        IS_BF16=(dt is torch.bfloat16),
        num_warps=4,
        num_stages=4,                    # best pipeline depth from tuning
    )
    return out


# ---------------------------------------------------------------------------
# Replacement factory
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_matvec