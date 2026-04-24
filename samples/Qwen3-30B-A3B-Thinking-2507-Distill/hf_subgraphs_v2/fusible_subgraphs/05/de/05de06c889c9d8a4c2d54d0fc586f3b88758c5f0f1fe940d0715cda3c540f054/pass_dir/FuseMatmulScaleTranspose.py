import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────
# Pattern: match matmul + scale (single output)
# The downstream .T node stays in the graph as a free view.
# ─────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    """Match: matmul(in_2, in_1) * in_0."""
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────
# Triton kernel – one CTA per output row.
# Accumulates in fp32; writes single output row at a time.
# Grid = (M,): M CTAs run simultaneously for maximum SM utilisation.
# ─────────────────────────────────────────────
@triton.jit
def _fused_matmul_scale_kernel(
    a_ptr,       # in_2: [M, K]  row-major
    b_ptr,       # in_1: [K, 1]  stride=(1,1)  → element k is at flat offset k
    scale_ptr,   # in_0: 0-d scalar
    out_ptr,     # tmp_1: [M, N] stride=(N, 1)
    M, K,
    BLOCK_K: tl.constexpr,   # tile size (power-of-2 >= K)
):
    m = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Row m of A: [BLOCK_K]
    a = tl.load(a_ptr + m * K + k_offs, mask=k_mask, other=0.0)
    # b: [BLOCK_K]
    b = tl.load(b_ptr + k_offs, mask=k_mask, other=0.0)

    # Dot product → fp32 accumulator (scalar)
    acc = tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)

    # Scale and cast back to input dtype
    result = (acc * tl.load(scale_ptr).to(tl.float32)).to(out_ptr.dtype.element_ty)

    # Store out[m, :]
    tl.store(out_ptr + m * K + k_offs, result, mask=k_mask)


# ─────────────────────────────────────────────
# Wrapper  (@torch.fx.wrap required by framework)
# ─────────────────────────────────────────────
@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """
    in_0 : scalar 0-d tensor  (logit_scale)
    in_1 : [K, 1]  column vector
    in_2 : [M, K]  left matrix
    Returns tmp_1 of shape [M, K]
    """
    M = in_2.shape[0]
    K = in_2.shape[1]

    out = torch.empty((M, K), dtype=in_2.dtype, device=in_2.device)

    BLOCK_K = max(16, triton.next_power_of_2(K))

    # One CTA per row – M CTAs can execute in parallel
    _fused_matmul_scale_kernel[(M,)](
        in_2, in_1, in_0,
        out,
        M, K,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_matmul_scale