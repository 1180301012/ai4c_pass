import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused batched-matmul + reshape for output width D=16
# in_1 : [K, M, N_INNER]   (N_INNER compile-time constant = 9)
# in_0 : [K, N_INNER, 1]
# result: [K*M // D, D]    (D=16)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['KM'],
)
@triton.jit
def _bmm_reshape_kernel_16(
    in1_ptr, in0_ptr, out_ptr,
    K, M, KM,
    N_INNER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < KM

    k_idx = offs // M
    m_idx = offs % M

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for n in tl.static_range(N_INNER):
        # in_1[k, m, n]  strides: (M*N_INNER, N_INNER, 1)
        in1_offs = k_idx * (M * N_INNER) + m_idx * N_INNER + n
        # in_0[k, n, 0]  strides: (N_INNER, 1, 1)  -> k*N_INNER + n
        in0_offs = k_idx * N_INNER + n

        x = tl.load(in1_ptr + in1_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(in0_ptr + in0_offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc + x * w

    # store – Triton auto-casts fp32 → pointer element type (fp16 / bf16)
    tl.store(out_ptr + offs, acc, mask=mask)


@torch.fx.wrap
def fused_matmul_reshape_16(in_0, in_1):
    """
    Fused replacement for:
        matmul = torch.matmul(in_1, in_0)
        out    = torch.reshape(matmul, [-1, 16])
    in_0: [K, 9, 1],  in_1: [K, M, 9]
    returns: [K*M // 16, 16]
    """
    K = in_1.shape[0]
    M = in_1.shape[1]
    D = 16
    N_INNER = 9
    KM = K * M

    out = torch.empty((KM // D, D), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: ((KM + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _bmm_reshape_kernel_16[grid](
        in_1, in_0, out,
        K=K, M=M, KM=KM,
        N_INNER=N_INNER,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pass API
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_matmul_reshape_16