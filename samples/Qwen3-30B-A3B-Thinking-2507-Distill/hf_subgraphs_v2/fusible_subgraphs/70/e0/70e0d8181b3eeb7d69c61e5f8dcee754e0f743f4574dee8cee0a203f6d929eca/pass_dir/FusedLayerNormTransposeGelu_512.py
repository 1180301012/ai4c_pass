import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ──────────────────────────────────────────────────────────────────────────────
# Fused LayerNorm + Transposed-Write + GELU  (one program per token)
#
# Write strategy: allocate output as contiguous [1, N, M], then each program
# pid (token m) writes:
#   out[n, m]  →  flat index  n*M + m
# for n = 0..N-1.  This is the correct address for a contiguous [1, N, M]
# tensor (strides [N*M, M, 1]).
#
# All shape constants (N=512, eps=1e-5) are tl.constexpr so the compiler can
# fold constants and eliminate dead branches.
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fused_ln_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M,                          # runtime: number of tokens (3999)
    eps,                        # runtime: 1e-5
    N:      tl.constexpr,       # 512  (fixed by the pattern)
    BLOCK:  tl.constexpr,       # 512  (same as N)
):
    pid    = tl.program_id(0)   # token index m  ∈ [0, M)
    base   = pid * N            # byte-offset of x[0, m, :]
    offs   = tl.arange(0, BLOCK)  # [0 … N-1]

    # ── Load input row (coalesced, N fp16 values) ──────────────────────────
    x  = tl.load(x_ptr + base + offs).to(tl.float32)

    # ── Load weight / bias (broadcast across all programs) ────────────────
    w  = tl.load(w_ptr + offs).to(tl.float32)
    b  = tl.load(b_ptr + offs).to(tl.float32)

    # ── Layer Norm ─────────────────────────────────────────────────────────
    rN  = 1.0 / N               # compile-time reciprocal (N is constexpr)
    mean  = tl.sum(x, axis=0) * rN
    x_c   = x - mean
    var   = tl.sum(x_c * x_c, axis=0) * rN
    rstd  = 1.0 / tl.sqrt(var + eps)
    x_hat = x_c * rstd
    y     = x_hat * w + b

    # ── Strided store to contiguous [1, N, M] output ──────────────────────
    # out[0, n, m]  →  flat index  n*M + pid
    # Coalesced store to natural [1, M, N] layout instead:
    # out[0, m, n]  →  flat index  m*N + n
    tl.store(out_ptr + pid * N + offs, y.to(x.dtype))


@torch.fx.wrap
def fused_ln_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [N]
    in_1 : weight [N]
    in_2 : input  [1, M, N]
    returns : [1, M, N]  layer_norm output (coalesced), then
              PyTorch applies transpose (free view) + GELU naturally.
    """
    N = in_2.shape[-1]         # 512
    M = in_2.numel() // N      # 3999

    # Allocate in NATURAL [1, M, N] order → coalesced reads AND writes.
    out = torch.empty((1, M, N), dtype=in_2.dtype, device=in_2.device)

    _fused_ln_gelu_kernel[(M,)](
        x_ptr=in_2,
        w_ptr=in_1,
        b_ptr=in_0,
        out_ptr=out,
        M=M,
        eps=1e-5,
        N=512,
        BLOCK=512,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_ln_transpose_gelu