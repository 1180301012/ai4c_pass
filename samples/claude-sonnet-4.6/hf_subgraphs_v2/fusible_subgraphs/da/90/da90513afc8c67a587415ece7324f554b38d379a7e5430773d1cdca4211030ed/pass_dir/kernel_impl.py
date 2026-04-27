import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}),
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
    ],
    key=['N', 'BLOCK_SIZE'],
)
@triton.jit
def fused_add_layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Out_ptr,
    N, inv_N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Add + LayerNorm kernel.
    One program per row; BLOCK_SIZE (power-of-2) >= N so the full row fits.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # ── Load inputs (promote to fp32 for accuracy) ──────────────────────────
    x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)

    # ── Fused element-wise add ───────────────────────────────────────────────
    z = x + y   # masked lanes already 0.0 due to other=0.0

    # ── Mean ─────────────────────────────────────────────────────────────────
    mean = tl.sum(z, axis=0) * inv_N   # masked lanes contribute 0 → correct

    # ── Variance ─────────────────────────────────────────────────────────────
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) * inv_N

    # ── Normalize ────────────────────────────────────────────────────────────
    inv_std = tl.rsqrt(var + eps)
    z_norm  = diff * inv_std

    # ── Affine transform ─────────────────────────────────────────────────────
    w   = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = z_norm * w + b

    # ── Store (Triton auto-casts fp32 → destination dtype) ───────────────────
    tl.store(Out_ptr + row_offset + cols, out, mask=mask)


# ── Shared dispatch wrapper (imported by ALL pass files) ─────────────────────
@torch.fx.wrap
def fused_add_layernorm_dispatch(x, y, weight, bias, route):
    """
    Route string selects the (N, BLOCK_SIZE) specialisation at runtime.
    All pass files return this single function from replacement_func(),
    so the framework never drops passes due to replacement_func_limit.
    """
    out = torch.empty_like(x)
    if route == "N768":
        N = 768
        M = x.numel() // N
        fused_add_layernorm_kernel[(M,)](
            x, y, weight, bias, out,
            N=N, inv_N=1.0 / N, eps=1e-5,
            BLOCK_SIZE=1024,
        )
    elif route == "N1024":
        N = 1024
        M = x.numel() // N
        fused_add_layernorm_kernel[(M,)](
            x, y, weight, bias, out,
            N=N, inv_N=1.0 / N, eps=1e-5,
            BLOCK_SIZE=1024,
        )
    elif route == "N16":
        N = 16
        M = x.numel() // N
        fused_add_layernorm_kernel[(M,)](
            x, y, weight, bias, out,
            N=N, inv_N=1.0 / N, eps=1e-5,
            BLOCK_SIZE=16,
        )
    return out