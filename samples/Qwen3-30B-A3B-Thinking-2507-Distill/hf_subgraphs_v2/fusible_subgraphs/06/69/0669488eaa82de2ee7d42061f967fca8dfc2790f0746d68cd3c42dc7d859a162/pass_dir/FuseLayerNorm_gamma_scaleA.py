"""
Pass A: Fuse relu + flatten + l2norm + scale + clamp + div + mul
Matches graphs with scale = 0.14433756729740643 (spatial dims 8x6 -> D=48).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: must mirror model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: one program per row, computes row-wise L2 normalisation
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _fused_ln_A_kernel(
    in0_ptr,   # gamma scalar [1]
    in1_ptr,   # input [N, D]
    out_ptr,   # output [N, D]
    D,         # number of elements per row (= H * W)
    BLOCK_D: tl.constexpr,  # next power-of-2 >= D
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # ---- load & relu -------------------------------------------------------
    x = tl.load(in1_ptr + row * D + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)

    # ---- L2 norm (promote to fp32 for accuracy) ----------------------------
    x_f32 = x.to(tl.float32)
    norm_sq = tl.sum(x_f32 * x_f32, axis=0)
    # clamp to avoid division by zero
    norm_sq  = tl.maximum(norm_sq, 1e-10)
    inv_norm = tl.rsqrt(norm_sq)

    # ---- load gamma scalar, multiply by inv_norm ---------------------------
    gamma = tl.load(in0_ptr).to(tl.float32)
    g_norm = gamma * inv_norm          # scalar

    # ---- normalize & store -------------------------------------------------
    x_norm = x_f32 * g_norm            # broadcast over [BLOCK_D]
    # Triton auto-casts fp32 → fp16/bf16 when target ptr has smaller dtype
    tl.store(out_ptr + row * D + offs, x_norm, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX does not trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_ln_A(in_0, in_1):
    """
    in_0 : gamma  shape [1]
    in_1 : input  shape [B, 133, H, W]
    returns: normalized output, same shape as in_1
    """
    # After flatten(2): [B, 133, H*W]  →  N = B*133,  D = H*W
    B       = in_1.shape[0]
    N       = B * 133
    D       = in_1.shape[-1] * in_1.shape[-2]
    BLOCK_D = triton.next_power_of_2(D)

    out = torch.empty_like(in_1)

    _fused_ln_A_kernel[(N,)](
        in_0,
        in_1.reshape(N, D),
        out.reshape(N, D),
        D,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg factory returning the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_ln_A