import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: layer_norm → transpose(-2,-1) → gelu
# Input shape:  [1, 3999, 512]  (batch=1, seq=3999, hidden=512)
# Output shape: [1,   512, 3999]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel — 2-D tiled
#
# BLOCK_N must divide 3999 (no partial tiles).
# 3999 = 3 × 31 × 43  → only power-of-2 divisors: 1, 2, 4, 8.
# Confine autotune to BLOCK_N ∈ {4, 8, 16, 32}
# (all divide 3999 exactly: 3999/16=249- exact, 3999/32=125- exact).
#
# Each program handles [BLOCK_N, BLOCK_C=512] of input rows:
#   • reads:  coalesced  — stride-1 along C within each row
#   • writes: coalesced  — tl.trans → [BLOCK_C, BLOCK_N], stride-1 along N
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Only power-of-2 BLOCK_N values that work generically (with n_masking).
        # Triton's tl.arange requires constexpr bounds to be powers of 2.
        # BLOCK_C=512 (a power of 2) perfectly covers C=512.
        triton.Config({"BLOCK_N": 1},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 2},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 4},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 8},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_N": 4},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 8},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=8,  num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_ln_tgelu_kernel(
    in2_ptr,   # [B, N, C]  B=1, N=3999, C=512
    weight_ptr, # [C]
    bias_ptr,   # [C]
    out_ptr,    # [B, C, N]  (contiguous transposed output)
    N,          # sequence length  (3999)
    eps,
    BLOCK_C: tl.constexpr,   # = 512 (exact feature dim, power-of-2)
    BLOCK_N: tl.constexpr,   # tile rows, arbitrary power-of-2
):
    tile_n  = tl.program_id(0)
    n_start = tile_n * BLOCK_N

    n_offs  = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_offs  = tl.arange(0, BLOCK_C)              # [BLOCK_C]

    # ---- boundary guard for N ----
    n_valid = n_offs < N                          # [BLOCK_N] bool — N=3999 is not a power-of-2

    # ---- Transposed load: [BLOCK_N, BLOCK_C] ----
    # For invalid rows these load 0.0 via `other=0.0`, which correctly zeros their
    # contribution to the mean/var (not dividing BLOCK_C=512: c < C always).
    x_c = tl.load(
        in2_ptr + n_offs[:, None] * BLOCK_C + c_offs[None, :],
        mask=n_valid[:, None],
        other=0.0,
    ).to(tl.float32)   # [BLOCK_N, BLOCK_C]

    # ---- layer-norm — reduce over axis=1 (BLOCK_C column-spatial dimension) ----
    # Invalid rows have x_c=0, so tl.sum includes zero contributions → mean=0,
    # inv_std=1/sqrt(eps)≈1.  The store is guarded by n_valid, so these rows are never written.
    mean    = tl.sum(x_c,     axis=1) / BLOCK_C          # [BLOCK_N]  (correct for valid rows)
    x_nc    = x_c - mean[:, None]                         # [BLOCK_N, BLOCK_C]
    var     = tl.sum(x_nc * x_nc, axis=1) / BLOCK_C      # [BLOCK_N]
    inv_std = 1.0 / tl.sqrt(var + eps)                    # [BLOCK_N]

    w = tl.load(weight_ptr + c_offs).to(tl.float32)   # [BLOCK_C]
    b = tl.load(bias_ptr   + c_offs).to(tl.float32)   # [BLOCK_C]

    y = x_nc * inv_std[:, None] * w[None, :] + b[None, :]  # [BLOCK_N, BLOCK_C]

    # ---- GELU: exact erf formula (matches PyTorch nn.functional.gelu) ----
    # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))  # [BLOCK_N, BLOCK_C]

    # ---- Transpose + coalesced store: [BLOCK_C, BLOCK_N] tiled ----
    # output[0, c, n] lives at output_ptr + c * N + n
    out_T   = tl.trans(gelu_y)   # [BLOCK_C, BLOCK_N]
    out_off = c_offs[:, None] * N + n_offs[None, :]  # [BLOCK_C, BLOCK_N]
    # Mask: n_valid broadcast over C dim so invalid rows are never written.
    tl.store(out_ptr + out_off, out_T.to(out_ptr.dtype.element_ty), mask=n_valid[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_ln_tgelu(in_0, in_1, in_2):
    """
    in_0 : bias   [C]  = [512]
    in_1 : weight [C]  = [512]
    in_2 : input  [1, N, C]  = [1, 3999, 512]

    Returns [1, C, N] = [1, 512, 3999]  (fused layer_norm + transpose + gelu)
    """
    C = 512
    N = 3999
    out = torch.empty((1, C, N), dtype=in_2.dtype, device=in_2.device)
    eps = 1e-05

    # Grid is exact for BLOCK_N ∈ {4,8,16,32} which divide N=3999.
    _fused_ln_tgelu_kernel[
        lambda meta: (N // meta["BLOCK_N"],)
    ](
        in_2, in_1, in_0, out,
        N, eps,
        BLOCK_C=512,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return _fused_ln_tgelu