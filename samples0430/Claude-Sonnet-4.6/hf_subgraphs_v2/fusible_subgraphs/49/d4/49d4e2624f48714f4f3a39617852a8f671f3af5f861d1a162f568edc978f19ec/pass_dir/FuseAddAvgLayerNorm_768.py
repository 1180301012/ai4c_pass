import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: (in_2 + in_3) / 2  |>  layer_norm(normalized_shape=(768,), eps=1e-12)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """
    in_0 = bias [768], in_1 = weight [768]
    in_2, in_3 = activations [B, 768]
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused add-average + layer-norm
#
# Design decisions for a [1, 768] workload on NVIDIA A30:
#   - One Triton program per row (B programs total)
#   - BLOCK_SIZE = 1024  (next power-of-2 >= 768; 256 positions are masked)
#   - Compute entirely in float32 for numerical stability
#   - Use tl.math.rsqrt for a single-instruction reciprocal sqrt
#   - No autotune: for tiny B the Python lookup overhead exceeds the savings
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_avg_layernorm_kernel(
    in2_ptr,
    in3_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,   # must be >= 768 and a power of 2
):
    # N=768 hardcoded: folds 1/768 at compile time;
    # mask offsets<768 is a compile-time predicate.
    row_start = tl.program_id(0) * 768
    offsets   = tl.arange(0, BLOCK_SIZE)
    mask      = offsets < 768

    # ---- load inputs and form the per-element average -----------------
    x = (tl.load(in2_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
       + tl.load(in3_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)) * 0.5

    # ---- mean and E[x²] in two passes over in-register data ----------
    # Padding zeros (other=0.0) contribute 0 to both sums, so no masking needed.
    sum_x  = tl.sum(x,     axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    mean    = sum_x  * (1.0 / 768.0)
    # var = E[x²] − E[x]²  (stable for near-zero mean data as here)
    var     = sum_x2 * (1.0 / 768.0) - mean * mean
    inv_std = tl.math.rsqrt(var + 1e-12)

    # ---- affine rescale ----------------------------------------------
    # padded positions: (x-mean) is wrong but mask=False → never stored
    w   = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (x - mean) * inv_std * w + b

    # ---- store (Triton auto-casts fp32 → bf16/f16 on write) ---------
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_avg_layernorm(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [768]
    in_1 : weight [768]
    in_2 : first  activation [B, 768]
    in_3 : second activation [B, 768]
    """
    B   = in_2.numel() // 768
    out = torch.empty_like(in_2)

    fused_add_avg_layernorm_kernel[(B,)](
        in_2, in_3, in_1, in_0, out,
        BLOCK_SIZE=1024,
        num_warps=1,    # one warp → purely intra-warp shuffle reduction (5 levels, no cross-warp sync)
        num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_add_avg_layernorm