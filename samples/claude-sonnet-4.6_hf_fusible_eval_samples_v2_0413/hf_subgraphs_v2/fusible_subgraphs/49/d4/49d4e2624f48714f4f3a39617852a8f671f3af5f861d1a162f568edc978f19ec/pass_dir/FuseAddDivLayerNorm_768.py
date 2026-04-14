import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches (in_2 + in_3) / 2 followed by layer_norm(…, (768,), …)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused Triton kernel: (x+y)*0.5  →  layer_norm(weight, bias)
#
# Variance via E[z²]−E[z]²: both sums computed from the already-in-register
# z_valid in a single vectorised pass (no second memory load needed).
# Autotuning finds the best num_warps for N=768 on this GPU.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=1,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
    ],
    key=[],      # N=768 is a compile-time constant; no runtime key needed
)
@triton.jit
def _fused_add_div_layernorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One CTA per row; N=768 is a compile-time constant (constexpr)."""
    N         = 768                   # constexpr: lets the compiler fold the mask
    row_idx   = tl.program_id(0)
    row_start = row_idx * N
    offsets   = tl.arange(0, BLOCK_SIZE)
    mask      = offsets < N           # static mask: first 768 always True

    # ---- load x, y ----------------------------------------------------------
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0)

    # ---- fused add + scale, matching PyTorch dtype rounding -----------------
    # PyTorch: tmp2 = x + y (original dtype),  tmp3 = tmp2 / 2
    z_half = (x + y) * 0.5          # stays in bf16/f16
    z      = z_half.to(tl.float32)  # upcast for stable reduction

    # ---- zero masked slots so they don't affect sums -----------------------
    z_valid = tl.where(mask, z, 0.0)

    # ---- single-pass mean and variance: E[z²] − E[z]² ----------------------
    sum_z  = tl.sum(z_valid,          axis=0)
    sum_z2 = tl.sum(z_valid * z_valid, axis=0)
    mean   = sum_z  / N
    var    = sum_z2 / N - mean * mean

    # ---- normalise ----------------------------------------------------------
    rstd   = 1.0 / tl.sqrt(var + eps)
    z_norm = tl.where(mask, (z - mean) * rstd, 0.0)

    # ---- affine: weight * z_norm + bias ------------------------------------
    w   = tl.load(w_ptr + offsets, mask=mask, other=1.0)
    b   = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = z_norm * w.to(tl.float32) + b.to(tl.float32)

    # ---- store (cast back to input dtype) ----------------------------------
    tl.store(out_ptr + row_start + offsets, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so the graph-rewriter can call it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_div_layernorm_768(in_0, in_1, in_2, in_3):
    """
    Fused: out = layer_norm((in_2 + in_3) / 2, weight=in_1, bias=in_0)

    in_0 : bias   [768]
    in_1 : weight [768]
    in_2 : [..., 768]
    in_3 : [..., 768]
    """
    out      = torch.empty_like(in_2)
    N        = in_2.shape[-1]       # 768
    num_rows = in_2.numel() // N    # 1 for reference shapes

    _fused_add_div_layernorm_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out,
        1e-12,          # eps  (N is now a compile-time constant inside kernel)
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory returning the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_add_div_layernorm_768