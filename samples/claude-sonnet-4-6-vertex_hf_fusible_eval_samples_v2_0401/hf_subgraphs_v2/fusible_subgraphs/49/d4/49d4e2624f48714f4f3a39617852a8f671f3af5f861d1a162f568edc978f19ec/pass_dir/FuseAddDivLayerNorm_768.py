import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match  (in_2 + in_3) / 2  followed by layer_norm(…,(768,),in_1,in_0,1e-12)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused Triton kernel
# BLOCK_SIZE is always next_power_of_2(768) = 1024.
# Autotune only searches over num_warps (no BLOCK_SIZE variation so the mask
# is always correct — OOB lanes are always zeroed).
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_div_ln_kernel(
    in2_ptr,
    in3_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    base = row * N

    # ---- fused (in2 + in3) * 0.5, compute in fp32 for precision -----------
    a = tl.load(in2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(in3_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = (a + b) * 0.5

    # ---- mean  (OOB lanes loaded as 0, so sum is exact) -------------------
    mean = tl.sum(x, 0) * (1.0 / N)

    # ---- centred values ----------------------------------------------------
    xc = x - mean

    # ---- variance  (zero OOB lanes before squaring; mask is compile-time) --
    mask_f = mask.to(tl.float32)
    xc_sq  = xc * xc * mask_f
    var    = tl.sum(xc_sq, 0) * (1.0 / N)

    # ---- normalise (hardware rsqrt) ----------------------------------------
    rstd = tl.rsqrt(var + eps)
    xn   = xc * rstd

    # ---- affine transform --------------------------------------------------
    w     = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b_val = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out   = xn * w + b_val

    # ---- store in original dtype -------------------------------------------
    tl.store(out_ptr + base + offs, out.to(DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper  — @torch.fx.wrap prevents FX from tracing into it
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_add_div_ln(in_0, in_1, in_2, in_3):
    """
    in_0 : layer-norm bias   [N]
    in_1 : layer-norm weight [N]
    in_2 : first  input tensor [..., N]
    in_3 : second input tensor [..., N]
    """
    N   = in_2.shape[-1]              # 768 for this pattern
    B   = in_2.numel() // N           # number of rows  (1 for [1,768])
    out = torch.empty_like(in_2)

    # BLOCK_SIZE fixed at next_power_of_2(N) so every element is covered;
    # the 256 extra lanes in 1024 are safely masked out.
    BLOCK_SIZE   = 1024               # triton.next_power_of_2(768)
    triton_dtype = tl.bfloat16 if in_2.dtype == torch.bfloat16 else tl.float16

    _fused_add_div_ln_kernel[(B,)](
        in_2, in_3, in_1, in_0, out,
        N, 1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=triton_dtype,
        num_warps=4,
        num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory that returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_add_div_ln