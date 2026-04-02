import torch
import triton
import triton.language as tl


# ── Triton fused kernel ─────────────────────────────────────────────────────

@triton.jit
def fused_add_layernorm_768_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    # N and eps are compile-time constants for this 768-specific kernel
    N: tl.constexpr = 768
    eps: tl.constexpr = 1e-05

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load and fuse the element-wise add in fp32
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    # Mean (masked positions are 0, so they don't affect tl.sum)
    mean = tl.sum(z, axis=0) / N

    # Variance (zero out masked positions)
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Affine parameters
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr  + offs, mask=mask, other=0.0).to(tl.float32)

    result = diff * inv_std * w + b

    # Store, cast back to original dtype
    tl.store(out_ptr + row * stride_row + offs,
             result.to(x_ptr.dtype.element_ty), mask=mask)


# ── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_add_layernorm_768(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [768]
    in_1 : weight [768]
    in_2 : first  input [*, 768]
    in_3 : second input [*, 768]
    """
    N       = 768
    BLOCK_N = 1024          # next power-of-2 >= 768
    num_rows = in_2.numel() // N
    out = torch.empty_like(in_2)

    fused_add_layernorm_768_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out,
        N,              # stride_row
        BLOCK_N=BLOCK_N,
        num_warps=4,    # Triton tutorial recommendation for N≈1024
        num_stages=1,
    )
    return out


# ── FX pattern / replacement hooks ─────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_768