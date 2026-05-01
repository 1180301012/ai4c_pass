import torch
import triton
import triton.language as tl


def pattern(in_6, in_5, in_2, in_1):
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_6, in_5, in_2, in_1):
    return (in_6, in_5, in_2, in_1)


# ── Kernel ──────────────────────────────────────────────────────────────────
# • N is tl.constexpr → compile-time mask, reciprocal multiply, unrolled tree.
# • BLOCK_SIZE=512    → next power-of-2 ≥ N=384.
# • num_warps=2       → 64 threads; 8 elements/thread = 16 B = 128-bit loads.
@triton.jit
def _fused_add_layer_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx   = tl.program_id(0)
    offsets   = tl.arange(0, BLOCK_SIZE)
    mask      = offsets < N
    row_start = row_idx * N

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    mean    = tl.sum(z, axis=0) * (1.0 / N)
    zm      = tl.where(mask, z - mean, 0.0)
    var     = tl.sum(zm * zm, axis=0) * (1.0 / N)
    inv_std = tl.rsqrt(var + 1e-12)
    z_norm  = zm * inv_std

    w   = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out = z_norm * w + b

    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm(in_6, in_5, weight, bias):
    N        = in_6.shape[-1]     # 384 — passed as constexpr
    num_rows = in_6.numel() // N  # 578

    out = torch.empty_like(in_6)

    _fused_add_layer_norm_kernel[(num_rows,)](
        in_6, in_5, weight, bias, out,
        N=N,
        BLOCK_SIZE=512,
        num_warps=2,
    )

    return out


def replacement_func():
    return fused_add_layer_norm