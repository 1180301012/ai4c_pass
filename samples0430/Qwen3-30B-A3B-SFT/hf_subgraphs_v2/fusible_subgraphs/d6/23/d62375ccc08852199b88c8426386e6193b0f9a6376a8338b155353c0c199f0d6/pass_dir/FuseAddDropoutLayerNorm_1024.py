import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1024}, num_warps=4),
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
        triton.Config({'BLOCK_H': 1024}, num_warps=16),
    ],
    key=['H'],
)
@triton.jit
def _fused_add_layer_norm_1024_kernel(
    in_0_ptr, in_1_ptr, weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    B, H, eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)

    base = row * H

    # Load inputs as float32 for numerical stability
    x = tl.load(in_0_ptr + base + offsets).to(tl.float32)
    y = tl.load(in_1_ptr + base + offsets).to(tl.float32)

    # Add (dropout with training=False is identity)
    z = x + y

    # Store the add result (out_add)
    tl.store(out_add_ptr + base + offsets, z)

    # Layer norm: compute mean and variance
    mean = tl.sum(z, axis=0) / H
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    z_norm = diff * inv_std

    # Load scale and shift
    w = tl.load(weight_ptr + offsets).to(tl.float32)
    b = tl.load(bias_ptr + offsets).to(tl.float32)

    out = z_norm * w + b

    # Store the layer norm result (out_ln)
    tl.store(out_ln_ptr + base + offsets, out)


@torch.fx.wrap
def fused_add_layer_norm_1024(in_0, in_1, weight, bias):
    # in_0: [B, T, H], in_1: [B, T, H], weight: [H], bias: [H]
    H = 1024
    B = in_0.numel() // H   # B*T rows
    eps = 1e-5

    out_add = torch.empty_like(in_0)
    out_ln = torch.empty_like(in_0)

    grid = (B,)

    _fused_add_layer_norm_1024_kernel[grid](
        in_0, in_1, weight, bias,
        out_add, out_ln,
        B, H, eps,
    )

    return out_add, out_ln


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, weight, bias):
    added = in_0 + in_1
    dropped = torch.nn.functional.dropout(added, p=0.1, training=False)
    normed = torch.nn.functional.layer_norm(dropped, (1024,), weight, bias, 1e-05)
    return dropped, normed


def replacement_args(in_0, in_1, weight, bias):
    return (in_0, in_1, weight, bias)


def replacement_func():
    return fused_add_layer_norm_1024