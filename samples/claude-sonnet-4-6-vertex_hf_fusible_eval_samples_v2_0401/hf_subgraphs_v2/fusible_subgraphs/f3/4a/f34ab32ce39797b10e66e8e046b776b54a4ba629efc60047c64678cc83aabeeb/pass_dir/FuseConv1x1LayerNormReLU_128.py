import torch
import triton
import triton.language as tl
import operator

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16},  num_warps=1),
        triton.Config({'BLOCK_C': 32},  num_warps=1),
        triton.Config({'BLOCK_C': 64},  num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
    ],
    key=['N', 'C_val'],
)
@triton.jit
def _triton_ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_val, eps,
    BLOCK_C: tl.constexpr,
):
    """Layer-norm only (no relu) — relu stays in original graph."""
    pid   = tl.program_id(0)
    c_off = tl.arange(0, BLOCK_C)
    mask  = c_off < C_val
    x    = tl.load(x_ptr + pid * C_val + c_off, mask=mask, other=0.0).to(tl.float32)
    x_m  = tl.where(mask, x, 0.0)
    mean = tl.sum(x_m) / C_val
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff) / C_val
    norm = diff * tl.rsqrt(var + eps)
    w    = tl.load(w_ptr + c_off, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + c_off, mask=mask, other=0.0).to(tl.float32)
    out  = norm * w + b       # NO relu — original relu node applies after
    tl.store(out_ptr + pid * C_val + c_off, out, mask=mask)


# Match ONLY layer_norm (all-positional args → ForceArgsTracer preserves structure).
# The original relu node stays in the graph and runs after the replacement.
# This avoids the inplace=True kwarg mismatch that blocks relu from matching.
def pattern(conv_out, ln_weight, ln_bias):
    return torch.nn.functional.layer_norm(conv_out, (128, 1, 1), ln_weight, ln_bias, 1e-05)


def replacement_args(conv_out, ln_weight, ln_bias):
    return (conv_out, ln_weight, ln_bias)


@torch.fx.wrap
def triton_layer_norm_128(conv_out, ln_weight, ln_bias):
    N   = conv_out.shape[0]
    C   = conv_out.shape[1]
    xf  = conv_out.contiguous().view(N, C)
    wf  = ln_weight.contiguous().view(C)
    bf  = ln_bias.contiguous().view(C)
    out = torch.empty((N, C), dtype=torch.float32, device=conv_out.device)
    _triton_ln_kernel[(N,)](xf, wf, bf, out, N, C, 1e-5)
    return out.to(conv_out.dtype).view_as(conv_out)


def replacement_func():
    return triton_layer_norm_128