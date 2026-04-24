import torch
import triton
import triton.language as tl
import inspect as _inspect


# ─── Pattern ────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, x):
    # Temporarily override silu's __signature__ so that ForceArgsTracer cannot
    # bind 'inplace=True' positionally.  When _force_empty_kwargs gets a
    # TypeError it falls back to the original (proxy,) / {'inplace': True} form,
    # which EXACTLY matches the kwargs stored in the Dynamo-compiled model graph.
    _orig_sig = getattr(torch.nn.functional.silu, '__signature__', None)
    torch.nn.functional.silu.__signature__ = _inspect.Signature([
        _inspect.Parameter('input', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    try:
        tmp_5 = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    finally:
        if _orig_sig is not None:
            torch.nn.functional.silu.__signature__ = _orig_sig
        else:
            try:
                del torch.nn.functional.silu.__signature__
            except AttributeError:
                pass
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, x):
    return (in_0, in_1, in_2, in_3, x)


# ─── Triton kernel ───────────────────────────────────────────────────────────
# Fused batch-norm (inference) + SiLU kernel.
# Grid: one program per channel (C).
# Each program processes all HW spatial elements for that channel.
# Reads input once, writes output once → 2× less bandwidth vs separate passes.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_silu_kernel(
    inp_ptr,     # [1, C, H, W] or [C, HW] — contiguous, channel-first
    mean_ptr,    # [C]
    var_ptr,     # [C]
    weight_ptr,  # [C]
    bias_ptr,    # [C]
    out_ptr,     # same shape/dtype as inp_ptr
    HW,          # H * W  (runtime scalar)
    eps,         # BN epsilon (runtime scalar)
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)

    # Per-channel BN parameters (computed in fp32 for numerical stability)
    mean    = tl.load(mean_ptr   + c).to(tl.float32)
    var     = tl.load(var_ptr    + c).to(tl.float32)
    w       = tl.load(weight_ptr + c).to(tl.float32)
    b       = tl.load(bias_ptr   + c).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Base pointer offset for this channel
    base = c * HW

    # Iterate over spatial elements in tiles of BLOCK_SIZE
    for start in tl.range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW

        x = tl.load(inp_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        # Batch-norm inference
        y = (x - mean) * inv_std * w + b
        # SiLU: y * sigmoid(y)
        out = y * tl.sigmoid(y)
        tl.store(out_ptr + base + offs, out, mask=mask)


# ─── Kernel wrapper ──────────────────────────────────────────────────────────
@torch.fx.wrap
def _fuse_bn_silu(in_0, in_1, in_2, in_3, x):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    x    : activation    [1, C, H, W]  (post-reshape, contiguous)
    """
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W

    device = x.device
    dtype  = x.dtype

    # Move BN params to device + correct dtype
    mean   = in_0.to(device=device, dtype=dtype)
    var    = in_1.to(device=device, dtype=dtype)
    weight = in_3.to(device=device, dtype=dtype)
    bias   = in_2.to(device=device, dtype=dtype)

    out = torch.empty_like(x)

    _fused_bn_silu_kernel[(C,)](
        x, mean, var, weight, bias, out,
        HW=HW,
        eps=1e-05,
    )

    return out


# ─── Replacement factory ─────────────────────────────────────────────────────
def replacement_func():
    return _fuse_bn_silu