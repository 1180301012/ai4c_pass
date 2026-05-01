"""
Shared Triton kernel: fused residual-add + layer-norm.
Loaded via importlib in FuseAddLayerNorm_H*.py so all three pass files
share the SAME fused_add_ln function object (satisfies replacement_func_limit=1).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_ln_kernel(
    in0_ptr,   # [B, S, H]  residual (bf16/fp16)
    emb_ptr,   # [B, S, H]  positional-embed (bf16/fp16)
    w_ptr,     # [H]        LN weight
    b_ptr,     # [H]        LN bias
    out_ptr,   # [B, S, H]  output
    N,                        # hidden dim (runtime)
    eps,
    BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # fused residual add (both tensors are contiguous row-major, stride = N)
    x0 = tl.load(in0_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    xe = tl.load(emb_ptr  + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x  = x0 + xe

    # layer-norm
    xs   = tl.where(mask, x, 0.0)
    mean = tl.sum(xs, 0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, 0) / N
    istd = 1.0 / tl.sqrt(var + eps)
    xn   = (x - mean) * istd

    w  = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b  = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = xn * w + b

    if IS_BF16:
        tl.store(out_ptr + row * N + offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * N + offs, out.to(tl.float16),  mask=mask)


@torch.fx.wrap
def fused_add_ln(in_0, weight, bias, embed):
    """
    Replace: tmp_13 = in_0 + embed ; layer_norm(tmp_13, (H,), weight, bias, eps)
    """
    N    = in_0.shape[-1]
    rows = in_0.numel() // N
    out  = torch.empty_like(in_0)

    BLOCK   = triton.next_power_of_2(N)
    BLOCK   = max(BLOCK, 16)
    IS_BF16 = (in_0.dtype == torch.bfloat16)
    nw = 8 if BLOCK >= 512 else (4 if BLOCK >= 128 else (2 if BLOCK >= 32 else 1))

    _add_ln_kernel[(rows,)](
        in_0, embed, weight, bias, out,
        N, 1e-5,
        BLOCK=BLOCK, IS_BF16=IS_BF16,
        num_warps=nw,
    )
    return out