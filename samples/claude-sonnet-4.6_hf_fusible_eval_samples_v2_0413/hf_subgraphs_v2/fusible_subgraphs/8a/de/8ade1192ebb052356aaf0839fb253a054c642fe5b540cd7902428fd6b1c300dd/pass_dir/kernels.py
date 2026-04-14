import torch
import triton
import triton.language as tl


# ─── Attention Mask Kernel ───────────────────────────────────────────────────

@triton.jit
def _attn_mask_kernel(in5_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(in5_ptr + offsets, mask=mask, other=1)
    result = tl.where(x == 1, 0.0, -3.4028234663852886e+38)
    tl.store(out_ptr + offsets, result.to(tl.float32), mask=mask)


@torch.fx.wrap
def triton_attention_mask(in_5):
    N = in_5.numel()
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _attn_mask_kernel[grid](in_5, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ─── LayerNorm Kernel (replaces layer_norm + identity dropout) ───────────────

@triton.jit
def _layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    D, eps,
    BLOCK_D: tl.constexpr,
    IS_BF16: tl.constexpr,
    NW: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # Load in original dtype then upcast for FP32 computation
    x_raw = tl.load(x_ptr + row_id * D + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    val = tl.where(mask, x_raw.to(tl.float32), 0.0)
    mean = tl.sum(val, axis=0) / D
    diff = tl.where(mask, val - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    out_f32 = diff * rstd * w + b

    if IS_BF16:
        out = out_f32.to(tl.bfloat16)
    else:
        out = out_f32.to(tl.float16)
    tl.store(out_ptr + row_id * D + offsets, out, mask=mask)


def _run_layernorm(x, weight, bias, D, BLOCK_D, nw):
    N_ROWS = x.numel() // D
    is_bf16 = x.dtype == torch.bfloat16
    out = torch.empty_like(x)
    _layernorm_kernel[(N_ROWS,)](
        x, weight, bias, out,
        D, 1e-5,
        BLOCK_D=BLOCK_D,
        IS_BF16=is_bf16,
        NW=nw,
        num_warps=nw,
    )
    return out


@torch.fx.wrap
def kernel_wrapper(a, b, c, d, route):
    if route == "ln_16":
        return _run_layernorm(a, c, d, 16, 16, 1)
    elif route == "ln_768":
        return _run_layernorm(a, c, d, 768, 1024, 8)
    else:  # ln_1024
        return _run_layernorm(a, c, d, 1024, 1024, 8)