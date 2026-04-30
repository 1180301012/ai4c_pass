import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    offs = tl.arange(0, BLOCK_SIZE)

    if BLOCK_SIZE == N:
        # No masking needed - optimized path for power-of-2 N
        z = (tl.load(x_ptr + base + offs).to(tl.float32) +
             tl.load(y_ptr + base + offs).to(tl.float32))
        mean = tl.sum(z, axis=0) / N
        z = z - mean
        rstd = 1.0 / tl.sqrt(tl.sum(z * z, axis=0) / N + 1e-5)
        z = z * rstd * tl.load(w_ptr + offs).to(tl.float32) + tl.load(b_ptr + offs).to(tl.float32)
        tl.store(out_ptr + base + offs, z)
    else:
        # Masked path for non-power-of-2 N
        mask = offs < N
        z = (tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32) +
             tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32))
        mean = tl.sum(z, axis=0) / N
        z = tl.where(mask, z - mean, 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(z * z, axis=0) / N + 1e-5)
        z = z * rstd * tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32) + tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + offs, z, mask=mask)


@torch.fx.wrap
def fused_add_layernorm(x, y, weight, bias):
    n = x.shape[-1]
    out = torch.empty_like(x)
    _fused_add_layernorm_kernel[(x.numel() // n,)](
        x, y, weight, bias, out,
        N=n,
        BLOCK_SIZE=1 << (n - 1).bit_length(),
        num_warps=2 if n >= 32 else 1,
    )
    return out