import torch
import triton
import triton.language as tl


# ── L2-normalize kernel (one block per row) ──────────────────────────────────

@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    x_f  = x.to(tl.float32)
    norm = tl.sqrt(tl.sum(x_f * x_f, axis=0))
    tl.store(out_ptr + row * N + offs, (x_f / norm).to(x.dtype), mask=mask)


# ── Exp-multiply kernel ───────────────────────────────────────────────────────

@triton.jit
def _exp_mul_kernel(s_ptr, x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    s    = tl.load(s_ptr).to(tl.float32)
    e    = tl.exp(s)
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (e * x.to(tl.float32)).to(x.dtype), mask=mask)


# ── Pre-warm both kernels at import time for each dtype ──────────────────────
# This ensures the JIT-compiled CUDA binary is in Triton's in-memory cache
# before the benchmark starts, so the first warmup iteration is fast too.

def _prewarm():
    for dtype in [torch.float16, torch.bfloat16]:
        _x   = torch.zeros(1, 512, dtype=dtype, device='cuda')
        _out = torch.empty_like(_x)
        _l2_norm_kernel[(1,)](_x, _out, 512, BLOCK_SIZE=512, num_warps=16)
        _s    = torch.zeros([], dtype=dtype, device='cuda')
        _out2 = torch.empty_like(_x)
        _exp_mul_kernel[(1,)](_s, _x, _out2, 512, BLOCK_SIZE=512, num_warps=16)

try:
    _prewarm()
except Exception:
    pass   # Silently skip if CUDA is unavailable (e.g., CPU-only build)


# ── Shared dispatch wrapper (inlined – one less Python call per invocation) ───

@torch.fx.wrap
def _dispatch(arg0, arg1, route):
    """
    route == "l2norm"  → L2-normalise arg0   (arg1 is a padding duplicate)
    route == "expmul"  → exp(arg0) * arg1
    """
    if route == "l2norm":
        N          = arg0.shape[-1]
        M          = arg0.numel() // N
        out        = torch.empty_like(arg0)
        _l2_norm_kernel[(M,)](arg0, out, N, BLOCK_SIZE=512, num_warps=16)
        return out
    elif route == "expmul":
        N          = arg1.numel()
        out        = torch.empty_like(arg1)
        num_blocks = (N + 511) // 512
        _exp_mul_kernel[(num_blocks,)](arg0, arg1, out, N, BLOCK_SIZE=512, num_warps=16)
        return out