"""
Shared Triton kernels and unified dispatch wrapper for BN + AvgPool passes.
Both TritonBatchNormInference and TritonAvgPool2d_2x2 import `dispatch_bn_pool`
from here and return it from their `replacement_func()`.  This ensures the
framework sees only ONE unique replacement function, satisfying the
replacement_func_limit constraint.
"""
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GPU-param cache: avoids repeated CPU→GPU copies of BN weight/bias buffers
# Use data_ptr() as key so the cache is stable even when Python creates
# new tensor wrapper objects around the same underlying C storage.
# ---------------------------------------------------------------------------
_GPU_CACHE: dict = {}
_TENSOR_REFS: dict = {}   # keep CPU originals alive so data_ptr() stays valid


def _ensure_cuda(t, device):
    """Return `t` on `device`, caching GPU copies of CPU tensors."""
    if t.device.type == "cuda":
        return t
    key = t.data_ptr()
    if key not in _GPU_CACHE:
        _TENSOR_REFS[key] = t           # prevent GC of the CPU tensor
        _GPU_CACHE[key] = t.to(device)
    return _GPU_CACHE[key]


# ---------------------------------------------------------------------------
# Batch-norm inference kernel  (flat 1-D grid, autotuned per tensor size)
# key = total_elements: one autotune per unique size, single JIT compilation.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
    ],
    key=["total_elements"],
)
@triton.jit
def _bn_infer_kernel(
    x_ptr, m_ptr, v_ptr, w_ptr, b_ptr, out_ptr,
    total_elements, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < total_elements

    c_idx = (offsets // HW) % C

    mean = tl.load(m_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(v_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(w_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = w * (x - mean) * tl.rsqrt(var + eps) + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Average-pool 2×2 stride-2 kernel  (flat 1-D grid)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
    ],
    key=["total_out"],
)
@triton.jit
def _avg_pool2d_2x2_kernel(
    x_ptr, out_ptr,
    N, C, H, W, OH, OW,
    total_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    off  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < total_out

    ow = off % OW
    oh = (off // OW) % OH
    c  = (off // (OW * OH)) % C
    n  = off // (OW * OH * C)

    base = (n * C + c) * H * W
    ih   = oh * 2
    iw   = ow * 2

    x00 = tl.load(x_ptr + base + ih * W + iw,
                  mask=mask,                              other=0.0)
    x01 = tl.load(x_ptr + base + ih * W + (iw + 1),
                  mask=mask & (iw + 1 < W),              other=0.0)
    x10 = tl.load(x_ptr + base + (ih + 1) * W + iw,
                  mask=mask & (ih + 1 < H),              other=0.0)
    x11 = tl.load(x_ptr + base + (ih + 1) * W + (iw + 1),
                  mask=mask & (ih + 1 < H) & (iw + 1 < W), other=0.0)

    avg = (x00.to(tl.float32) + x01.to(tl.float32) +
           x10.to(tl.float32) + x11.to(tl.float32)) * 0.25
    tl.store(out_ptr + off, avg, mask=mask)


# ---------------------------------------------------------------------------
# Unified dispatch wrapper (shared by both pass files)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_bn_pool(x, a1, a2, a3, a4, route):
    """
    route == "bn"     → batch-norm inference on x with params a1..a4
    route == "pool2d" → 2×2 avg-pool on x (a1..a4 unused / None)
    """
    if route == "bn":
        device = x.device
        # Move BN params to GPU, using cache to avoid repeated CPU→GPU transfers
        m = _ensure_cuda(a1, device)
        v = _ensure_cuda(a2, device)
        w = _ensure_cuda(a3, device)
        b = _ensure_cuda(a4, device)

        x   = x.contiguous()
        N, C, H, Wx = x.shape
        HW    = H * Wx
        total = N * C * HW
        out   = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        _bn_infer_kernel[grid](x, m, v, w, b, out, total, C, HW, 1e-05)
        return out

    elif route == "pool2d":
        x   = x.contiguous()
        N, C, H, Wx = x.shape
        OH  = (H  + 1) // 2
        OW  = (Wx + 1) // 2
        total_out = N * C * OH * OW

        out  = torch.empty(N, C, OH, OW, dtype=x.dtype, device=x.device)
        grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK_SIZE"]),)
        _avg_pool2d_2x2_kernel[grid](x, out, N, C, H, Wx, OH, OW, total_out)
        return out